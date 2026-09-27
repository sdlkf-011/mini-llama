#include "mini_llama/gguf.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

namespace mini_llama {

// 将 GGML 类型编号转换为开发者快照中使用的短名称
const char* GgmlTypeName(uint32_t type) {
  switch (type) {
    case kGgmlTypeF32:
      return "F32";
    case kGgmlTypeF16:
      return "F16";
    case kGgmlTypeQ40:
      return "Q4_0";
    case kGgmlTypeQ41:
      return "Q4_1";
    case kGgmlTypeQ80:
      return "Q8_0";
    default:
      return "UNKNOWN";
  }
}

// 将 GGUF 元数据类型编号转换为可读名称，未知编号统一标记为 unknown
const char* GgufValueTypeName(uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
      return "uint8";
    case kGgufTypeInt8:
      return "int8";
    case kGgufTypeUint16:
      return "uint16";
    case kGgufTypeInt16:
      return "int16";
    case kGgufTypeUint32:
      return "uint32";
    case kGgufTypeInt32:
      return "int32";
    case kGgufTypeFloat32:
      return "float32";
    case kGgufTypeBool:
      return "bool";
    case kGgufTypeString:
      return "string";
    case kGgufTypeArray:
      return "array";
    case kGgufTypeUint64:
      return "uint64";
    case kGgufTypeInt64:
      return "int64";
    case kGgufTypeFloat64:
      return "float64";
    default:
      return "unknown";
  }
}

namespace {

constexpr uint32_t kGgufVersion = 3;
constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;

// 从小端 GGUF 流中读取一个定长标量；短读统一返回失败
template <typename T>
bool ReadLe(std::ifstream& file, T& value) {
  char bytes[sizeof(T)];
  if (!file.read(bytes, sizeof(bytes))) {
    return false;
  }
  std::memcpy(&value, bytes, sizeof(value));
  return true;
}

// 读取 GGUF 使用的 uint32 标量
bool ReadU32(std::ifstream& file, uint32_t& value) { return ReadLe(file, value); }

// 读取 GGUF 使用的 uint64 标量
bool ReadU64(std::ifstream& file, uint64_t& value) { return ReadLe(file, value); }

// 读取带 uint64 长度前缀的字符串，并限制分配规模以抵御损坏文件
bool ReadString(std::ifstream& file, std::string& value) {
  uint64_t length = 0;
  if (!ReadU64(file, length) || length > kMaxStringBytes ||
      length > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  value.resize(static_cast<size_t>(length));
  return length == 0 || static_cast<bool>(file.read(
                            value.data(), static_cast<std::streamsize>(length)));
}

// 校验并读取 GGUF 文件头，避免后续解析在错误版本或超大计数上运行
bool ReadHeader(std::ifstream& file, uint64_t& tensor_count,
                uint64_t& metadata_count) {
  char magic[4] = {};
  uint32_t version = 0;
  if (!file.read(magic, sizeof(magic)) || std::memcmp(magic, "GGUF", 4) != 0 ||
      !ReadU32(file, version) || version != kGgufVersion ||
      !ReadU64(file, tensor_count) || !ReadU64(file, metadata_count)) {
    return false;
  }
  return tensor_count <= kMaxTensorEntries && metadata_count <= kMaxMetadataEntries;
}

// 跳过任意 GGUF 元数据值，递归处理数组并对元素数量设置上限
bool SkipValue(std::ifstream& file, uint32_t type) {
  switch (type) {
    case kGgufTypeUint8:
    case kGgufTypeInt8:
    case kGgufTypeBool: {
      uint8_t ignored = 0;
      return ReadLe(file, ignored);
    }
    case kGgufTypeUint16:
    case kGgufTypeInt16: {
      uint16_t ignored = 0;
      return ReadLe(file, ignored);
    }
    case kGgufTypeUint32:
    case kGgufTypeInt32:
    case kGgufTypeFloat32: {
      uint32_t ignored = 0;
      return ReadU32(file, ignored);
    }
    case kGgufTypeString: {
      std::string ignored;
      return ReadString(file, ignored);
    }
    case kGgufTypeUint64:
    case kGgufTypeInt64:
    case kGgufTypeFloat64: {
      uint64_t ignored = 0;
      return ReadU64(file, ignored);
    }
    case kGgufTypeArray: {
      uint32_t element_type = 0;
      uint64_t count = 0;
      if (!ReadU32(file, element_type) || !ReadU64(file, count) ||
          count > kMaxArrayElements || element_type == kGgufTypeArray) {
        return false;
      }
      for (uint64_t index = 0; index < count; ++index) {
        if (!SkipValue(file, element_type)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

// 在无符号加法前检查溢出，用于文件偏移和数据范围计算
bool AddChecked(uint64_t first, uint64_t second, uint64_t& result) {
  if (second > std::numeric_limits<uint64_t>::max() - first) {
    return false;
  }
  result = first + second;
  return true;
}

// 在无符号乘法前检查溢出，用于维度和字节大小计算
bool MultiplyChecked(uint64_t first, uint64_t second, uint64_t& result) {
  if (first != 0 && second > std::numeric_limits<uint64_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

// 计算张量元素总数；零维度和整数溢出均视为非法张量
bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& result) {
  result = 1;
  for (uint64_t dimension : tensor.shape) {
    if (dimension == 0 || !MultiplyChecked(result, dimension, result)) {
      return false;
    }
  }
  return true;
}

// 按项目支持的 GGML 子集计算张量原始字节数
bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& result) {
  uint64_t elements = 0;
  if (!TensorNumel(tensor, elements)) {
    return false;
  }
  switch (tensor.type) {
    case kGgmlTypeF32:
      return MultiplyChecked(elements, 4, result);
    case kGgmlTypeF16:
      return MultiplyChecked(elements, 2, result);
    case kGgmlTypeQ40:
      if (elements % kQuantBlockSize != 0) {
        return false;
      }
      return MultiplyChecked(elements / kQuantBlockSize, 18, result);
    case kGgmlTypeQ41:
      if (elements % kQuantBlockSize != 0) {
        return false;
      }
      return MultiplyChecked(elements / kQuantBlockSize, 20, result);
    case kGgmlTypeQ80:
      if (elements % kQuantBlockSize != 0) {
        return false;
      }
      return MultiplyChecked(elements / kQuantBlockSize, 34, result);
    default:
      // 未知类型没有可安全推导的布局；保留元数据读取能力并报告零字节
      result = 0;
      return true;
  }
}

// 将一个元数据标量读取成人类可读文本，数组只保留类型和长度摘要
bool ReadValueAsString(std::ifstream& file, uint32_t type, std::string& value) {
  std::ostringstream text;
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << static_cast<unsigned>(v);
      break;
    }
    case kGgufTypeInt8: {
      int8_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << static_cast<int>(v);
      break;
    }
    case kGgufTypeUint16: {
      uint16_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeInt16: {
      int16_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeUint32: {
      uint32_t v;
      if (!ReadU32(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeInt32: {
      int32_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeFloat32: {
      float v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeBool: {
      uint8_t v;
      if (!ReadLe(file, v) || v > 1) {
        return false;
      }
      text << (v ? "true" : "false");
      break;
    }
    case kGgufTypeString:
      return ReadString(file, value);
    case kGgufTypeUint64: {
      uint64_t v;
      if (!ReadU64(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeInt64: {
      int64_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeFloat64: {
      double v;
      if (!ReadLe(file, v)) {
        return false;
      }
      text << v;
      break;
    }
    case kGgufTypeArray: {
      uint32_t element_type = 0;
      uint64_t count = 0;
      if (!ReadU32(file, element_type) || !ReadU64(file, count) ||
          count > kMaxArrayElements || element_type == kGgufTypeArray) {
        return false;
      }
      text << "[" << count << " x " << GgufValueTypeName(element_type) << "]";
      for (uint64_t index = 0; index < count; ++index) {
        if (!SkipValue(file, element_type)) {
          return false;
        }
      }
      break;
    }
    default:
      return false;
  }
  value = text.str();
  return true;
}

// 解析 general.alignment 元数据，并验证其为不超过上限的二次幂
uint64_t ParseAlignment(const std::vector<GgufMetadataKv>& metadata) {
  uint64_t alignment = kDefaultAlignment;
  for (const GgufMetadataKv& item : metadata) {
    if (item.key != "general.alignment") {
      continue;
    }
    try {
      size_t parsed = 0;
      const unsigned long long candidate = std::stoull(item.value_str, &parsed);
      if (parsed != item.value_str.size()) {
        return 0;
      }
      alignment = static_cast<uint64_t>(candidate);
    } catch (const std::exception&) {
      return 0;
    }
  }

  if (alignment == 0 || alignment > kMaxAlignment ||
      (alignment & (alignment - 1)) != 0) {
    return 0;
  }
  return alignment;
}

// 将当前位置向 alignment 上界对齐，并检查加法溢出
bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 || AddChecked(value, alignment - remainder, result);
}

// 获取文件长度并将输入流复位到文件起点，供原始数据范围校验使用
bool GetFileSize(std::ifstream& file, uint64_t& size) {
  file.seekg(0, std::ios::end);
  const std::streampos end = file.tellg();
  if (end < 0) {
    return false;
  }
  size = static_cast<uint64_t>(end);
  file.seekg(0, std::ios::beg);
  return static_cast<bool>(file);
}

using MetadataVisitor = std::function<bool(std::ifstream&, uint32_t)>;

// 打开 GGUF 并扫描元数据，在目标键处交给类型化访问器读取值
bool ScanMetadata(const std::string& path, const std::string& key,
                  const MetadataVisitor& visitor) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }
  uint64_t tensor_count = 0, metadata_count = 0;
  if (!ReadHeader(file, tensor_count, metadata_count)) {
    return false;
  }

  for (uint64_t index = 0; index < metadata_count; ++index) {
    std::string current_key;
    uint32_t type = 0;
    if (!ReadString(file, current_key) || !ReadU32(file, type)) {
      return false;
    }
    if (current_key == key) {
      return visitor(file, type);
    }
    if (!SkipValue(file, type)) {
      return false;
    }
  }
  return false;
}

// 读取整数元数据并转换为 int64，拒绝超出目标范围的无符号值
bool ReadIntegerValue(std::ifstream& file, uint32_t type, int64_t& value) {
  switch (type) {
    case kGgufTypeUint8: {
      uint8_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      value = v;
      return true;
    }
    case kGgufTypeInt8: {
      int8_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      value = v;
      return true;
    }
    case kGgufTypeUint16: {
      uint16_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      value = v;
      return true;
    }
    case kGgufTypeInt16: {
      int16_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      value = v;
      return true;
    }
    case kGgufTypeUint32: {
      uint32_t v;
      if (!ReadU32(file, v)) {
        return false;
      }
      value = v;
      return true;
    }
    case kGgufTypeInt32: {
      int32_t v;
      if (!ReadLe(file, v)) {
        return false;
      }
      value = v;
      return true;
    }
    case kGgufTypeUint64: {
      uint64_t v;
      if (!ReadU64(file, v) ||
          v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
      }
      value = static_cast<int64_t>(v);
      return true;
    }
    case kGgufTypeInt64:
      return ReadLe(file, value);
    default:
      return false;
  }
}

// 读取浮点元数据，并统一转换为 double
bool ReadFloatValue(std::ifstream& file, uint32_t type, double& value) {
  if (type == kGgufTypeFloat32) {
    float v;
    if (!ReadLe(file, v)) {
      return false;
    }
    value = v;
    return true;
  }
  if (type == kGgufTypeFloat64) {
    return ReadLe(file, value);
  }
  return false;
}

// 读取字符串数组元数据，要求数组元素类型确实为 string
bool ReadStringArrayValue(std::ifstream& file, uint32_t type,
                          std::vector<std::string>& values) {
  uint32_t element_type = 0;
  uint64_t count = 0;
  if (type != kGgufTypeArray || !ReadU32(file, element_type) ||
      !ReadU64(file, count) || element_type != kGgufTypeString ||
      count > kMaxArrayElements || count > std::numeric_limits<size_t>::max()) {
    return false;
  }
  values.clear();
  values.reserve(static_cast<size_t>(count));

  for (uint64_t index = 0; index < count; ++index) {
    std::string value;
    if (!ReadString(file, value)) {
      return false;
    }
    values.push_back(std::move(value));
  }
  return true;
}

// 读取整数数组元数据，逐项检查转换到 int32 的范围
bool ReadIntArrayValue(std::ifstream& file, uint32_t type,
                       std::vector<int32_t>& values) {
  uint32_t element_type = 0;
  uint64_t count = 0;
  if (type != kGgufTypeArray || !ReadU32(file, element_type) ||
      !ReadU64(file, count) || element_type == kGgufTypeArray ||
      count > kMaxArrayElements || count > std::numeric_limits<size_t>::max()) {
    return false;
  }
  values.clear();
  values.reserve(static_cast<size_t>(count));

  for (uint64_t index = 0; index < count; ++index) {
    int64_t value = 0;
    if (!ReadIntegerValue(file, element_type, value) ||
        value < std::numeric_limits<int32_t>::min() ||
        value > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    values.push_back(static_cast<int32_t>(value));
  }
  return true;
}

}  // namespace

// 读取并校验 GGUF v3 头、元数据、张量信息及数据段边界
bool GgufReader::Load(const std::string& path) {
  // 每次加载都从干净状态开始，避免复用对象时残留上一次文件内容
  version = 0;
  n_tensors = 0;
  n_metadata = 0;
  metadata.clear();
  tensors.clear();
  data_offset = 0;
  load_error.clear();
  auto fail = [this](const std::string& message) {
    // 底层读取器只保存错误，不直接打印，调用者可决定如何展示错误
    load_error = message;
    return false;
  };

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    load_error = "failed to open: " + path;
    return false;
  }

  // magic
  char magic[4];
  if (!file.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
    load_error = "invalid GGUF magic";
    return false;
  }

  // version
  if (!ReadU32(file, version)) {
    load_error = "failed to read version";
    return false;
  }
  if (version != kGgufVersion) {
    load_error = "unsupported GGUF version: " + std::to_string(version);
    return false;
  }

  // n_tensors, n_metadata
  if (!ReadU64(file, n_tensors) || !ReadU64(file, n_metadata)) {
    load_error = "failed to read header counts";
    return false;
  }
  if (n_tensors > kMaxTensorEntries) {
    load_error = "too many tensors: " + std::to_string(n_tensors);
    return false;
  }
  if (n_metadata > kMaxMetadataEntries) {
    load_error = "too many metadata entries: " + std::to_string(n_metadata);
    return false;
  }

  // Metadata
  metadata.reserve(static_cast<size_t>(n_metadata));
  std::unordered_set<std::string> metadata_keys;
  for (uint64_t index = 0; index < n_metadata; ++index) {
    // 元数据值转成摘要文本，完整数组只在专用 getter 中再次精确读取
    GgufMetadataKv item;
    if (!ReadString(file, item.key)) {
      load_error =
          "failed to read metadata key at FlatIndex " + std::to_string(index);
      return false;
    }
    if (!metadata_keys.insert(item.key).second) {
      load_error = "duplicate metadata key: " + item.key;
      return false;
    }
    if (!ReadU32(file, item.type)) {
      load_error = "failed to read metadata type for key: " + item.key;
      return false;
    }
    if (!ReadValueAsString(file, item.type, item.value_str)) {
      load_error = "failed to read metadata value for key: " + item.key;
      return false;
    }
    metadata.push_back(std::move(item));
  }

  // Tensor info
  tensors.reserve(static_cast<size_t>(n_tensors));
  for (uint64_t index = 0; index < n_tensors; ++index) {
    // 读取张量描述后立即计算其理论大小，尽早拒绝非法维度和不支持类型
    GgufTensorInfo tensor;
    uint32_t dimensions = 0;
    if (!ReadString(file, tensor.name)) {
      load_error =
          "failed to read tensor name at FlatIndex " + std::to_string(index);
      return false;
    }
    if (!ReadU32(file, dimensions)) {
      load_error = "failed to read tensor n_dims for: " + tensor.name;
      return false;
    }
    if (dimensions == 0 || dimensions > kMaxTensorDims) {
      load_error = "invalid tensor n_dims for: " + tensor.name;
      return false;
    }
    tensor.shape.resize(dimensions);

    for (uint32_t dimension_index = 0; dimension_index < dimensions;
         ++dimension_index) {
      if (!ReadU64(file, tensor.shape[dimension_index])) {
        load_error = "failed to read tensor shape for: " + tensor.name;
        return false;
      }
      if (tensor.shape[dimension_index] == 0) {
        load_error = "invalid zero tensor dimension for: " + tensor.name;
        return false;
      }
    }

    if (!ReadU32(file, tensor.type)) {
      load_error = "failed to read tensor type for: " + tensor.name;
      return false;
    }
    if (!ReadU64(file, tensor.offset)) {
      load_error = "failed to read tensor offset for: " + tensor.name;
      return false;
    }
    tensors.push_back(std::move(tensor));
  }

  std::unordered_set<std::string> tensor_names;
  for (const GgufTensorInfo& tensor : tensors) {
    if (!tensor_names.insert(tensor.name).second) {
      load_error = "duplicate tensor name: " + tensor.name;
      return false;
    }
  }

  // 计算数据偏移量（对齐于 general.alignment，默认 32）
  const std::streampos current = file.tellg();
  if (current < 0) {
    load_error = "failed to determine GGUF data offset";
    return false;
  }
  data_offset = static_cast<uint64_t>(current);
  const uint64_t alignment = ParseAlignment(metadata);
  if (alignment == 0) {
    load_error = "invalid general.alignment";
    return false;
  }
  const uint64_t padding = (alignment - (data_offset % alignment)) % alignment;
  if (!AddChecked(data_offset, padding, data_offset)) {
    load_error = "GGUF data offset overflow";
    return false;
  }

  file.seekg(0, std::ios::end);
  const std::streampos end = file.tellg();
  if (end < 0 || static_cast<uint64_t>(end) < data_offset) {
    load_error = "GGUF file ended before tensor data section";
    return false;
  }
  const uint64_t file_size = static_cast<uint64_t>(end);
  for (const GgufTensorInfo& tensor : tensors) {
    uint64_t tensor_size = 0;
    if (!TensorDataSize(tensor, tensor_size)) {
      load_error = "invalid tensor data size for: " + tensor.name;
      return false;
    }
    if (tensor_size == 0) {
      continue;
    }
    uint64_t tensor_start = 0;
    if (!AddChecked(data_offset, tensor.offset, tensor_start)) {
      load_error = "tensor offset overflow for: " + tensor.name;
      return false;
    }
    uint64_t tensor_end = 0;
    if (!AddChecked(tensor_start, tensor_size, tensor_end)) {
      load_error = "tensor data range overflow for: " + tensor.name;
      return false;
    }
    if (tensor_end > file_size) {
      load_error = "tensor data outside file bounds: " + tensor.name;
      return false;
    }
  }
  return true;
}

// 将读取器中的头信息、元数据摘要和张量范围打印到 stdout
void InspectGguf(const GgufReader& reader) {
  // 该函数专门负责开发者可视化输出；Load 本身保持无副作用、无终端输出
  std::cout << "GGUF version=" << reader.version << " tensors=" << reader.n_tensors
            << " metadata=" << reader.n_metadata
            << " data_offset=" << reader.data_offset << "\n";

  for (const GgufMetadataKv& item : reader.metadata) {
    // value_str 已由 Load 生成，数组以数量和元素类型摘要展示，避免大量输出
    std::cout << "metadata " << item.key << " (" << GgufValueTypeName(item.type)
              << "): " << item.value_str << "\n";
  }

  for (const GgufTensorInfo& tensor : reader.tensors) {
    // 形状、类型、相对偏移和理论字节数构成张量快照的最小有用信息
    std::cout << "tensor " << tensor.name << " shape=[";
    for (size_t index = 0; index < tensor.shape.size(); ++index) {
      if (index != 0) {
        std::cout << ",";
      }
      std::cout << tensor.shape[index];
    }
    uint64_t size = 0;
    std::cout << "] type=" << GgmlTypeName(tensor.type)
              << " offset=" << tensor.offset
              << " bytes=" << (TensorDataSize(tensor, size) ? size : 0) << "\n";
  }
}

// 从文件数据段读取指定张量的原始字节，并在读取前完成所有范围检查
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset) {
  // 先计算大小和绝对偏移，任何溢出或未知布局都在打开文件前失败
  uint64_t size = 0, absolute_offset = 0;
  if (!TensorDataSize(info, size) ||
      !AddChecked(data_offset, info.offset, absolute_offset) ||
      size > std::numeric_limits<size_t>::max()) {
    return {};
  }
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    return {};
  }

  // 文件大小检查确保 seek/read 不会访问数据段之外的区域
  uint64_t file_size = 0;
  if (!GetFileSize(file, file_size) || absolute_offset > file_size ||
      size > file_size - absolute_offset ||
      absolute_offset > std::numeric_limits<std::streamoff>::max() ||
      size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    return {};
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(size));

  // 读取失败返回空 vector；成功时返回与 GGUF 原始布局完全一致的字节序列
  file.seekg(static_cast<std::streamoff>(absolute_offset), std::ios::beg);
  if (!file || (size != 0 && !file.read(reinterpret_cast<char*>(bytes.data()),
                                        static_cast<std::streamsize>(size)))) {
    return {};
  }
  return bytes;
}

// 暴露支持的 GGML 类型字节大小计算，并保持失败时输出值为零
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size) {
  out_size = 0;
  return TensorDataSize(info, out_size);
}

// 查找并读取整数类型元数据
bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out) {
  // ScanMetadata 负责定位键，访问器只处理整数类型并执行范围转换
  return ScanMetadata(path, key, [&out](std::ifstream& file, uint32_t type) {
    return ReadIntegerValue(file, type, out);
  });
}

// 查找并读取 float32 或 float64 类型元数据
bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out) {
  // 仅接受 GGUF 的 float32/float64，整数不会被隐式当作浮点元数据
  return ScanMetadata(path, key, [&out](std::ifstream& file, uint32_t type) {
    return ReadFloatValue(file, type, out);
  });
}

// 查找并读取字符串类型元数据
bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out) {
  // 类型不匹配时访问器直接失败，避免把摘要文本误当成原始字符串
  return ScanMetadata(path, key, [&out](std::ifstream& file, uint32_t type) {
    return type == kGgufTypeString && ReadString(file, out);
  });
}

// 查找并读取 string 数组元数据。
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out) {
  // 数组读取器会校验元素类型、数量上限以及每个字符串的长度前缀
  return ScanMetadata(path, key, [&out](std::ifstream& file, uint32_t type) {
    return ReadStringArrayValue(file, type, out);
  });
}

// 查找并读取可转换为 int32 的整数数组元数据
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out) {
  // 每个整数都检查 int32 范围，防止窄化转换产生静默截断
  return ScanMetadata(path, key, [&out](std::ifstream& file, uint32_t type) {
    return ReadIntArrayValue(file, type, out);
  });
}

}  // namespace mini_llama
