#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_H_

#include <cstdint>
#include <string>
#include <vector>

namespace mini_llama {

// GGML 类型(子集)
enum GgmlType : uint32_t {
  kGgmlTypeF32 = 0,
  kGgmlTypeF16 = 1,
  kGgmlTypeQ40 = 2,
  kGgmlTypeQ41 = 3,
  kGgmlTypeQ80 = 8,
};

// 将 GGML 类型编号转换为可读名称；未知编号返回 UNKNOWN。
const char* GgmlTypeName(uint32_t type);

// GGUF 值类型
enum GgufValueType : uint32_t {
  kGgufTypeUint8 = 0,
  kGgufTypeInt8 = 1,
  kGgufTypeUint16 = 2,
  kGgufTypeInt16 = 3,
  kGgufTypeUint32 = 4,
  kGgufTypeInt32 = 5,
  kGgufTypeFloat32 = 6,
  kGgufTypeBool = 7,
  kGgufTypeString = 8,
  kGgufTypeArray = 9,
  kGgufTypeUint64 = 10,
  kGgufTypeInt64 = 11,
  kGgufTypeFloat64 = 12,
};

// 将 GGUF 元数据类型编号转换为可读名称；未知编号返回 unknown。
const char* GgufValueTypeName(uint32_t type);

// GGUF 张量信息
struct GgufTensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  uint32_t type = 0;    // GGML 类型
  uint64_t offset = 0;  // 相对于数据段起始位置的偏移
};

// GGUF 元数据键值对
struct GgufMetadataKv {
  std::string key;
  uint32_t type = 0;      // GGUF 值类型
  std::string value_str;  // 人类可读表示
};

// 最小 GGUF 读取器(v3)
struct GgufReader {
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  std::vector<GgufMetadataKv> metadata;
  std::vector<GgufTensorInfo> tensors;
  uint64_t data_offset = 0;  // 张量数据开始的绝对文件偏移

  // 加载并解析 GGUF 文件。出错时返回 false（错误信息在 load_error 中）
  bool Load(const std::string& path);

  std::string load_error;
};

// 将 GGUF 内容打印到 stdout（用于 inspect-gguf）
void InspectGguf(const GgufReader& reader);

// ---------------------------------------------------------------
// 张量数据读取
// ---------------------------------------------------------------

// 从 GGUF 文件数据段读取原始张量字节
// 出错时返回空 vector
std::vector<uint8_t> ReadGgufTensorRaw(const std::string& path,
                                       const GgufTensorInfo& info,
                                       uint64_t data_offset);

// 根据张量的 GGML 类型和形状计算其数据的字节大小。
bool GgufTensorDataSize(const GgufTensorInfo& info, uint64_t& out_size);

// ---------------------------------------------------------------
// 类型化元数据读取（重新打开文件并定位到该键）
// 如果找到且类型匹配，则返回 true
// ---------------------------------------------------------------
bool GgufGetMetadataInt(const std::string& path, const std::string& key,
                        int64_t& out);
bool GgufGetMetadataFloat(const std::string& path, const std::string& key,
                          double& out);
bool GgufGetMetadataString(const std::string& path, const std::string& key,
                           std::string& out);
bool GgufGetMetadataStringArray(const std::string& path, const std::string& key,
                                std::vector<std::string>& out);
bool GgufGetMetadataIntArray(const std::string& path, const std::string& key,
                             std::vector<int32_t>& out);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_H_
