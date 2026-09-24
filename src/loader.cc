#include "mini_llama/loader.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// 用于解析模型配置与张量元数据的轻量 JSON 辅助实现
// 当前只支持约定的 model.json 固定格式
namespace mini_llama {

static MiniLlamaModel LoadFailure(const std::string& message) {
  // 失败时返回可检查状态的空模型，不把异常继续传播给命令行调用方
  MiniLlamaModel model;
  model.loaded = false;
  model.load_error = message;
  std::cerr << "Failed to load model: " << message << std::endl;
  return model;
}

static size_t FindConfigKey(const std::string& content, const std::string& key) {
  // 固定格式中配置键全局唯一，定位键名后再寻找其后的冒号和值起点
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    throw std::runtime_error("missing config key: " + key);
  }

  // 固定配置不允许同名键重复，避免读取结果依赖“第一个值”规则
  if (content.find(marker, key_pos + marker.size()) != std::string::npos) {
    throw std::runtime_error("duplicate config key: " + key);
  }

  size_t colon_pos = content.find(':', key_pos + marker.size());
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("missing ':' after config key: " + key);
  }

  // 跳过冒号后的 JSON 空白，返回数值文本的第一个字符位置
  size_t value_pos = colon_pos + 1;
  while (value_pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[value_pos]))) {
    ++value_pos;
  }
  if (value_pos >= content.size()) {
    throw std::runtime_error("missing value for config key: " + key);
  }
  return value_pos;
}

static bool IsJsonValueTerminator(char character) {
  // 数值后只允许空白、逗号、对象结束或数组结束符
  return std::isspace(static_cast<unsigned char>(character)) || character == ',' ||
         character == '}' || character == ']';
}

static void ValidateNumberTerminator(const std::string& content, size_t value_pos,
                                     size_t parsed, const std::string& key) {
  // stoi/stof 允许部分转换，因此需要确认已解析数字后没有非法尾随字符
  size_t position = value_pos + parsed;
  while (position < content.size() &&
         std::isspace(static_cast<unsigned char>(content[position]))) {
    ++position;
  }

  if (position < content.size() && !IsJsonValueTerminator(content[position])) {
    throw std::runtime_error("invalid numeric value for config key: " + key);
  }
}

static int ParseRequiredInt(const std::string& content, const std::string& key) {
  // 必需整数字段先定位，再使用 stoi 完成格式与 int 范围检查
  size_t value_pos = FindConfigKey(content, key);
  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for config key: " + key);
  }

  if (parsed == 0) {
    throw std::runtime_error("invalid integer for config key: " + key);
  }

  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

static std::string TrimWs(const std::string& text) {
  // shape 中逗号分隔的维度可能带换行和缩进，解析前统一去除两端空白
  size_t start = text.find_first_not_of(" \t\n\r");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = text.find_last_not_of(" \t\n\r");
  return text.substr(start, end - start + 1);
}

static int ParseStrictIntToken(const std::string& token, const std::string& field) {
  // strtoll 可以区分无数字、范围溢出和带非法后缀三类错误
  std::string trimmed = TrimWs(token);
  if (trimmed.empty()) {
    throw std::runtime_error("missing integer value for " + field);
  }

  char* end = nullptr;
  errno = 0;

  int64_t value = std::strtoll(trimmed.c_str(), &end, 10);
  if (end == trimmed.c_str() || errno == ERANGE ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid integer value for " + field + ": " + trimmed);
  }

  while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }

  if (*end != '\0') {
    throw std::runtime_error("invalid integer value for " + field + ": " + trimmed);
  }

  return static_cast<int>(value);
}

static size_t ParseStrictSizeValue(const std::string& content, size_t value_pos,
                                   const std::string& field) {
  // offset 和 byte_size 不允许负数，并且必须能由当前平台的 size_t 表示
  while (value_pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[value_pos]))) {
    ++value_pos;
  }

  if (value_pos >= content.size() || content[value_pos] == '-') {
    throw std::runtime_error("invalid non-negative integer for " + field);
  }

  size_t parsed = 0;
  uint64_t value = 0;
  try {
    value = std::stoull(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid non-negative integer for " + field);
  }

  if (parsed == 0 || value > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("invalid non-negative integer for " + field);
  }
  ValidateNumberTerminator(content, value_pos, parsed, field);
  return static_cast<size_t>(value);
}

static float ParseRequiredFloat(const std::string& content, const std::string& key) {
  // 必需浮点字段允许普通小数和科学计数法，有限性在配置校验阶段检查
  size_t value_pos = FindConfigKey(content, key);
  size_t parsed = 0;
  float value = 0.0f;
  try {
    value = std::stof(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid float for config key: " + key);
  }

  if (parsed == 0) {
    throw std::runtime_error("invalid float for config key: " + key);
  }
  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

static size_t FindOptionalKey(const std::string& content, const std::string& key) {
  // 可选字段不存在时返回 npos，其余语法错误仍然抛出异常
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    return std::string::npos;
  }

  size_t colon_pos = content.find(':', key_pos + marker.size());
  if (colon_pos == std::string::npos) {
    throw std::runtime_error("missing ':' after key: " + key);
  }

  size_t value_pos = colon_pos + 1;
  while (value_pos < content.size() &&
         std::isspace(static_cast<unsigned char>(content[value_pos]))) {
    ++value_pos;
  }

  if (value_pos >= content.size()) {
    throw std::runtime_error("missing value for key: " + key);
  }
  return value_pos;
}

static std::string ParseOptionalStringField(const std::string& content,
                                            const std::string& key,
                                            const std::string& fallback) {
  // 当前固定格式中的 tokenizer 字符串不包含转义引号，可直接寻找结束引号
  size_t value_pos = FindOptionalKey(content, key);
  if (value_pos == std::string::npos) {
    return fallback;
  }
  if (content[value_pos] != '"') {
    throw std::runtime_error("expected string value for key: " + key);
  }

  size_t end = content.find('"', value_pos + 1);
  if (end == std::string::npos) {
    throw std::runtime_error("unterminated string value for key: " + key);
  }
  return content.substr(value_pos + 1, end - value_pos - 1);
}

static int ParseOptionalIntField(const std::string& content, const std::string& key,
                                 int fallback) {
  // tokenizer 的特殊 token id 缺省时保留 TokenizerInfo 中的默认值
  size_t value_pos = FindOptionalKey(content, key);
  if (value_pos == std::string::npos) {
    return fallback;
  }

  size_t parsed = 0;
  int value = 0;
  try {
    value = std::stoi(content.substr(value_pos), &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer for key: " + key);
  }

  if (parsed == 0) {
    throw std::runtime_error("invalid integer for key: " + key);
  }
  ValidateNumberTerminator(content, value_pos, parsed, key);
  return value;
}

static std::string ParseObjectForKey(const std::string& content,
                                     const std::string& key) {
  // 从指定键后的第一个左花括号开始，通过深度计数截取完整对象
  const std::string marker = "\"" + key + "\"";
  size_t key_pos = content.find(marker);
  if (key_pos == std::string::npos) {
    return "";
  }

  size_t object_start = content.find('{', key_pos + marker.size());
  if (object_start == std::string::npos) {
    throw std::runtime_error("missing object for key: " + key);
  }

  int depth = 0;
  for (size_t position = object_start; position < content.size(); ++position) {
    if (content[position] == '{') {
      ++depth;
    } else if (content[position] == '}') {
      --depth;
      if (depth == 0) {
        return content.substr(object_start, position - object_start + 1);
      }
    }
  }
  throw std::runtime_error("unclosed object for key: " + key);
}

static TokenizerInfo ParseTokenizerJson(const std::string& content) {
  // tokenizer 对象整体可缺省，此时使用 ASCII tokenizer 默认配置
  TokenizerInfo tokenizer;
  std::string object = ParseObjectForKey(content, "tokenizer");
  if (object.empty()) {
    return tokenizer;
  }

  // 每个 tokenizer 字段均允许独立缺省
  tokenizer.type = ParseOptionalStringField(object, "type", tokenizer.type);
  tokenizer.path = ParseOptionalStringField(object, "path", tokenizer.path);
  tokenizer.bos_id = ParseOptionalIntField(object, "bos_id", tokenizer.bos_id);
  tokenizer.eos_id = ParseOptionalIntField(object, "eos_id", tokenizer.eos_id);
  tokenizer.unk_id = ParseOptionalIntField(object, "unk_id", tokenizer.unk_id);

  // 当前项目只支持 ASCII 和外部 JSON 词表两种 tokenizer
  if (tokenizer.type != "ascii" && tokenizer.type != "json_vocab") {
    throw std::runtime_error("unsupported tokenizer type: " + tokenizer.type);
  }
  if (tokenizer.bos_id < 0 || tokenizer.eos_id < 0 || tokenizer.unk_id < 0) {
    throw std::runtime_error("tokenizer special ids must be non-negative");
  }
  if (tokenizer.bos_id == tokenizer.eos_id || tokenizer.bos_id == tokenizer.unk_id ||
      tokenizer.eos_id == tokenizer.unk_id) {
    throw std::runtime_error("tokenizer special ids must be distinct");
  }
  if (tokenizer.type == "json_vocab" && tokenizer.path.empty()) {
    throw std::runtime_error("json_vocab tokenizer requires a path");
  }
  return tokenizer;
}

static void ValidateConfig(const ModelConfig& config) {
  // 所有维度都参与内存分配、循环边界或除法，必须严格为正数
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.max_seq_len <= 0) {
    throw std::runtime_error("all model dimensions must be positive");
  }
  if (config.dim % config.n_heads != 0) {
    throw std::runtime_error("dim must be divisible by n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("n_heads must be divisible by n_kv_heads");
  }

  // RoPE 每次处理一对分量，因此单头维度必须是正偶数
  if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
    throw std::runtime_error("head_dim must be a positive even number");
  }
  if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
    throw std::runtime_error("rope_theta must be positive");
  }
  if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
    throw std::runtime_error("rms_norm_eps must be positive");
  }
}

static size_t CheckedTensorByteSize(const TensorInfo& info) {
  // 逐维相乘并在每次乘法前检查 size_t 溢出
  size_t num_elements = 1;
  for (int dimension : info.shape) {
    if (dimension <= 0) {
      throw std::runtime_error("invalid tensor shape for " + info.name);
    }

    const size_t dimension_size = static_cast<size_t>(dimension);
    if (num_elements > std::numeric_limits<size_t>::max() / dimension_size) {
      throw std::runtime_error("tensor size overflow for " + info.name);
    }
    num_elements *= dimension_size;
  }

  if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("tensor byte size overflow for " + info.name);
  }
  return num_elements * sizeof(float);
}

static std::vector<std::pair<std::string, std::vector<int>>> ExpectedTensorShapes(
    const ModelConfig& config) {
  // 先登记模型级张量，再按层生成标准名称与期望形状
  std::vector<std::pair<std::string, std::vector<int>>> expected = {
      {"token_embedding", {config.vocab_size, config.dim}},
      {"final_norm", {config.dim}},
      {"lm_head", {config.vocab_size, config.dim}},
  };

  for (int layer = 0; layer < config.n_layers; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    expected.push_back({prefix + "attention_norm", {config.dim}});
    expected.push_back(
        {prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
    expected.push_back(
        {prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
    expected.push_back({prefix + "ffn_norm", {config.dim}});
    expected.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
    expected.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
  }
  return expected;
}

static std::string ShapeToString(const std::vector<int>& shape) {
  // 统一错误信息和 inspect 输出中的 shape 表示
  std::string result = "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index > 0) {
      result += ", ";
    }
    result += std::to_string(shape[index]);
  }
  result += "]";
  return result;
}

static ModelConfig ParseConfigJson(const std::string& content) {
  // 配置字段全部为必需项，head_dim 由 dim 和 n_heads 推导
  ModelConfig config;
  config.vocab_size = ParseRequiredInt(content, "vocab_size");
  config.dim = ParseRequiredInt(content, "dim");
  config.hidden_dim = ParseRequiredInt(content, "hidden_dim");
  config.n_layers = ParseRequiredInt(content, "n_layers");
  config.n_heads = ParseRequiredInt(content, "n_heads");
  config.n_kv_heads = ParseRequiredInt(content, "n_kv_heads");
  config.max_seq_len = ParseRequiredInt(content, "max_seq_len");
  config.rope_theta = ParseRequiredFloat(content, "rope_theta");
  config.rms_norm_eps = ParseRequiredFloat(content, "rms_norm_eps");

  // 避免非法 n_heads 在正式校验前触发除零
  config.head_dim = config.n_heads > 0 ? config.dim / config.n_heads : 0;
  ValidateConfig(config);
  return config;
}

// 解析 model.json 中的 tensors 数组
static std::vector<TensorInfo> ParseTensorMetadata(const std::string& content) {
  std::vector<TensorInfo> result;

  // 先定位 tensors 数组的起始位置
  const std::string tensors_marker = "\"tensors\"";
  size_t tensors_pos = content.find(tensors_marker);
  if (tensors_pos == std::string::npos) {
    throw std::runtime_error("missing 'tensors' field in model.json");
  }

  size_t bracket_open = content.find('[', tensors_pos);
  if (bracket_open == std::string::npos) {
    throw std::runtime_error("missing '[' after 'tensors'");
  }

  // depth 用于区分外层 tensors 数组和每个 shape 子数组
  size_t position = bracket_open + 1;
  int depth = 1;

  while (position < content.size() && depth > 0) {
    // 跳过空白和逗号，寻找下一个张量对象
    while (position < content.size() && content[position] != '{') {
      if (content[position] == '[') {
        ++depth;
      } else if (content[position] == ']') {
        --depth;
      }
      ++position;
    }
    if (depth <= 0 || position >= content.size()) {
      break;
    }

    // 固定格式的张量对象内部不包含嵌套对象，因此第一个右花括号即为结束
    TensorInfo info;
    size_t object_start = position;
    size_t object_end = content.find('}', object_start);
    if (object_end == std::string::npos) {
      throw std::runtime_error("unclosed tensor object");
    }
    std::string object = content.substr(object_start, object_end - object_start + 1);

    bool has_name = false;
    bool has_dtype = false;
    bool has_shape = false;
    bool has_offset = false;
    bool has_byte_size = false;

    // 解析张量名称
    size_t name_pos = object.find("\"name\"");
    if (name_pos != std::string::npos) {
      size_t name_colon = object.find(':', name_pos);
      if (name_colon != std::string::npos) {
        size_t name_quote = object.find('"', name_colon + 1);
        size_t name_end = name_quote == std::string::npos
                              ? std::string::npos
                              : object.find('"', name_quote + 1);
        if (name_quote != std::string::npos && name_end != std::string::npos) {
          info.name = object.substr(name_quote + 1, name_end - name_quote - 1);
          has_name = true;
        }
      }
    }

    // 解析张量数据类型
    size_t dtype_pos = object.find("\"dtype\"");
    if (dtype_pos != std::string::npos) {
      size_t dtype_colon = object.find(':', dtype_pos);
      if (dtype_colon != std::string::npos) {
        size_t dtype_quote = object.find('"', dtype_colon + 1);
        size_t dtype_end = dtype_quote == std::string::npos
                               ? std::string::npos
                               : object.find('"', dtype_quote + 1);
        if (dtype_quote != std::string::npos && dtype_end != std::string::npos) {
          info.dtype = object.substr(dtype_quote + 1, dtype_end - dtype_quote - 1);
          has_dtype = true;
        }
      }
    }

    // 解析 shape 数组，并严格检查每个维度都是完整 int
    size_t shape_pos = object.find("\"shape\"");
    if (shape_pos != std::string::npos) {
      size_t shape_bracket = object.find('[', shape_pos);
      size_t shape_end = shape_bracket == std::string::npos
                             ? std::string::npos
                             : object.find(']', shape_bracket);
      if (shape_bracket != std::string::npos && shape_end != std::string::npos) {
        std::string shape_text =
            object.substr(shape_bracket + 1, shape_end - shape_bracket - 1);
        std::stringstream stream(shape_text);
        std::string token;
        while (std::getline(stream, token, ',')) {
          info.shape.push_back(ParseStrictIntToken(token, "tensor shape"));
        }
        has_shape = true;
      }
    }

    // 解析 model.bin 中的字节偏移
    size_t offset_pos = object.find("\"offset\"");
    if (offset_pos != std::string::npos) {
      size_t offset_colon = object.find(':', offset_pos);
      if (offset_colon == std::string::npos) {
        throw std::runtime_error("tensor offset field is malformed");
      }
      info.offset = ParseStrictSizeValue(object, offset_colon + 1, "tensor offset");
      has_offset = true;
    }

    // 解析张量在 model.bin 中占用的字节数
    size_t byte_size_pos = object.find("\"byte_size\"");
    if (byte_size_pos != std::string::npos) {
      size_t byte_size_colon = object.find(':', byte_size_pos);
      if (byte_size_colon == std::string::npos) {
        throw std::runtime_error("tensor byte_size field is malformed");
      }
      info.byte_size =
          ParseStrictSizeValue(object, byte_size_colon + 1, "tensor byte_size");
      has_byte_size = true;
    }

    // 五个元数据字段全部必需，错误信息尽量带上已解析出的张量名称
    if (!has_name || info.name.empty()) {
      throw std::runtime_error("tensor missing name field");
    }
    if (!has_dtype) {
      throw std::runtime_error("tensor missing dtype field: " + info.name);
    }
    if (!has_shape) {
      throw std::runtime_error("tensor missing shape field: " + info.name);
    }
    if (!has_offset) {
      throw std::runtime_error("tensor missing offset field: " + info.name);
    }
    if (!has_byte_size) {
      throw std::runtime_error("tensor missing byte_size field: " + info.name);
    }

    result.push_back(std::move(info));
    position = object_end + 1;
  }
  return result;
}

static void ValidateManifest(const ModelManifest& manifest) {
  if (manifest.tensors.empty()) {
    throw std::runtime_error("model.json contains no tensors");
  }

  // 名称索引用于重复检测和必需张量查询，偏移副本用于区间重叠检查
  std::map<std::string, TensorInfo> tensor_map;
  std::vector<TensorInfo> by_offset;
  by_offset.reserve(manifest.tensors.size());

  for (const auto& info : manifest.tensors) {
    if (info.dtype != "float32") {
      throw std::runtime_error("unsupported dtype '" + info.dtype +
                               "' for tensor: " + info.name);
    }
    if (info.shape.empty()) {
      throw std::runtime_error("tensor shape must not be empty: " + info.name);
    }

    // shape 推导的 F32 字节数必须和清单声明完全一致
    size_t expected_bytes = CheckedTensorByteSize(info);
    if (expected_bytes != info.byte_size) {
      throw std::runtime_error("tensor byte_size mismatch for " + info.name +
                               ": expected " + std::to_string(expected_bytes) +
                               ", manifest says " + std::to_string(info.byte_size));
    }
    if (info.offset > std::numeric_limits<size_t>::max() - info.byte_size) {
      throw std::runtime_error("tensor offset overflow for " + info.name);
    }

    auto inserted = tensor_map.emplace(info.name, info);
    if (!inserted.second) {
      throw std::runtime_error("duplicate tensor name: " + info.name);
    }
    by_offset.push_back(info);
  }

  // 按 offset 排序后比较相邻区间即可发现所有重叠
  std::sort(by_offset.begin(), by_offset.end(),
            [](const TensorInfo& left, const TensorInfo& right) {
              return left.offset < right.offset;
            });

  for (size_t index = 1; index < by_offset.size(); ++index) {
    size_t previous_end =
        by_offset[index - 1].offset + by_offset[index - 1].byte_size;
    if (by_offset[index].offset < previous_end) {
      throw std::runtime_error(
          "tensor ranges overlap: " + by_offset[index - 1].name + " and " +
          by_offset[index].name);
    }
  }

  // 最后按模型结构检查所有必需张量及其逻辑形状
  for (const auto& expected : ExpectedTensorShapes(manifest.config)) {
    auto found = tensor_map.find(expected.first);
    if (found == tensor_map.end()) {
      throw std::runtime_error("missing required tensor: " + expected.first);
    }
    if (found->second.shape != expected.second) {
      throw std::runtime_error("tensor shape mismatch for " + expected.first +
                               ": expected " + ShapeToString(expected.second) +
                               ", manifest says " +
                               ShapeToString(found->second.shape));
    }
  }
}

ModelManifest ParseManifest(const std::string& config_path) {
  ModelManifest manifest;

  // 一次性读取 model.json，后续轻量解析均在内存字符串上完成
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    throw std::runtime_error("failed to open config: " + config_path);
  }
  std::stringstream buffer;
  buffer << config_file.rdbuf();
  if (config_file.bad()) {
    throw std::runtime_error("failed to read config: " + config_path);
  }

  // 先解析配置和 tokenizer，再读取依赖配置校验的张量清单
  std::string content = buffer.str();
  manifest.config = ParseConfigJson(content);
  manifest.tokenizer = ParseTokenizerJson(content);
  manifest.tensors = ParseTensorMetadata(content);
  ValidateManifest(manifest);
  return manifest;
}

static Tensor LoadTensorByName(std::ifstream& weights_file,
                               const std::map<std::string, TensorInfo>& tensor_map,
                               const std::string& name) {
  // 模型结构通过名称查找权重，因此不依赖 tensors 在 JSON 中的排列顺序
  auto found = tensor_map.find(name);
  if (found == tensor_map.end()) {
    throw std::runtime_error("missing required tensor: " + name);
  }
  const TensorInfo& info = found->second;

  // 标准流 API 使用 streamoff 和 streamsize，转换前检查范围
  if (info.offset >
      static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
    throw std::runtime_error("tensor offset exceeds stream limit for " + name);
  }
  if (info.byte_size >
      static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("tensor byte size exceeds stream limit for " + name);
  }

  // 清除可能残留的流状态，并精确定位到张量起始字节
  weights_file.clear();
  weights_file.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
  if (!weights_file.good()) {
    throw std::runtime_error("failed to seek to offset for tensor: " + name);
  }

  // 当前阶段所有权重均以连续 F32 数据直接读入 Tensor
  Tensor tensor(info.shape, 0.0f);
  weights_file.read(reinterpret_cast<char*>(tensor.data.data()),
                    static_cast<std::streamsize>(info.byte_size));
  if (static_cast<size_t>(weights_file.gcount()) != info.byte_size) {
    throw std::runtime_error("weights file ended while reading tensor: " + name);
  }
  return tensor;
}

MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path) {
  // 清单异常转换成失败模型，保持加载接口不抛异常的调用约定
  ModelManifest manifest;
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& error) {
    return LoadFailure(error.what());
  }

  std::cout << "Loaded config: vocab=" << manifest.config.vocab_size
            << " dim=" << manifest.config.dim
            << " layers=" << manifest.config.n_layers
            << " heads=" << manifest.config.n_heads
            << " kv_heads=" << manifest.config.n_kv_heads
            << " head_dim=" << manifest.config.head_dim
            << " hidden_dim=" << manifest.config.hidden_dim << std::endl;

  // 构建名称索引，后续按模型结构顺序逐个加载张量
  std::map<std::string, TensorInfo> tensor_map;
  for (const auto& info : manifest.tensors) {
    tensor_map[info.name] = info;
  }

  std::ifstream weights_file(weights_path, std::ios::binary);
  if (!weights_file.is_open()) {
    return LoadFailure("failed to open weights: " + weights_path);
  }

  MiniLlamaModel model;
  model.config = manifest.config;

  try {
    // 先加载嵌入表，再按层加载 Attention 和 FFN 权重
    model.token_embedding =
        LoadTensorByName(weights_file, tensor_map, "token_embedding");

    model.layers.resize(static_cast<size_t>(manifest.config.n_layers));

    for (int layer = 0; layer < manifest.config.n_layers; ++layer) {
      LayerWeights& layer_weights = model.layers[static_cast<size_t>(layer)];
      const std::string prefix = "layers." + std::to_string(layer) + ".";

      layer_weights.attention_norm =
          LoadTensorByName(weights_file, tensor_map, prefix + "attention_norm");
      layer_weights.wq = LoadTensorByName(weights_file, tensor_map, prefix + "wq");
      layer_weights.wk = LoadTensorByName(weights_file, tensor_map, prefix + "wk");
      layer_weights.wv = LoadTensorByName(weights_file, tensor_map, prefix + "wv");
      layer_weights.wo = LoadTensorByName(weights_file, tensor_map, prefix + "wo");
      layer_weights.ffn_norm =
          LoadTensorByName(weights_file, tensor_map, prefix + "ffn_norm");
      layer_weights.w_gate =
          LoadTensorByName(weights_file, tensor_map, prefix + "w_gate");
      layer_weights.w_up =
          LoadTensorByName(weights_file, tensor_map, prefix + "w_up");
      layer_weights.w_down =
          LoadTensorByName(weights_file, tensor_map, prefix + "w_down");
    }

    // 模型末尾是最终归一化参数和词表输出头
    model.final_norm = LoadTensorByName(weights_file, tensor_map, "final_norm");
    model.lm_head = LoadTensorByName(weights_file, tensor_map, "lm_head");
  } catch (const std::exception& error) {
    return LoadFailure(error.what());
  }

  // 最大张量结束位置就是清单描述的 model.bin 精确长度
  weights_file.clear();
  weights_file.seekg(0, std::ios::end);
  std::streamoff file_size = weights_file.tellg();
  if (file_size < 0) {
    return LoadFailure("failed to determine weights file size");
  }

  size_t expected_end = 0;
  for (const auto& info : manifest.tensors) {
    size_t tensor_end = info.offset + info.byte_size;
    if (tensor_end > expected_end) {
      expected_end = tensor_end;
    }
  }

  // 文件过长通常表示版本不匹配，过短表示权重被截断
  if (static_cast<uintmax_t>(file_size) != static_cast<uintmax_t>(expected_end)) {
    if (static_cast<uintmax_t>(file_size) > static_cast<uintmax_t>(expected_end)) {
      return LoadFailure(
          "weights file has trailing bytes: file size=" + std::to_string(file_size) +
          ", expected=" + std::to_string(expected_end));
    }
    return LoadFailure(
        "weights file is too short: file size=" + std::to_string(file_size) +
        ", expected=" + std::to_string(expected_end));
  }

  // 所有张量和文件边界均验证完成后才发布 loaded 状态
  model.loaded = true;
  model.load_error.clear();
  std::cout << "Model loaded successfully from " << weights_path << std::endl;
  return model;
}

bool InspectModel(const std::string& config_path) {
  // InspectModel 只解析 model.json，不打开 model.bin
  ModelManifest manifest;
  try {
    manifest = ParseManifest(config_path);
  } catch (const std::exception& error) {
    std::cerr << "Failed to inspect model: " << error.what() << std::endl;
    return false;
  }

  const ModelConfig& config = manifest.config;
  std::cout << "=== Model Config ===" << std::endl;
  std::cout << "  vocab_size:    " << config.vocab_size << std::endl;
  std::cout << "  dim:           " << config.dim << std::endl;
  std::cout << "  hidden_dim:    " << config.hidden_dim << std::endl;
  std::cout << "  n_layers:      " << config.n_layers << std::endl;
  std::cout << "  n_heads:       " << config.n_heads << std::endl;
  std::cout << "  n_kv_heads:    " << config.n_kv_heads << std::endl;
  std::cout << "  head_dim:      " << config.head_dim << std::endl;
  std::cout << "  max_seq_len:   " << config.max_seq_len << std::endl;
  std::cout << "  rope_theta:    " << config.rope_theta << std::endl;
  std::cout << "  rms_norm_eps:  " << config.rms_norm_eps << std::endl;

  std::cout << std::endl;
  std::cout << "=== Tensors (" << manifest.tensors.size() << ") ===" << std::endl;

  size_t total_bytes = 0;
  for (const auto& tensor : manifest.tensors) {
    // 逐项输出原始清单信息，便于核对 shape 和二进制布局
    std::cout << "  " << tensor.name << std::endl;
    std::cout << "    shape:     [";

    for (size_t index = 0; index < tensor.shape.size(); ++index) {
      if (index > 0) {
        std::cout << ", ";
      }
      std::cout << tensor.shape[index];
    }
    std::cout << "]" << std::endl;
    std::cout << "    dtype:     " << tensor.dtype << std::endl;
    std::cout << "    offset:    " << tensor.offset << std::endl;
    std::cout << "    byte_size: " << tensor.byte_size << std::endl;

    if (total_bytes > std::numeric_limits<size_t>::max() - tensor.byte_size) {
      std::cerr << "Failed to inspect model: total tensor bytes overflow"
                << std::endl;
      return false;
    }
    total_bytes += tensor.byte_size;
  }

  std::cout << std::endl;
  std::cout << "=== Tokenizer ===" << std::endl;
  std::cout << "  type: " << manifest.tokenizer.type << std::endl;
  std::cout << "  path: "
            << (manifest.tokenizer.path.empty() ? "<none>" : manifest.tokenizer.path)
            << std::endl;
  std::cout << "  bos_id: " << manifest.tokenizer.bos_id << std::endl;
  std::cout << "  eos_id: " << manifest.tokenizer.eos_id << std::endl;
  std::cout << "  unk_id: " << manifest.tokenizer.unk_id << std::endl;
  std::cout << std::endl;
  std::cout << "Total tensor bytes: " << total_bytes << std::endl;
  return true;
}

}  // namespace mini_llama
