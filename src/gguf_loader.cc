#include "mini_llama/gguf_loader.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {
namespace {

constexpr size_t kQuantBlockSize = 32;

MiniLlamaModel LoadFailure(const std::string& message) {
  // 统一保持加载器“不抛异常、返回可检查失败状态”的对外约定
  MiniLlamaModel model;
  model.load_error = message;
  return model;
}

// 将 GGUF（llama.cpp 风格）张量名映射到项目内部的模型字段名
std::string MapTensorName(const std::string& gguf_name) {
  // 模型级张量不带层号，可直接映射到 MiniLlamaModel 字段
  if (gguf_name == "token_embd.weight") {
    return "token_embedding";
  }
  if (gguf_name == "output_norm.weight") {
    return "final_norm";
  }
  if (gguf_name == "output.weight") {
    return "lm_head";
  }

  // 层张量遵循 blk.{layer}.{operation}.{weight|bias} 的 GGUF 约定
  if (gguf_name.rfind("blk.", 0) != 0) {
    return "";
  }
  const size_t layer_end = gguf_name.find('.', 4);
  if (layer_end == std::string::npos || layer_end == 4) {
    return "";
  }
  const std::string layer = gguf_name.substr(4, layer_end - 4);
  // 层号必须是纯十进制数字，防止格式相近的扩展名称被错误接收
  for (char character : layer) {
    if (character < '0' || character > '9') {
      return "";
    }
  }

  const size_t operation_end = gguf_name.find('.', layer_end + 1);
  if (operation_end == std::string::npos) {
    return "";
  }
  const std::string operation =
      gguf_name.substr(layer_end + 1, operation_end - layer_end - 1);
  const std::string suffix = gguf_name.substr(operation_end + 1);

  // 偏置只支持当前 LayerWeights 已定义的 Q/K/V 字段；其他张量留给后续扩展
  std::string mapped_operation;
  if (suffix == "bias") {
    if (operation == "attn_q") {
      mapped_operation = "bq";
    } else if (operation == "attn_k") {
      mapped_operation = "bk";
    } else if (operation == "attn_v") {
      mapped_operation = "bv";
    } else {
      return "";
    }
  } else if (suffix != "weight") {
    return "";
  } else if (operation == "attn_norm") {
    mapped_operation = "attention_norm";
  } else if (operation == "attn_q") {
    mapped_operation = "wq";
  } else if (operation == "attn_k") {
    mapped_operation = "wk";
  } else if (operation == "attn_v") {
    mapped_operation = "wv";
  } else if (operation == "attn_output") {
    mapped_operation = "wo";
  } else if (operation == "ffn_norm") {
    mapped_operation = "ffn_norm";
  } else if (operation == "ffn_gate") {
    mapped_operation = "w_gate";
  } else if (operation == "ffn_up") {
    mapped_operation = "w_up";
  } else if (operation == "ffn_down") {
    mapped_operation = "w_down";
  } else {
    return "";
  }
  return "layers." + layer + "." + mapped_operation;
}

// GGUF 以 ne0 为最快变化维；反转维度后可直接复用其字节顺序作为行主序
std::vector<int> MapShape(const std::vector<uint64_t>& gguf_shape) {
  std::vector<int> shape;
  shape.reserve(gguf_shape.size());
  // 反向遍历同时完成 uint64_t 到项目 int 维度类型的安全转换
  for (auto dimension = gguf_shape.rbegin(); dimension != gguf_shape.rend();
       ++dimension) {
    if (*dimension == 0 ||
        *dimension > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("GGUF tensor dimension exceeds Tensor limits");
    }
    shape.push_back(static_cast<int>(*dimension));
  }
  return shape;
}

uint16_t ReadLittleEndian16(const uint8_t* bytes) {
  // 显式组装字节，避免加载结果依赖宿主机端序或未对齐指针访问
  return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t ReadLittleEndian32(const uint8_t* bytes) {
  // GGUF 固定使用小端格式，因此不能直接 reinterpret_cast 为宿主标量
  return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

// GGML 量化块使用 IEEE 754 binary16 保存缩放参数，此处无外部依赖转换
float HalfToFloat(uint16_t half) {
  // 先拆分符号、指数和尾数，再分别处理零、次正规数、普通数和 Inf/NaN
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000) << 16;
  const uint32_t exponent = (half >> 10) & 0x1f;
  const uint32_t mantissa = half & 0x03ff;
  uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // 半精度次正规数范围很小，直接用 ldexp 可避免手工规格化错误
      const float value = std::ldexp(static_cast<float>(mantissa), -24);
      return sign == 0 ? value : -value;
    }
  } else if (exponent == 0x1f) {
    // 保留 Inf/NaN 的尾数载荷并扩展到 float32 位布局
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    // binary16 与 binary32 的指数偏置差为 127 - 15 = 112
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }

  // 通过 memcpy 恢复浮点值，避免违反严格别名规则
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

size_t CheckedElementCount(const std::vector<int>& shape, const std::string& name) {
  size_t count = 1;
  for (int dimension : shape) {
    // 每次乘法前检查溢出，防止后续 Tensor 分配得到截断后的大小
    const size_t converted = static_cast<size_t>(dimension);
    if (converted == 0 || count > std::numeric_limits<size_t>::max() / converted) {
      throw std::runtime_error("GGUF tensor element count overflow: " + name);
    }
    count *= converted;
  }
  return count;
}

// 仅将 norm、bias 和 embedding 等稠密张量展开为连续 F32
Tensor DecodeDenseTensor(const std::string& path, const GgufReader& reader,
                         const GgufTensorInfo& info) {
  // 先建立项目侧逻辑形状，并独立计算元素数与 GGUF 物理字节数
  const std::vector<int> shape = MapShape(info.shape);
  const size_t element_count = CheckedElementCount(shape, info.name);
  uint64_t expected_size = 0;
  if (!GgufTensorDataSize(info, expected_size) || expected_size == 0 ||
      expected_size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("unsupported or invalid GGUF tensor type for: " +
                             info.name);
  }

  const std::vector<uint8_t> bytes =
      ReadGgufTensorRaw(path, info, reader.data_offset);
  // 精确大小检查可区分完整读取与短读，避免对残缺量化块进行解码
  if (bytes.size() != static_cast<size_t>(expected_size)) {
    throw std::runtime_error("failed to read GGUF tensor data: " + info.name);
  }

  Tensor tensor(shape, 0.0f);
  if (info.type == kGgmlTypeF32) {
    // F32 仍逐元素按小端恢复，保证在不同宿主端序下行为一致
    for (size_t index = 0; index < element_count; ++index) {
      const uint32_t bits = ReadLittleEndian32(bytes.data() + index * 4);
      std::memcpy(&tensor.data[index], &bits, sizeof(float));
    }
    return tensor;
  }
  if (info.type == kGgmlTypeF16) {
    // F16 权重逐元素扩展为当前前向路径唯一接受的 F32
    for (size_t index = 0; index < element_count; ++index) {
      tensor.data[index] =
          HalfToFloat(ReadLittleEndian16(bytes.data() + index * sizeof(uint16_t)));
    }
    return tensor;
  }

  if (element_count % kQuantBlockSize != 0) {
    throw std::runtime_error("invalid quantized GGUF tensor shape: " + info.name);
  }
  const size_t block_count = element_count / kQuantBlockSize;
  size_t block_bytes = 0;
  // GGML 三种量化格式均以 32 个元素为块，但块头和载荷大小不同
  if (info.type == kGgmlTypeQ40) {
    block_bytes = 18;
  } else if (info.type == kGgmlTypeQ41) {
    block_bytes = 20;
  } else if (info.type == kGgmlTypeQ80) {
    block_bytes = 34;
  } else {
    throw std::runtime_error("unsupported GGML tensor type for: " + info.name);
  }

  for (size_t block = 0; block < block_count; ++block) {
    // 每个块首部都以 F16 保存缩放系数，随后是该块的量化值
    const uint8_t* source = bytes.data() + block * block_bytes;
    const float scale = HalfToFloat(ReadLittleEndian16(source));
    const size_t output_offset = block * kQuantBlockSize;

    if (info.type == kGgmlTypeQ80) {
      // Q8_0 的 32 个 int8 值连续存放，反量化公式为 scale * q
      const int8_t* quantized = reinterpret_cast<const int8_t*>(source + 2);
      for (size_t index = 0; index < kQuantBlockSize; ++index) {
        tensor.data[output_offset + index] =
            scale * static_cast<float>(quantized[index]);
      }
      continue;
    }

    const float minimum =
        info.type == kGgmlTypeQ41
            ? HalfToFloat(ReadLittleEndian16(source + sizeof(uint16_t)))
            : 0.0f;
    const uint8_t* quantized = source + (info.type == kGgmlTypeQ41 ? 4 : 2);
    // 每字节的低/高四位分别对应块的前半段和后半段元素
    for (size_t index = 0; index < kQuantBlockSize / 2; ++index) {
      const uint8_t packed = quantized[index];
      const int low = packed & 0x0f;
      const int high = packed >> 4;
      if (info.type == kGgmlTypeQ40) {
        // Q4_0 使用以 8 为零点的有符号四位值
        tensor.data[output_offset + index] = scale * static_cast<float>(low - 8);
        tensor.data[output_offset + index + kQuantBlockSize / 2] =
            scale * static_cast<float>(high - 8);
      } else {
        // Q4_1 使用非负四位值，并为整个块增加独立最小值
        tensor.data[output_offset + index] = scale * low + minimum;
        tensor.data[output_offset + index + kQuantBlockSize / 2] =
            scale * high + minimum;
      }
    }
  }
  return tensor;
}

// 线性权重使用带类型的存储：Q4/Q8 保留 GGML 块，F32 保持连续浮点数组
// 这样加载阶段不会为量化矩阵额外分配一份完整的 F32 Tensor
QuantizedTensor DecodeLinearWeight(const std::string& path, const GgufReader& reader,
                                   const GgufTensorInfo& info) {
  const std::vector<int> shape = MapShape(info.shape);
  if (shape.size() != 2) {
    throw std::runtime_error("GGUF linear weight must be two-dimensional: " +
                             info.name);
  }
  // QuantizedTensor 当前没有 F16 存储分支，因此仅 F16 仍转换为 F32
  if (info.type == kGgmlTypeF16) {
    return ToQuantizedTensor(DecodeDenseTensor(path, reader, info));
  }

  // 将 GGML 文件类型映射到项目内部的量化类型标签
  QuantType type;
  if (info.type == kGgmlTypeF32) {
    type = QuantType::kF32;
  } else if (info.type == kGgmlTypeQ80) {
    type = QuantType::kQ80;
  } else if (info.type == kGgmlTypeQ40) {
    type = QuantType::kQ40;
  } else if (info.type == kGgmlTypeQ41) {
    type = QuantType::kQ41;
  } else {
    throw std::runtime_error("unsupported GGML linear weight type for: " +
                             info.name);
  }

  uint64_t expected_size = 0;
  if (!GgufTensorDataSize(info, expected_size) || expected_size == 0 ||
      expected_size > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("unsupported or invalid GGUF tensor type for: " +
                             info.name);
  }
  // 原始字节只在构造对应块数组期间存在，不会展开成 element_count 个 float
  const std::vector<uint8_t> bytes =
      ReadGgufTensorRaw(path, info, reader.data_offset);
  if (bytes.size() != static_cast<size_t>(expected_size)) {
    throw std::runtime_error("failed to read GGUF tensor data: " + info.name);
  }
  if (type == QuantType::kF32) {
    // F32 也放入 QuantizedTensor，使所有线性权重共享同一前向接口
    QuantizedTensor tensor;
    tensor.type = QuantType::kF32;
    tensor.shape = shape;
    tensor.f32_data.resize(CheckedElementCount(shape, info.name));
    for (size_t index = 0; index < tensor.f32_data.size(); ++index) {
      const uint32_t bits = ReadLittleEndian32(bytes.data() + index * 4);
      std::memcpy(&tensor.f32_data[index], &bits, sizeof(float));
    }
    return tensor;
  }
  // GGML 量化块沿最快变化的输入维排列，不能跨越两个矩阵行
  if (shape[1] % static_cast<int>(kQuantBlockSize) != 0) {
    throw std::runtime_error("GGUF quantized linear row size must be divisible by " +
                             std::to_string(kQuantBlockSize) + ": " + info.name);
  }

  const size_t element_count = CheckedElementCount(shape, info.name);
  const size_t block_count = element_count / kQuantBlockSize;
  QuantizedTensor tensor;
  tensor.type = type;
  tensor.shape = shape;
  if (type == QuantType::kQ80) {
    // Q8_0 块由一个 F16 scale 和 32 个 int8 权重组成
    tensor.q8_0_data.resize(block_count);
    for (size_t block = 0; block < block_count; ++block) {
      const uint8_t* source = bytes.data() + block * sizeof(BlockQ80);
      tensor.q8_0_data[block].d = ReadLittleEndian16(source);
      std::memcpy(tensor.q8_0_data[block].qs, source + sizeof(uint16_t),
                  kQ80BlockSize);
    }
  } else if (type == QuantType::kQ40) {
    // Q4_0 每字节保存两个四位权重，scale 保持文件中的 F16 位模式
    tensor.q4_0_data.resize(block_count);
    for (size_t block = 0; block < block_count; ++block) {
      const uint8_t* source = bytes.data() + block * sizeof(BlockQ40);
      tensor.q4_0_data[block].d = ReadLittleEndian16(source);
      std::memcpy(tensor.q4_0_data[block].qs, source + sizeof(uint16_t),
                  kQ40BlockSize / 2);
    }
  } else {
    // Q4_1 比 Q4_0 多一个 F16 minimum，量化值本身仍按半字节打包
    tensor.q4_1_data.resize(block_count);
    for (size_t block = 0; block < block_count; ++block) {
      const uint8_t* source = bytes.data() + block * sizeof(BlockQ41);
      tensor.q4_1_data[block].d = ReadLittleEndian16(source);
      tensor.q4_1_data[block].m = ReadLittleEndian16(source + sizeof(uint16_t));
      std::memcpy(tensor.q4_1_data[block].qs, source + sizeof(uint16_t) * 2,
                  kQ41BlockSize / 2);
    }
  }
  return tensor;
}

int RequiredPositiveInt(const std::string& path, const std::string& key) {
  int64_t value = 0;
  // 类型化 getter 同时拒绝缺失键和非整数元数据
  if (!GgufGetMetadataInt(path, key, value)) {
    throw std::runtime_error("missing integer GGUF metadata: " + key);
  }
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid positive GGUF metadata: " + key);
  }
  return static_cast<int>(value);
}

int OptionalPositiveInt(const std::string& path, const std::string& key,
                        int fallback) {
  int64_t value = 0;
  // 可选键缺失时使用调用方给出的架构默认值；存在时仍严格检查范围
  if (!GgufGetMetadataInt(path, key, value)) {
    return fallback;
  }
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid positive GGUF metadata: " + key);
  }
  return static_cast<int>(value);
}

float OptionalPositiveFloat(const std::string& path, const std::string& key,
                            float fallback) {
  double value = 0.0;
  // 元数据接口使用 double 承接 F32/F64，确认范围后再缩窄为模型配置的 float
  if (!GgufGetMetadataFloat(path, key, value)) {
    return fallback;
  }
  if (!std::isfinite(value) || value <= 0.0 ||
      value > std::numeric_limits<float>::max()) {
    throw std::runtime_error("invalid positive GGUF metadata: " + key);
  }
  return static_cast<float>(value);
}

const GgufTensorInfo& FindTensor(const GgufReader& reader, const std::string& name) {
  // 配置构建阶段只查找一次嵌入张量，保持读取器接口简单且不复制描述对象
  for (const GgufTensorInfo& tensor : reader.tensors) {
    if (tensor.name == name) {
      return tensor;
    }
  }
  throw std::runtime_error("missing required GGUF tensor: " + name);
}

// 从架构名前缀读取标准 GGUF 配置，不依赖某个模型的固定层数或维度
ModelConfig BuildConfig(const std::string& path, const GgufReader& reader) {
  std::string architecture;
  // 架构名同时决定元数据键前缀
  if (!GgufGetMetadataString(path, "general.architecture", architecture) ||
      architecture.empty()) {
    throw std::runtime_error("missing string GGUF metadata: general.architecture");
  }
  const std::string prefix = architecture + ".";

  ModelConfig config;
  // 结构参数来自 GGUF 标准架构键；KV 头数允许缺省为查询头数
  config.dim = RequiredPositiveInt(path, prefix + "embedding_length");
  config.hidden_dim = RequiredPositiveInt(path, prefix + "feed_forward_length");
  config.n_layers = RequiredPositiveInt(path, prefix + "block_count");
  config.n_heads = RequiredPositiveInt(path, prefix + "attention.head_count");
  config.n_kv_heads =
      OptionalPositiveInt(path, prefix + "attention.head_count_kv", config.n_heads);
  config.max_seq_len = RequiredPositiveInt(path, prefix + "context_length");
  config.rope_theta =
      OptionalPositiveFloat(path, prefix + "rope.freq_base", config.rope_theta);
  config.rms_norm_eps = OptionalPositiveFloat(
      path, prefix + "attention.layer_norm_rms_epsilon", config.rms_norm_eps);

  // 这些整除约束是 GQA 头映射和 RoPE 二元旋转的前置条件
  if (config.dim % config.n_heads != 0 || config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error("incompatible GGUF attention dimensions");
  }

  config.head_dim = config.dim / config.n_heads;
  if (config.head_dim % 2 != 0) {
    throw std::runtime_error("GGUF attention head dimension must be even");
  }

  // 词表大小优先读取架构元数据；缺省时以嵌入矩阵的输出维为准
  config.vocab_size = OptionalPositiveInt(path, prefix + "vocab_size", 0);
  const std::vector<int> embedding_shape =
      MapShape(FindTensor(reader, "token_embd.weight").shape);
  if (embedding_shape.size() != 2 || embedding_shape[1] != config.dim) {
    throw std::runtime_error("GGUF token embedding shape is incompatible");
  }
  if (config.vocab_size == 0) {
    config.vocab_size = embedding_shape[0];
  } else if (config.vocab_size != embedding_shape[0]) {
    throw std::runtime_error("GGUF vocabulary size does not match token embedding");
  }

  // Qwen2 的 rotary pairing 使用前后半维配对，其他架构保留项目默认模式
  if (architecture == "qwen2") {
    config.rope_type = RopeType::kNeoX;
  }
  return config;
}

void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                  const std::string& name) {
  // 复用 Tensor 的统一错误格式，并传入完整权重名便于定位损坏张量
  tensor.AssertShape(expected, name.c_str());
}

void RequireShape(const QuantizedTensor& tensor, const std::vector<int>& expected,
                  const std::string& name) {
  // 这里只校验逻辑形状；物理块数量在加载和 Linear 分发时分别严格检查。
  tensor.AssertShape(expected, name.c_str());
}

void RequireOptionalShape(const Tensor& tensor, const std::vector<int>& expected,
                          const std::string& name) {
  // 空 Tensor 表示 GGUF 中未提供可选偏置，不应按缺失必需权重处理
  if (!tensor.data.empty()) {
    RequireShape(tensor, expected, name);
  }
}

void ValidateLoadedShapes(const MiniLlamaModel& model) {
  const ModelConfig& config = model.config;
  // Q 投影使用全部查询头，K/V 投影只使用 GQA 的 KV 头
  const int query_dim = config.n_heads * config.head_dim;
  const int kv_dim = config.n_kv_heads * config.head_dim;

  RequireShape(model.token_embedding, {config.vocab_size, config.dim},
               "GGUF token_embedding");
  RequireShape(model.final_norm, {config.dim}, "GGUF final_norm");
  RequireShape(model.lm_head, {config.vocab_size, config.dim}, "GGUF lm_head");
  if (model.layers.size() != static_cast<size_t>(config.n_layers)) {
    throw std::runtime_error("GGUF layer count mismatch");
  }

  for (int layer = 0; layer < config.n_layers; ++layer) {
    // 按前向计算的 [out, in] 约定逐层验证，加载完成后不再猜测布局
    const LayerWeights& weights = model.layers[static_cast<size_t>(layer)];
    const std::string prefix = "GGUF layers." + std::to_string(layer) + ".";
    RequireShape(weights.attention_norm, {config.dim}, prefix + "attention_norm");
    RequireShape(weights.wq, {query_dim, config.dim}, prefix + "wq");
    RequireShape(weights.wk, {kv_dim, config.dim}, prefix + "wk");
    RequireShape(weights.wv, {kv_dim, config.dim}, prefix + "wv");
    RequireOptionalShape(weights.bq, {query_dim}, prefix + "bq");
    RequireOptionalShape(weights.bk, {kv_dim}, prefix + "bk");
    RequireOptionalShape(weights.bv, {kv_dim}, prefix + "bv");
    RequireShape(weights.wo, {config.dim, query_dim}, prefix + "wo");
    RequireShape(weights.ffn_norm, {config.dim}, prefix + "ffn_norm");
    RequireShape(weights.w_gate, {config.hidden_dim, config.dim}, prefix + "w_gate");
    RequireShape(weights.w_up, {config.hidden_dim, config.dim}, prefix + "w_up");
    RequireShape(weights.w_down, {config.dim, config.hidden_dim}, prefix + "w_down");
  }
}

}  // namespace

MiniLlamaModel LoadGgufModel(const std::string& gguf_path) {
  try {
    // 第一阶段只解析并校验 GGUF 目录，尚不分配模型权重
    GgufReader reader;
    if (!reader.Load(gguf_path)) {
      return LoadFailure(reader.load_error);
    }

    MiniLlamaModel model;
    // 配置验证通过后再按实际层数创建权重容器
    model.config = BuildConfig(gguf_path, reader);
    model.layers.resize(static_cast<size_t>(model.config.n_layers));

    // 只索引当前模型认识的张量；其他 GGUF 扩展张量由对应后续模块处理
    std::map<std::string, const GgufTensorInfo*> tensors;
    for (const GgufTensorInfo& tensor : reader.tensors) {
      const std::string mapped_name = MapTensorName(tensor.name);
      if (!mapped_name.empty() && !tensors.emplace(mapped_name, &tensor).second) {
        throw std::runtime_error("duplicate mapped GGUF tensor: " + mapped_name);
      }
    }

    auto find_required = [&](const std::string& name) {
      // 必需权重缺失立即失败，避免发布部分初始化的模型
      const auto found = tensors.find(name);
      if (found == tensors.end()) {
        throw std::runtime_error("missing required GGUF tensor: " + name);
      }
      return found->second;
    };
    auto load_dense = [&](const std::string& name) {
      // norm、bias 和 embedding 仍由普通 F32 Tensor 承载。
      return DecodeDenseTensor(gguf_path, reader, *find_required(name));
    };
    auto load_linear = [&](const std::string& name) {
      // 所有 Linear 矩阵走量化感知路径，避免误用稠密解码函数。
      return DecodeLinearWeight(gguf_path, reader, *find_required(name));
    };
    auto load_optional = [&](const std::string& name) {
      // 可选偏置不存在时返回空 Tensor，存在时与普通权重走同一解码路径
      const auto found = tensors.find(name);
      return found == tensors.end()
                 ? Tensor()
                 : DecodeDenseTensor(gguf_path, reader, *found->second);
    };

    // 第二阶段按 MiniLlamaModel 的结构顺序加载模型级张量和每层权重
    model.token_embedding = load_dense("token_embedding");
    for (int layer = 0; layer < model.config.n_layers; ++layer) {
      LayerWeights& weights = model.layers[static_cast<size_t>(layer)];
      const std::string prefix = "layers." + std::to_string(layer) + ".";
      weights.attention_norm = load_dense(prefix + "attention_norm");
      weights.wq = load_linear(prefix + "wq");
      weights.wk = load_linear(prefix + "wk");
      weights.wv = load_linear(prefix + "wv");
      weights.bq = load_optional(prefix + "bq");
      weights.bk = load_optional(prefix + "bk");
      weights.bv = load_optional(prefix + "bv");
      weights.wo = load_linear(prefix + "wo");
      weights.ffn_norm = load_dense(prefix + "ffn_norm");
      weights.w_gate = load_linear(prefix + "w_gate");
      weights.w_up = load_linear(prefix + "w_up");
      weights.w_down = load_linear(prefix + "w_down");
    }
    model.final_norm = load_dense("final_norm");

    // GGUF 允许省略 output.weight 表示与 token_embd.weight 共享权重
    const auto output = tensors.find("lm_head");
    model.lm_head = output == tensors.end()
                        ? ToQuantizedTensor(model.token_embedding)
                        : DecodeLinearWeight(gguf_path, reader, *output->second);

    ValidateLoadedShapes(model);
    // 只有全部张量加载并通过形状校验后，模型才可进入前向路径
    model.loaded = true;
    model.load_error.clear();
    return model;
  } catch (const std::exception& error) {
    // 将解析、分配和形状异常统一转换为接口约定的失败模
    return LoadFailure(error.what());
  }
}

}  // namespace mini_llama
