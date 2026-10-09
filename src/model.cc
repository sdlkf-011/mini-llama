#include "mini_llama/model.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "mini_llama/quant.h"

namespace mini_llama {

namespace {

// 要求非空且各维为正，并在逐维相乘前检查 size_t 溢出
size_t WeightNumel(const std::vector<int>& shape, const std::string& name) {
  if (shape.empty()) {
    throw std::runtime_error("Model weight has empty shape: " + name);
  }
  size_t numel = 1;
  for (int dim : shape) {
    if (dim <= 0 ||
        numel > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error("Model weight has invalid shape: " + name);
    }
    numel *= static_cast<size_t>(dim);
  }
  return numel;
}

// 将元素/块数量换算为字节数，乘法溢出时附带权重名称报错
size_t StorageBytes(size_t count, size_t element_size, const std::string& name) {
  if (count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::runtime_error("Model weight byte count overflow: " + name);
  }
  return count * element_size;
}

// 上传与重新量化共用的入口校验：已加载、正维度、GQA/RoPE 约束及层数一致
void ValidateModelConfig(const MiniLlamaModel& model) {
  const auto& c = model.config;
  if (!model.loaded) {
    throw std::runtime_error("Model weight operation requires a loaded model");
  }
  if (c.vocab_size <= 0 || c.dim <= 0 || c.hidden_dim <= 0 || c.n_layers <= 0 ||
      c.n_heads <= 0 || c.n_kv_heads <= 0 || c.head_dim <= 0 || c.max_seq_len <= 0 ||
      static_cast<size_t>(c.dim) != static_cast<size_t>(c.n_heads) * c.head_dim ||
      c.n_heads % c.n_kv_heads != 0 || c.head_dim % 2 != 0 ||
      !std::isfinite(c.rope_theta) || c.rope_theta <= 0.0f ||
      !std::isfinite(c.rms_norm_eps) || c.rms_norm_eps < 0.0f ||
      model.layers.size() != static_cast<size_t>(c.n_layers)) {
    throw std::runtime_error("Model weight operation: invalid config or layers");
  }
}

// 上传和重新量化使用同一套权重名称与逻辑形状
// 模板可接受只读或可变模型，分别用于上传校验和原地权重转换
template <typename Model, typename Visit>
void VisitLinearWeights(Model& model, Visit visit) {
  const auto& c = model.config;
  const int kv_dim = c.n_kv_heads * c.head_dim;
  visit(model.lm_head, "lm_head", std::vector<int>{c.vocab_size, c.dim});
  for (size_t i = 0; i < model.layers.size(); ++i) {
    auto& layer = model.layers[i];
    const std::string prefix = "layers." + std::to_string(i) + ".";
    visit(layer.wq, prefix + "wq", std::vector<int>{c.dim, c.dim});
    visit(layer.wk, prefix + "wk", std::vector<int>{kv_dim, c.dim});
    visit(layer.wv, prefix + "wv", std::vector<int>{kv_dim, c.dim});
    visit(layer.wo, prefix + "wo", std::vector<int>{c.dim, c.dim});
    visit(layer.w_gate, prefix + "w_gate", std::vector<int>{c.hidden_dim, c.dim});
    visit(layer.w_up, prefix + "w_up", std::vector<int>{c.hidden_dim, c.dim});
    visit(layer.w_down, prefix + "w_down", std::vector<int>{c.dim, c.hidden_dim});
  }
}

// 同时核对逻辑 shape、活动存储数组和理论字节数
// 量化块逐输出行排列，每行独立向上补齐，不跨行共享尾块
void ValidateLinearStorage(const QuantizedTensor& weight,
                           const std::vector<int>& expected,
                           const std::string& name) {
  weight.AssertShape(expected, name.c_str());
  const size_t numel = WeightNumel(weight.shape, name);
  if (weight.shape.size() != 2) {
    throw std::runtime_error("Model Linear weight must have rank 2: " + name);
  }
  size_t actual = 0;
  size_t count = numel;
  size_t element_size = sizeof(float);
  int block_size = 0;
  switch (weight.type) {
    case QuantType::kF32:
      actual = weight.f32_data.size();
      break;
    case QuantType::kQ80:
      actual = weight.q8_0_data.size();
      element_size = sizeof(BlockQ80);
      block_size = kQ80BlockSize;
      break;
    case QuantType::kQ40:
      actual = weight.q4_0_data.size();
      element_size = sizeof(BlockQ40);
      block_size = kQ40BlockSize;
      break;
    case QuantType::kQ41:
      actual = weight.q4_1_data.size();
      element_size = sizeof(BlockQ41);
      block_size = kQ41BlockSize;
      break;
    default:
      throw std::runtime_error("Model weight has unknown quantization type: " +
                               name);
  }
  if (block_size != 0) {
    count = static_cast<size_t>(weight.shape[0]) *
            ((static_cast<size_t>(weight.shape[1]) + block_size - 1) / block_size);
  }
  StorageBytes(count, element_size, name);
  if (actual != count) {
    throw std::runtime_error("Model weight storage size mismatch: " + name +
                             ", expected " + std::to_string(count) + ", got " +
                             std::to_string(actual));
  }
}

// 先分配并上传一份权重，再追加到临时快照并更新总数与字节统计
// 本函数不发布模型快照，局部对象负责失败时的资源回收
void AppendWeight(CudaModelWeights& storage, CudaUploadedWeight uploaded,
                  const void* data, size_t bytes) {
  if (bytes > std::numeric_limits<size_t>::max() - storage.uploaded_bytes) {
    throw std::runtime_error("CUDA model weight byte count overflow");
  }
  uploaded.buffer.Reset(bytes, storage.device_id);
  uploaded.buffer.Upload(data, bytes);
  storage.weights.push_back(std::move(uploaded));
  storage.uploaded_weight_count = storage.weights.size();
  storage.uploaded_bytes += bytes;
}

// 校验 embedding/norm/bias 的 F32 存储，标记为非 Linear 后复用上传流程
void UploadDenseWeight(CudaModelWeights& storage, const Tensor& weight,
                       const std::string& name, const std::vector<int>& expected) {
  // 空 Tensor 表示未提供的普通 F32 权重（典型情况是可选 bias）
  if (weight.data.empty()) return;
  weight.AssertShape(expected, name.c_str());
  const size_t numel = WeightNumel(weight.shape, name);
  if (weight.data.size() != numel) {
    throw std::runtime_error("CUDA tensor storage size mismatch: " + name);
  }
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.linear = false;
  uploaded.shape = weight.shape;
  AppendWeight(storage, std::move(uploaded), weight.data.data(),
               StorageBytes(numel, sizeof(float), name));
}

// 重新量化时将活动存储转换为完整 F32 临时张量；仅用于权重转换
Tensor QuantizedTensorToF32(const QuantizedTensor& weight) {
  switch (weight.type) {
    case QuantType::kF32:
      return ToTensor(weight);
    case QuantType::kQ80:
      return DequantizeFromQ80(weight.q8_0_data, weight.shape);
    case QuantType::kQ40:
      return DequantizeFromQ40(weight.q4_0_data, weight.shape);
    case QuantType::kQ41:
      return DequantizeFromQ41(weight.q4_1_data, weight.shape);
  }
  throw std::runtime_error("Model weight has unknown quantization type");
}

// 内部目标只由 Q8_0/Q4_0 公开入口传入；先准备全部新权重，再统一提交
// 已是目标格式的权重仅校验，全部无需转换时保留现有 CUDA 快照
void QuantizeModel(MiniLlamaModel& model, QuantType target) {
  ValidateModelConfig(model);

  // 先准备全部新数据，任一权重校验或量化失败都不改变原模型
  std::vector<std::pair<QuantizedTensor*, QuantizedTensor>> replacements;
  VisitLinearWeights(model, [&](QuantizedTensor& weight, const std::string& name,
                                const std::vector<int>& expected) {
    ValidateLinearStorage(weight, expected, name);
    if (weight.type == target) return;
    const Tensor dense = QuantizedTensorToF32(weight);
    QuantizedTensor converted;
    converted.type = target;
    converted.shape = weight.shape;
    if (target == QuantType::kQ80) {
      converted.q8_0_data = QuantizeToQ80(dense);
    } else {
      converted.q4_0_data = QuantizeToQ40(dense);
    }
    replacements.emplace_back(&weight, std::move(converted));
  });
  if (replacements.empty()) return;
  for (auto& replacement : replacements) {
    *replacement.first = std::move(replacement.second);
  }
  // CPU 权重已发生变化，解除当前模型对旧设备快照的引用以避免使用过期数据
  ClearModelCudaWeights(model);
}

// 按快照中的 Linear 类型计数，稠密权重不计入；没有快照时返回零
size_t UploadedLinearCount(const MiniLlamaModel& model, QuantType type) {
  if (!model.cuda_weights) return 0;
  size_t count = 0;
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.linear && weight.type == type) ++count;
  }
  return count;
}

// 通过成员指针读取快照字段，无快照返回零；既用于上传统计也用于运行统计
size_t RuntimeStat(const MiniLlamaModel& model, size_t CudaModelWeights::*field) {
  return model.cuda_weights ? model.cuda_weights.get()->*field : 0;
}

}  // namespace

// Tensor 仅承载 F32，实际存储字节数就是元素数乘 sizeof(float)
static size_t TensorBytes(const Tensor& tensor) {
  return tensor.data.size() * sizeof(float);
}

// 只统计 type 指定的活动存储数组，不把未使用 vector 的容量计入权重数据
static size_t QuantizedTensorBytes(const QuantizedTensor& tensor) {
  switch (tensor.type) {
    case QuantType::kF32:
      return tensor.f32_data.size() * sizeof(float);
    case QuantType::kQ80:
      return tensor.q8_0_data.size() * sizeof(BlockQ80);
    case QuantType::kQ40:
      return tensor.q4_0_data.size() * sizeof(BlockQ40);
    case QuantType::kQ41:
      return tensor.q4_1_data.size() * sizeof(BlockQ41);
  }
  return 0;
}

// 等价 F32 大小按逻辑元素数计算，与当前量化块格式无关
static size_t QuantizedTensorF32Bytes(const QuantizedTensor& qt) {
  return qt.num_elements() * sizeof(float);
}

// 统计模型持有的权重数据，不包含对象、shape vector 和 allocator 元数据
size_t ModelWeightBytes(const MiniLlamaModel& model) {
  size_t bytes = TensorBytes(model.token_embedding) + TensorBytes(model.final_norm) +
                 QuantizedTensorBytes(model.lm_head);
  for (const LayerWeights& layer : model.layers) {
    // Norm 和 bias 按 F32 统计，Linear 权重按当前实际存储格式统计
    bytes += TensorBytes(layer.attention_norm) + TensorBytes(layer.bq) +
             TensorBytes(layer.bk) + TensorBytes(layer.bv) +
             TensorBytes(layer.ffn_norm) + QuantizedTensorBytes(layer.wq) +
             QuantizedTensorBytes(layer.wk) + QuantizedTensorBytes(layer.wv) +
             QuantizedTensorBytes(layer.wo) + QuantizedTensorBytes(layer.w_gate) +
             QuantizedTensorBytes(layer.w_up) + QuantizedTensorBytes(layer.w_down);
  }
  return bytes;
}

// 稠密张量沿用实际大小，QuantizedTensor 则按逻辑元素数换算为 F32
size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
  size_t bytes = TensorBytes(model.token_embedding) + TensorBytes(model.final_norm) +
                 QuantizedTensorF32Bytes(model.lm_head);
  for (const LayerWeights& layer : model.layers) {
    // 稠密权重保持原大小，Linear 权重按逻辑元素数统一换算为 F32
    bytes +=
        TensorBytes(layer.attention_norm) + TensorBytes(layer.bq) +
        TensorBytes(layer.bk) + TensorBytes(layer.bv) + TensorBytes(layer.ffn_norm) +
        QuantizedTensorF32Bytes(layer.wq) + QuantizedTensorF32Bytes(layer.wk) +
        QuantizedTensorF32Bytes(layer.wv) + QuantizedTensorF32Bytes(layer.wo) +
        QuantizedTensorF32Bytes(layer.w_gate) + QuantizedTensorF32Bytes(layer.w_up) +
        QuantizedTensorF32Bytes(layer.w_down);
  }
  return bytes;
}

// 构造设备权重快照并保留 CPU 权重；ForwardToken 可借用其中命中的 F32 Linear
// 只有所有上传完成才替换模型引用，任一异常保留原快照
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id) {
  if (!CudaRuntimeBuilt()) {
    throw std::runtime_error("CUDA backend was not built; cannot upload model");
  }
  ValidateModelConfig(model);

  // 所有 Linear 在分配显存前校验，当前上传入口仅接受 F32/Q8_0
  VisitLinearWeights(
      model, [](const QuantizedTensor& weight, const std::string& name,
                const std::vector<int>& expected) {
        ValidateLinearStorage(weight, expected, name);
        if (weight.type != QuantType::kF32 && weight.type != QuantType::kQ80) {
          throw std::runtime_error("CUDA Linear supports only F32/Q8_0: " + name);
        }
      });

  // 明确选择目标设备；各 Buffer 仍记录自己的设备并在操作时进行保护
  CudaGetDeviceInfo(device_id);
  CudaSetDevice(device_id);
  auto storage = std::make_shared<CudaModelWeights>();
  storage->device_id = device_id;
  const auto& c = model.config;
  const int kv_dim = c.n_kv_heads * c.head_dim;
  UploadDenseWeight(*storage, model.token_embedding, "token_embedding",
                    {c.vocab_size, c.dim});
  UploadDenseWeight(*storage, model.final_norm, "final_norm", {c.dim});
  for (size_t i = 0; i < model.layers.size(); ++i) {
    const auto& layer = model.layers[i];
    const std::string prefix = "layers." + std::to_string(i) + ".";
    UploadDenseWeight(*storage, layer.attention_norm, prefix + "attention_norm",
                      {c.dim});
    UploadDenseWeight(*storage, layer.ffn_norm, prefix + "ffn_norm", {c.dim});
    UploadDenseWeight(*storage, layer.bq, prefix + "bq", {c.dim});
    UploadDenseWeight(*storage, layer.bk, prefix + "bk", {kv_dim});
    UploadDenseWeight(*storage, layer.bv, prefix + "bv", {kv_dim});
  }
  VisitLinearWeights(model, [&](const QuantizedTensor& weight,
                                const std::string& name, const std::vector<int>&) {
    CudaUploadedWeight uploaded;
    uploaded.name = name;
    uploaded.type = weight.type;
    uploaded.shape = weight.shape;
    if (weight.type == QuantType::kF32) {
      AppendWeight(*storage, std::move(uploaded), weight.f32_data.data(),
                   StorageBytes(weight.f32_data.size(), sizeof(float), name));
    } else {
      uploaded.block_count = weight.q8_0_data.size();
      AppendWeight(*storage, std::move(uploaded), weight.q8_0_data.data(),
                   StorageBytes(weight.q8_0_data.size(), sizeof(BlockQ80), name));
    }
  });
  // 全部成功后发布快照；此前的分配和上传异常不会替换模型中的旧引用
  model.cuda_weights = std::move(storage);
}

// 解除当前模型的快照引用；其他副本仍持有时不会释放共享的设备权重
void ClearModelCudaWeights(MiniLlamaModel& model) { model.cuda_weights.reset(); }

// 只检查快照引用是否存在，不校验其中权重，也不查询设备或判断推理可用性
bool ModelHasCudaWeights(const MiniLlamaModel& model) {
  return model.cuda_weights != nullptr;
}

// 读取上传总数，包含 Linear 与 embedding/norm/bias；没有快照时返回零
size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::uploaded_weight_count);
}

// 统计快照中的 F32 Linear，不包含同为 F32 的稠密权重
size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model) {
  return UploadedLinearCount(model, QuantType::kF32);
}

// 统计快照中的 Q8_0 Linear
size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model) {
  return UploadedLinearCount(model, QuantType::kQ80);
}

// 统计快照中的 Q4_0 Linear；当前正常上传不会产生该类型
size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model) {
  return UploadedLinearCount(model, QuantType::kQ40);
}

// 统计快照中的 Q4_1 Linear；当前正常上传不会产生该类型
size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model) {
  return UploadedLinearCount(model, QuantType::kQ41);
}

// 读取上传权重数据的显存字节总数，不包含激活、KV Cache 或对象开销
size_t ModelCudaMemoryBytes(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::uploaded_bytes);
}

// 仅清除运行字段，保留上传字段与权重；共享快照的所有模型副本都会看到重置
void ResetModelCudaRuntimeStats(MiniLlamaModel& model) {
  if (!model.cuda_weights) return;
  auto& storage = *model.cuda_weights;
  storage.linear_calls = 0;
  storage.activation_calls = 0;
  storage.attention_calls = 0;
  storage.attention_cpu_fallbacks = 0;
  storage.kv_cache_write_bytes = 0;
  storage.kv_cache_read_bytes = 0;
  storage.host_to_device_copies = 0;
  storage.device_to_host_copies = 0;
  storage.host_to_device_bytes = 0;
  storage.device_to_host_bytes = 0;
}

// 读取已记录的 CUDA Linear 调用数；当前 CPU 前向不会更新该字段
size_t ModelCudaLinearCalls(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::linear_calls);
}

// 读取已记录的 CUDA 激活/辅助算子调用数
size_t ModelCudaActivationCalls(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::activation_calls);
}

// 读取已记录的设备 Attention 调用数；当前尚无设备 Attention 执行入口
size_t ModelCudaAttentionCalls(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::attention_calls);
}

// 读取设备推理期间的 CPU Attention 回退数，不将普通 CPU 前向计作回退
size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::attention_cpu_fallbacks);
}

// 读取已记录的设备 KV Cache 写入字节数
size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::kv_cache_write_bytes);
}

// 读取已记录的设备 KV Cache 读取字节数
size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::kv_cache_read_bytes);
}

// 读取推理阶段已记录的主机到设备拷贝次数，与权重上传总数分开统计
size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::host_to_device_copies);
}

// 读取推理阶段已记录的设备到主机拷贝次数
size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::device_to_host_copies);
}

// 读取推理阶段已记录的主机到设备拷贝字节数
size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::host_to_device_bytes);
}

// 读取推理阶段已记录的设备到主机拷贝字节数
size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model) {
  return RuntimeStat(model, &CudaModelWeights::device_to_host_bytes);
}

// 将所有 Linear 转为 Q8_0；真正改变权重后解除当前模型的旧 CUDA 快照引用
void QuantizeModelToQ80(MiniLlamaModel& model) {
  QuantizeModel(model, QuantType::kQ80);
}
// 将所有 Linear 转为 Q4_0，沿用现有单张量量化规则与统一提交流程
void QuantizeModelToQ40(MiniLlamaModel& model) {
  QuantizeModel(model, QuantType::kQ40);
}

}  // namespace mini_llama
