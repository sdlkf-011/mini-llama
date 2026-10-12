#include "mini_llama/model.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "mini_llama/quant.h"

namespace mini_llama {

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
static size_t QuantizedTensorBytesF32(const QuantizedTensor& qt) {
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
                 QuantizedTensorBytesF32(model.lm_head);
  for (const LayerWeights& layer : model.layers) {
    // 稠密权重保持原大小，Linear 权重按逻辑元素数统一换算为 F32
    bytes +=
        TensorBytes(layer.attention_norm) + TensorBytes(layer.bq) +
        TensorBytes(layer.bk) + TensorBytes(layer.bv) + TensorBytes(layer.ffn_norm) +
        QuantizedTensorBytesF32(layer.wq) + QuantizedTensorBytesF32(layer.wk) +
        QuantizedTensorBytesF32(layer.wv) + QuantizedTensorBytesF32(layer.wo) +
        QuantizedTensorBytesF32(layer.w_gate) + QuantizedTensorBytesF32(layer.w_up) +
        QuantizedTensorBytesF32(layer.w_down);
  }
  return bytes;
}

// 要求非空且各维为正，并在逐维相乘前检查 size_t 溢出
static size_t WeightNumel(const std::vector<int>& shape, const std::string& name) {
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
static size_t StorageBytes(size_t count, size_t element_size,
                           const std::string& name) {
  if (count > std::numeric_limits<size_t>::max() / element_size) {
    throw std::runtime_error("Model weight byte count overflow: " + name);
  }
  return count * element_size;
}

// 上传与重新量化共用的入口校验：已加载、正维度、GQA/RoPE 约束及层数一致
static void ValidateModelConfig(const MiniLlamaModel& model) {
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
static void VisitLinearWeights(Model& model, Visit visit) {
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
static void ValidateLinearStorage(const QuantizedTensor& weight,
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

// 先完成上传再发布权重记录；上传失败由局部 Buffer 回收显存
static void AppendWeight(CudaModelWeights& storage, CudaUploadedWeight uploaded,
                         const void* data, size_t bytes, int device_id) {
  if (bytes > std::numeric_limits<size_t>::max() - storage.uploaded_bytes) {
    throw std::runtime_error("CUDA model weight byte count overflow");
  }
  uploaded.buffer.Reset(bytes, device_id);
  uploaded.buffer.Upload(data, bytes);
  storage.weights.push_back(std::move(uploaded));
  storage.uploaded_weight_count += 1;
  storage.uploaded_bytes += bytes;
}

// F32 Linear 保持原始行主序 [out,in]，不执行重排或量化
static void UploadF32LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
  if (weight.type != QuantType::kF32) {
    return;
  }
  ValidateLinearStorage(weight, weight.shape, name);
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kF32;
  uploaded.linear = true;
  uploaded.shape = weight.shape;
  uploaded.block_count = 0;
  AppendWeight(storage, std::move(uploaded), weight.f32_data.data(),
               StorageBytes(weight.f32_data.size(), sizeof(float), name), device_id);
}

// Embedding/Norm/Bias 均为 F32；只有可选 Bias 可以用空存储表示省略
static void UploadF32TensorWeight(CudaModelWeights& storage, const Tensor& weight,
                                  const std::string& name, int device_id) {
  const bool optional =
      name.size() >= 3 && (name.compare(name.size() - 3, 3, ".bq") == 0 ||
                           name.compare(name.size() - 3, 3, ".bk") == 0 ||
                           name.compare(name.size() - 3, 3, ".bv") == 0);
  if (optional && weight.data.empty()) {
    return;
  }
  const size_t expected_numel = WeightNumel(weight.shape, name);
  if (weight.data.size() != expected_numel) {
    throw std::runtime_error("CUDA weight upload: tensor data size mismatch for " +
                             name);
  }
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kF32;
  uploaded.linear = false;
  uploaded.shape = weight.shape;
  uploaded.block_count = 0;
  AppendWeight(storage, std::move(uploaded), weight.data.data(),
               StorageBytes(expected_numel, sizeof(float), name), device_id);
}

// Q8_0 每个输出行独立补齐尾块，上传原始块并记录真实块数
static void UploadQ80LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
  if (weight.type != QuantType::kQ80) {
    return;
  }
  ValidateLinearStorage(weight, weight.shape, name);
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kQ80;
  uploaded.linear = true;
  uploaded.shape = weight.shape;
  uploaded.block_count = weight.q8_0_data.size();
  AppendWeight(storage, std::move(uploaded), weight.q8_0_data.data(),
               StorageBytes(weight.q8_0_data.size(), sizeof(BlockQ80), name),
               device_id);
}

// Q4_0 每个输出行独立补齐尾块，上传原始块并记录真实块数
static void UploadQ40LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
  if (weight.type != QuantType::kQ40) {
    return;
  }
  ValidateLinearStorage(weight, weight.shape, name);
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kQ40;
  uploaded.linear = true;
  uploaded.shape = weight.shape;
  uploaded.block_count = weight.q4_0_data.size();
  AppendWeight(storage, std::move(uploaded), weight.q4_0_data.data(),
               StorageBytes(weight.q4_0_data.size(), sizeof(BlockQ40), name),
               device_id);
}

// Q4_1 每个输出行独立补齐尾块，上传原始块并记录真实块数
static void UploadQ41LinearWeight(CudaModelWeights& storage,
                                  const QuantizedTensor& weight,
                                  const std::string& name, int device_id) {
  if (weight.type != QuantType::kQ41) {
    return;
  }
  ValidateLinearStorage(weight, weight.shape, name);
  CudaUploadedWeight uploaded;
  uploaded.name = name;
  uploaded.type = QuantType::kQ41;
  uploaded.linear = true;
  uploaded.shape = weight.shape;
  uploaded.block_count = weight.q4_1_data.size();
  AppendWeight(storage, std::move(uploaded), weight.q4_1_data.data(),
               StorageBytes(weight.q4_1_data.size(), sizeof(BlockQ41), name),
               device_id);
}

// 按活动类型分发；未知枚举必须拒绝，不能跳过必需权重
static void UploadLinearWeight(CudaModelWeights& storage,
                               const QuantizedTensor& weight,
                               const std::string& name, int device_id) {
  switch (weight.type) {
    case QuantType::kF32:
      UploadF32LinearWeight(storage, weight, name, device_id);
      return;
    case QuantType::kQ80:
      UploadQ80LinearWeight(storage, weight, name, device_id);
      return;
    case QuantType::kQ40:
      UploadQ40LinearWeight(storage, weight, name, device_id);
      return;
    case QuantType::kQ41:
      UploadQ41LinearWeight(storage, weight, name, device_id);
      return;
  }
  throw std::runtime_error("CUDA weight upload: unknown quant type for " + name);
}

// 所有权重在分配之前预检；完整上传成功才替换旧快照，CPU 副本保留
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id) {
  if (!model.loaded) {
    throw std::runtime_error("CUDA weight upload requires a loaded model");
  }
  if (!CudaRuntimeBuilt()) {
    throw std::runtime_error("CUDA backend was not built; cannot upload model");
  }
  ValidateModelConfig(model);
  VisitLinearWeights(model, [](const QuantizedTensor& w, const std::string& name,
                               const std::vector<int>& expected) {
    ValidateLinearStorage(w, expected, name);
  });
  const auto& c = model.config;
  model.token_embedding.AssertShape({c.vocab_size, c.dim}, "CUDA token_embedding");
  model.final_norm.AssertShape({c.dim}, "CUDA final_norm");
  for (const auto& lw : model.layers) {
    lw.attention_norm.AssertShape({c.dim}, "CUDA attention_norm");
    lw.ffn_norm.AssertShape({c.dim}, "CUDA ffn_norm");
    if (!lw.bq.data.empty()) {
      lw.bq.AssertShape({c.dim}, "CUDA bq");
    }
    if (!lw.bk.data.empty()) {
      lw.bk.AssertShape({c.n_kv_heads * c.head_dim}, "CUDA bk");
    }
    if (!lw.bv.data.empty()) {
      lw.bv.AssertShape({c.n_kv_heads * c.head_dim}, "CUDA bv");
    }
  }
  CudaSetDevice(device_id);
  auto storage = std::make_shared<CudaModelWeights>();
  storage->device_id = device_id;
  UploadF32TensorWeight(*storage, model.token_embedding, "token_embedding",
                        device_id);
  UploadF32TensorWeight(*storage, model.final_norm, "final_norm", device_id);
  UploadLinearWeight(*storage, model.lm_head, "lm_head", device_id);
  for (size_t layer = 0; layer < model.layers.size(); ++layer) {
    const auto& lw = model.layers[layer];
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    UploadF32TensorWeight(*storage, lw.attention_norm, prefix + "attention_norm",
                          device_id);
    UploadLinearWeight(*storage, lw.wq, prefix + "wq", device_id);
    UploadLinearWeight(*storage, lw.wk, prefix + "wk", device_id);
    UploadLinearWeight(*storage, lw.wv, prefix + "wv", device_id);
    UploadF32TensorWeight(*storage, lw.bq, prefix + "bq", device_id);
    UploadF32TensorWeight(*storage, lw.bk, prefix + "bk", device_id);
    UploadF32TensorWeight(*storage, lw.bv, prefix + "bv", device_id);
    UploadLinearWeight(*storage, lw.wo, prefix + "wo", device_id);
    UploadF32TensorWeight(*storage, lw.ffn_norm, prefix + "ffn_norm", device_id);
    UploadLinearWeight(*storage, lw.w_gate, prefix + "w_gate", device_id);
    UploadLinearWeight(*storage, lw.w_up, prefix + "w_up", device_id);
    UploadLinearWeight(*storage, lw.w_down, prefix + "w_down", device_id);
  }
  model.cuda_weights = std::move(storage);
}

// 解除当前引用，其他模型副本仍可继续持有同一快照
void ClearModelCudaWeights(MiniLlamaModel& model) { model.cuda_weights.reset(); }
bool ModelHasCudaWeights(const MiniLlamaModel& model) {
  return model.cuda_weights != nullptr;
}
size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->uploaded_weight_count : 0;
}

// 分类型数量只统计 Linear，不包含 Embedding/Norm/Bias
static size_t ModelCudaUploadedWeightCountByType(const MiniLlamaModel& model,
                                                 QuantType type) {
  if (!model.cuda_weights) {
    return 0;
  }
  size_t count = 0;
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.linear && weight.type == type) {
      count += 1;
    }
  }
  return count;
}
size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kF32);
}
size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kQ80);
}
size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kQ40);
}
size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model) {
  return ModelCudaUploadedWeightCountByType(model, QuantType::kQ41);
}
size_t ModelCudaMemoryBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->uploaded_bytes : 0;
}

// 清零执行统计，保留权重、上传统计和显存所有权
void ResetModelCudaRuntimeStats(const MiniLlamaModel& model) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->linear_calls = 0;
  model.cuda_weights->activation_calls = 0;
  model.cuda_weights->attention_calls = 0;
  model.cuda_weights->attention_cpu_fallbacks = 0;
  model.cuda_weights->kv_cache_write_bytes = 0;
  model.cuda_weights->kv_cache_read_bytes = 0;
  model.cuda_weights->host_to_device_copies = 0;
  model.cuda_weights->device_to_host_copies = 0;
  model.cuda_weights->host_to_device_bytes = 0;
  model.cuda_weights->device_to_host_bytes = 0;
}

// 没有 CUDA 快照时统一返回零，便于 CPU 后端共用查询接口
size_t ModelCudaLinearCalls(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->linear_calls : 0;
}
size_t ModelCudaActivationCalls(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->activation_calls : 0;
}
size_t ModelCudaAttentionCalls(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->attention_calls : 0;
}
size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->attention_cpu_fallbacks : 0;
}
size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->kv_cache_write_bytes : 0;
}
size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->kv_cache_read_bytes : 0;
}
size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->host_to_device_copies : 0;
}
size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_to_host_copies : 0;
}
size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->host_to_device_bytes : 0;
}
size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_to_host_bytes : 0;
}

// 重新量化时将活动存储转换为完整 F32 临时张量；仅用于权重转换
static Tensor QuantizedTensorToF32(const QuantizedTensor& weight) {
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

// 只转换当前张量的活动存储，保留逻辑形状并释放旧格式容量
static void QuantizeQtToQ80(QuantizedTensor& qt) {
  if (qt.type == QuantType::kQ80) {
    return;
  }
  const Tensor t = QuantizedTensorToF32(qt);
  auto blocks = QuantizeToQ80(t);
  qt.q8_0_data = std::move(blocks);
  qt.type = QuantType::kQ80;
  qt.f32_data.clear();
  qt.f32_data.shrink_to_fit();
  qt.q4_0_data.clear();
  qt.q4_0_data.shrink_to_fit();
  qt.q4_1_data.clear();
  qt.q4_1_data.shrink_to_fit();
}

// 只转换当前张量的活动存储，保留逻辑形状并释放旧格式容量
static void QuantizeQtToQ40(QuantizedTensor& qt) {
  if (qt.type == QuantType::kQ40) {
    return;
  }
  const Tensor t = QuantizedTensorToF32(qt);
  auto blocks = QuantizeToQ40(t);
  qt.q4_0_data = std::move(blocks);
  qt.type = QuantType::kQ40;
  qt.f32_data.clear();
  qt.f32_data.shrink_to_fit();
  qt.q8_0_data.clear();
  qt.q8_0_data.shrink_to_fit();
  qt.q4_1_data.clear();
  qt.q4_1_data.shrink_to_fit();
}

// 先为全部 Linear 准备转换结果，任一失败不改变原模型或 CUDA 快照
static void QuantizeModel(MiniLlamaModel& model, QuantType target) {
  ValidateModelConfig(model);
  std::vector<std::pair<QuantizedTensor*, QuantizedTensor>> replacements;
  VisitLinearWeights(model, [&](QuantizedTensor& weight, const std::string& name,
                                const std::vector<int>& expected) {
    ValidateLinearStorage(weight, expected, name);
    if (weight.type == target) {
      return;
    }
    QuantizedTensor converted = weight;
    if (target == QuantType::kQ80) {
      QuantizeQtToQ80(converted);
    } else {
      QuantizeQtToQ40(converted);
    }
    replacements.emplace_back(&weight, std::move(converted));
  });
  if (replacements.empty()) {
    return;
  }
  for (auto& r : replacements) {
    *r.first = std::move(r.second);
  }
  // CPU 权重已变化，旧显存快照必须解除，避免后续前向读取过期数据
  ClearModelCudaWeights(model);
}

void QuantizeModelToQ80(MiniLlamaModel& model) {
  QuantizeModel(model, QuantType::kQ80);
}
void QuantizeModelToQ40(MiniLlamaModel& model) {
  QuantizeModel(model, QuantType::kQ40);
}

}  // namespace mini_llama
