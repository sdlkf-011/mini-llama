#include "mini_llama/forward.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/cuda_attention.h"
#include "mini_llama/cuda_matmul.h"
#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_quant.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

namespace mini_llama {

// 检查普通张量的逻辑形状，并保留调用阶段信息
static void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                         const char* caller) {
  tensor.AssertShape(expected, caller);
}

// 检查模型配置之间的约束，避免后续头映射、RoPE 和缓存访问越界
static void ValidateConfig(const ModelConfig& config) {
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.head_dim <= 0 || config.max_seq_len <= 0) {
    throw std::runtime_error("ForwardToken: model config values must be positive");
  }
  // 先提升到 64 位再比较，避免非法的大 head 配置在校验时发生 int 乘法溢出
  if (config.dim != static_cast<int64_t>(config.n_heads) * config.head_dim) {
    throw std::runtime_error("ForwardToken: dim must equal n_heads * head_dim");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::runtime_error(
        "ForwardToken: n_heads must be divisible by n_kv_heads");
  }
  if (config.head_dim % 2 != 0) {
    throw std::runtime_error("ForwardToken: head_dim must be even for RoPE");
  }
  if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
    throw std::runtime_error("ForwardToken: rope_theta must be finite and positive");
  }
  if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps < 0.0f) {
    throw std::runtime_error(
        "ForwardToken: rms_norm_eps must be finite and non-negative");
  }
}

// 检查一层 Transformer 的全部权重形状，区分查询头和 GQA 的 KV 头
static void ValidateLayerWeights(const LayerWeights& weights,
                                 const ModelConfig& config, int layer) {
  const std::string prefix = "ForwardToken layer " + std::to_string(layer);
  const int query_dim = config.n_heads * config.head_dim;
  const int kv_dim = config.n_kv_heads * config.head_dim;

  // 注意力投影中 Q 使用全部查询头，K/V 仅使用 KV 头
  RequireShape(weights.attention_norm, {config.dim},
               (prefix + " attention_norm").c_str());
  weights.wq.AssertShape({query_dim, config.dim}, (prefix + " wq").c_str());
  weights.wk.AssertShape({kv_dim, config.dim}, (prefix + " wk").c_str());
  weights.wv.AssertShape({kv_dim, config.dim}, (prefix + " wv").c_str());
  weights.wo.AssertShape({config.dim, query_dim}, (prefix + " wo").c_str());

  // Q/K/V 偏置是可选项，只有实际存在时才要求形状匹配
  if (!weights.bq.data.empty()) {
    RequireShape(weights.bq, {query_dim}, (prefix + " bq").c_str());
  }
  if (!weights.bk.data.empty()) {
    RequireShape(weights.bk, {kv_dim}, (prefix + " bk").c_str());
  }
  if (!weights.bv.data.empty()) {
    RequireShape(weights.bv, {kv_dim}, (prefix + " bv").c_str());
  }

  // FFN 的 gate/up 扩展到 hidden_dim，down 再投影回模型维度
  RequireShape(weights.ffn_norm, {config.dim}, (prefix + " ffn_norm").c_str());
  weights.w_gate.AssertShape({config.hidden_dim, config.dim},
                             (prefix + " w_gate").c_str());
  weights.w_up.AssertShape({config.hidden_dim, config.dim},
                           (prefix + " w_up").c_str());
  weights.w_down.AssertShape({config.dim, config.hidden_dim},
                             (prefix + " w_down").c_str());
}

// 在进入数值计算前集中验证模型状态、输入范围、缓存和全部权重
static void ValidateForwardInputs(const MiniLlamaContext& ctx,
                                  const MiniLlamaModel& model, int token) {
  if (ctx.model != &model) {
    throw std::runtime_error("ForwardToken: context belongs to a different model");
  }
  const ModelConfig& c = model.config;
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }

  // 先验证配置，再使用配置值检查 token、位置和各个张量
  ValidateConfig(c);
  if (token < 0 || token >= c.vocab_size) {
    throw std::out_of_range("ForwardToken token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= c.max_seq_len) {
    throw std::out_of_range("ForwardToken position out of range");
  }
  if (model.layers.size() != static_cast<size_t>(c.n_layers)) {
    throw std::runtime_error("ForwardToken layer count does not match model config");
  }

  if (model.token_embedding.data.size() <
      static_cast<size_t>(c.vocab_size) * c.dim) {
    throw std::runtime_error(
        "ForwardToken token embedding tensor is smaller than c");
  }

  // 模型级张量和上下文缓存必须与配置完全一致
  RequireShape(model.token_embedding, {c.vocab_size, c.dim},
               "ForwardToken token_embedding");
  RequireShape(model.final_norm, {c.dim}, "ForwardToken final_norm");
  model.lm_head.AssertShape({c.vocab_size, c.dim}, "ForwardToken lm_head");
  RequireShape(ctx.kv_cache.keys,
               {c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim},
               "ForwardToken key cache");
  RequireShape(ctx.kv_cache.values,
               {c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim},
               "ForwardToken value cache");

  // 每层权重都在计算开始前检查，避免执行到一半才修改部分 KV Cache
  for (int layer = 0; layer < c.n_layers; ++layer) {
    ValidateLayerWeights(model.layers[layer], c, layer);
  }
}

// 空偏置表示无需加偏置；非空偏置必须与一维投影结果同 shape
// 复制输入后累加，调用方持有的投影结果不会被原地修改
static Tensor AddOptionalBias(const Tensor& x, const Tensor& bias,
                              const char* caller) {
  if (bias.data.empty()) {
    return x;
  }

  if (x.num_dims() != 1 || bias.num_dims() != 1 || x.shape[0] != bias.shape[0]) {
    throw std::runtime_error(std::string(caller) +
                             ": bias shape mismatch x=" + x.ShapeStringShort() +
                             " bias=" + bias.ShapeStringShort());
  }
  bias.AssertShape(x.shape, caller);

  Tensor y = x;
  for (size_t i = 0; i < x.size(); ++i) {
    y.data[i] += bias.data[i];
  }
  return y;
}

// 在发生实际操作之后记录统计，权重上传不计入 Token 前向统计
static const CudaUploadedWeight* FindCudaWeight(const MiniLlamaModel& model,
                                                const std::string& name,
                                                QuantType type) {
  if (!model.cuda_weights) {
    return nullptr;
  }

  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.name == name && weight.type == type) {
      return &weight;
    }
  }
  return nullptr;
}

static void RecordCudaLinearCall(const MiniLlamaModel& model) {
  if (model.cuda_weights) {
    model.cuda_weights->linear_calls += 1;
  }
}

static void RecordCudaAttentionCall(const MiniLlamaModel& model) {
  if (model.cuda_weights) {
    model.cuda_weights->attention_calls += 1;
  }
}

static void RecordCudaAttentionCpuFallback(const MiniLlamaModel& model) {
  if (model.cuda_weights) {
    model.cuda_weights->attention_cpu_fallbacks += 1;
  }
}

static void RecordCudaActivationCall(const MiniLlamaModel& model, size_t calls = 1) {
  if (model.cuda_weights) {
    model.cuda_weights->activation_calls += calls;
  }
}

static void RecordCudaHostToDeviceCopy(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->host_to_device_copies += 1;
  model.cuda_weights->host_to_device_bytes += bytes;
}

static void RecordCudaDeviceToHostCopy(const MiniLlamaModel& model, size_t bytes) {
  if (!model.cuda_weights) {
    return;
  }
  model.cuda_weights->device_to_host_copies += 1;
  model.cuda_weights->device_to_host_bytes += bytes;
}

static void RecordCudaKvCacheWrite(const MiniLlamaModel& model, size_t bytes) {
  if (model.cuda_weights) {
    model.cuda_weights->kv_cache_write_bytes += bytes;
  }
}

static void RecordCudaKvCacheRead(const MiniLlamaModel& model, size_t bytes) {
  if (model.cuda_weights) {
    model.cuda_weights->kv_cache_read_bytes += bytes;
  }
}

static int CudaDeviceId(const MiniLlamaModel& model) {
  return model.cuda_weights ? model.cuda_weights->device_id : 0;
}

static bool CudaLinearSupported(const QuantizedTensor& weight) {
  return weight.type == QuantType::kF32 || weight.type == QuantType::kQ80 ||
         weight.type == QuantType::kQ40 || weight.type == QuantType::kQ41;
}

static bool CudaLinearReady(const MiniLlamaModel& model, const std::string& name,
                            const QuantizedTensor& weight) {
  return model.cuda_weights && CudaLinearSupported(weight) &&
         FindCudaWeight(model, name, weight.type) != nullptr;
}

// 主机边界只在兼容包装中出现；完整 CUDA Token 路径不会上传激活
static CudaTensor UploadCudaTensor(const MiniLlamaModel& model, const Tensor& x) {
  auto device = CudaTensorFromHost(x, CudaDeviceId(model));
  RecordCudaHostToDeviceCopy(model, device.bytes());
  return device;
}
static Tensor DownloadCudaTensor(const MiniLlamaModel& model, const CudaTensor& x) {
  Tensor result = x.Download();
  RecordCudaDeviceToHostCopy(model, result.size() * sizeof(float));
  return result;
}
static void RecordCudaLinearCopy(const MiniLlamaModel& model, const Tensor& x,
                                 const Tensor& y) {
  RecordCudaLinearCall(model);
  RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
  RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
}

static CudaTensor ForwardLinearDevice(const MiniLlamaModel& model,
                                      const std::string& name, const CudaTensor& x,
                                      const QuantizedTensor& weight);

// 保留主机 Linear 包装，按常驻权重类型分发；CUDA 主链路调用设备输入版本
static Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name,
                            const Tensor& x, const QuantizedTensor& weight) {
  if (!model.cuda_weights || !CudaLinearSupported(weight)) {
    return Linear(x, weight);
  }

  const auto* resident = FindCudaWeight(model, name, weight.type);
  if (!resident) {
    throw std::runtime_error("CUDA forward Linear missing uploaded weight: " + name);
  }

  Tensor y;
  switch (weight.type) {
    case QuantType::kF32:
      y = CudaLinearDeviceWeight(x, resident->buffer.data(), resident->shape,
                                 nullptr, CudaDeviceId(model));
      break;
    case QuantType::kQ80:
      y = CudaQ80LinearDeviceWeight(x, resident->buffer.data(),
                                    resident->block_count, resident->shape, nullptr,
                                    CudaDeviceId(model));
      break;
    case QuantType::kQ40:
      y = CudaQ40LinearDeviceWeight(x, resident->buffer.data(),
                                    resident->block_count, resident->shape, nullptr,
                                    CudaDeviceId(model));
      break;
    case QuantType::kQ41:
      y = CudaQ41LinearDeviceWeight(x, resident->buffer.data(),
                                    resident->block_count, resident->shape, nullptr,
                                    CudaDeviceId(model));
      break;
  }
  RecordCudaLinearCopy(model, x, y);
  return y;
}

// Q/K/V 共用归一化输入，分别投影并添加可选偏置；输出保持一维以便按头 reshape
static void ForwardQkvProjection(const MiniLlamaModel& model,
                                 const std::string& layer_prefix, const Tensor& h,
                                 const LayerWeights& lw, Tensor& q_flat,
                                 Tensor& k_flat, Tensor& v_flat) {
  if (CudaLinearReady(model, layer_prefix + "wq", lw.wq) &&
      CudaLinearReady(model, layer_prefix + "wk", lw.wk) &&
      CudaLinearReady(model, layer_prefix + "wv", lw.wv)) {
    auto input = UploadCudaTensor(model, h);
    auto q = ForwardLinearDevice(model, layer_prefix + "wq", input, lw.wq);
    auto k = ForwardLinearDevice(model, layer_prefix + "wk", input, lw.wk);
    auto v = ForwardLinearDevice(model, layer_prefix + "wv", input, lw.wv);
    q_flat = AddOptionalBias(DownloadCudaTensor(model, q), lw.bq, "forward_layer q");
    k_flat = AddOptionalBias(DownloadCudaTensor(model, k), lw.bk, "forward_layer k");
    v_flat = AddOptionalBias(DownloadCudaTensor(model, v), lw.bv, "forward_layer v");
    return;
  }

  q_flat = AddOptionalBias(ForwardLinear(model, layer_prefix + "wq", h, lw.wq),
                           lw.bq, "forward_layer q");
  k_flat = AddOptionalBias(ForwardLinear(model, layer_prefix + "wk", h, lw.wk),
                           lw.bk, "forward_layer k");
  v_flat = AddOptionalBias(ForwardLinear(model, layer_prefix + "wv", h, lw.wv),
                           lw.bv, "forward_layer v");
}

// 主机接口保留后端兼容性，完整设备链路使用下面的 Device 版本
static Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& x,
                             const Tensor& weight, float eps) {
  if (!model.cuda_weights) {
    return RmsNorm(x, weight, eps);
  }

  Tensor y = CudaRmsNorm(x, weight, eps, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
  RecordCudaHostToDeviceCopy(model, weight.size() * sizeof(float));
  RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
  return y;
}

static Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                            const Tensor& up) {
  if (!model.cuda_weights) {
    return SwiGlu(gate, up);
  }

  auto input_gate = UploadCudaTensor(model, gate);
  auto input_up = UploadCudaTensor(model, up);
  auto silu = CudaSiluDeviceInput(input_gate, CudaDeviceId(model));
  auto result = CudaElementwiseMulDeviceInput(silu, input_up, CudaDeviceId(model));
  RecordCudaActivationCall(model, 2);
  return DownloadCudaTensor(model, result);
}

static Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a,
                         const Tensor& b) {
  if (model.cuda_weights) {
    auto da = UploadCudaTensor(model, a);
    auto db = UploadCudaTensor(model, b);
    auto result = CudaElementwiseAddDeviceInput(da, db, CudaDeviceId(model));
    RecordCudaActivationCall(model);
    return DownloadCudaTensor(model, result);
  }

  if (a.shape != b.shape) {
    throw std::runtime_error("forward_add: shape mismatch");
  }

  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) {
    y.data[i] = a.data[i] + b.data[i];
  }
  return y;
}

static Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& x) {
  if (!model.cuda_weights) {
    return Softmax(x);
  }

  Tensor y = CudaSoftmax(x, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  RecordCudaHostToDeviceCopy(model, x.size() * sizeof(float));
  RecordCudaDeviceToHostCopy(model, y.size() * sizeof(float));
  return y;
}

// 主机 RoPE 包装供兼容路径使用，V 不参与旋转
static void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k, int pos,
                        float theta, RopeType rope_type) {
  if (!model.cuda_weights) {
    Rope(q, k, pos, theta, rope_type);
    return;
  }

  CudaRope(q, k, pos, theta, rope_type, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  RecordCudaHostToDeviceCopy(model, q.size() * sizeof(float));
  RecordCudaHostToDeviceCopy(model, k.size() * sizeof(float));
  RecordCudaDeviceToHostCopy(model, q.size() * sizeof(float));
  RecordCudaDeviceToHostCopy(model, k.size() * sizeof(float));
}

// 按 token_id 查找嵌入表，得到 x[dim]
static Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
  const int dim = model.config.dim;
  const size_t offset = static_cast<size_t>(token_id) * static_cast<size_t>(dim);
  Tensor x({dim}, 0.0f);

  for (int i = 0; i < dim; ++i) {
    x.data[i] = model.token_embedding.data[offset + i];
  }
  return x;
}

// GQA: 同一组查询头共享一个 KV 头
static int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  return q_head / (n_heads / n_kv_heads);
}

// q: [n_heads, head_dim], k/v: [n_kv_heads, head_dim]
// CPU 路径使用主机缓存和 CPU 注意力。
static Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q,
                               const Tensor& k, const Tensor& v, int pos, int layer,
                               KvCache& kv_cache, CudaKvCache* cuda_kv_cache,
                               int n_heads, int n_kv_heads, int head_dim) {
  // 写入当前位置后才读 [0, pos]，既包含当前 token，也保持因果约束
  kv_cache.Write(layer, pos, k, v);
  if (cuda_kv_cache != nullptr && !cuda_kv_cache->empty()) {
    cuda_kv_cache->Write(layer, pos, k, v);
    RecordCudaHostToDeviceCopy(model, k.size() * sizeof(float));
    RecordCudaHostToDeviceCopy(model, v.size() * sizeof(float));
    RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
    auto out = CudaAttentionDecode(q, *cuda_kv_cache, layer, pos, n_heads,
                                   n_kv_heads, head_dim, CudaDeviceId(model));
    RecordCudaAttentionCall(model);
    RecordCudaKvCacheRead(model, 2 * static_cast<size_t>(pos + 1) * n_heads *
                                     head_dim * sizeof(float));
    RecordCudaHostToDeviceCopy(model, q.size() * sizeof(float));
    RecordCudaDeviceToHostCopy(model, out.size() * sizeof(float));
    return out;
  }

  if (model.cuda_weights) {
    RecordCudaAttentionCpuFallback(model);
  }

  Tensor attn_out({n_heads, head_dim}, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // 每个查询头独立持有临时 scores/probs，并只写自己对应的输出行
  auto compute_head = [&](int h) {
    const int kv_head = MapQHeadToKvHead(h, n_heads, n_kv_heads);
    Tensor scores({pos + 1}, 0.0f);
    for (int t = 0; t <= pos; ++t) {
      const float* k_ptr = kv_cache.KeyPtr(layer, t, kv_head);
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q.data[h * head_dim + d] * k_ptr[d];
      }
      scores.data[t] = dot * scale;
    }

    // 对历史位置归一化，再用概率加权同一个 KV 头中的 V
    Tensor probs = ForwardSoftmax(model, scores);
    for (int d = 0; d < head_dim; ++d) {
      float val = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        const float* v_ptr = kv_cache.valuePtr(layer, t, kv_head);
        val += probs.data[t] * v_ptr[d];
      }
      attn_out.data[h * head_dim + d] = val;
    }
  };

  if (model.cuda_weights) {
    for (int h = 0; h < n_heads; ++h) {
      compute_head(h);
    }
    return attn_out;
  }

  // GQA 的多个查询头可只读同一 KV 头；此阶段不再写共享缓存
  ParallelFor(n_heads, [&](int begin, int end) {
    for (int h = begin; h < end; ++h) {
      compute_head(h);
    }
  });
  return attn_out;
}

// SwiGLU FFN: gate/up -> SiLU(gate) * up -> down
static Tensor FfnForward(const MiniLlamaModel& model,
                         const std::string& layer_prefix, const Tensor& h,
                         const LayerWeights& lw) {
  if (CudaLinearReady(model, layer_prefix + "w_gate", lw.w_gate) &&
      CudaLinearReady(model, layer_prefix + "w_up", lw.w_up) &&
      CudaLinearReady(model, layer_prefix + "w_down", lw.w_down)) {
    auto input = UploadCudaTensor(model, h);
    auto gate =
        ForwardLinearDevice(model, layer_prefix + "w_gate", input, lw.w_gate);
    auto up = ForwardLinearDevice(model, layer_prefix + "w_up", input, lw.w_up);
    auto silu = CudaSiluDeviceInput(gate, CudaDeviceId(model));
    auto ff = CudaElementwiseMulDeviceInput(silu, up, CudaDeviceId(model));
    RecordCudaActivationCall(model, 2);
    auto out = ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
    return DownloadCudaTensor(model, out);
  }

  Tensor gate = ForwardLinear(model, layer_prefix + "w_gate", h, lw.w_gate);
  Tensor up = ForwardLinear(model, layer_prefix + "w_up", h, lw.w_up);
  return ForwardLinear(model, layer_prefix + "w_down",
                       ForwardSwiGlu(model, gate, up), lw.w_down);
}

// 单层 Pre-Norm Transformer: Attention + residual, FFN + residual
static Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                           const Tensor& x, int layer, const LayerWeights& lw,
                           const ModelConfig& c) {
  const int dim = c.dim;
  const int n_heads = c.n_heads;
  const int n_kv_heads = c.n_kv_heads;
  const int head_dim = c.head_dim;
  const int pos = ctx.pos;
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  // Attention：Pre-Norm -> Q/K/V 与 bias -> 分头 -> RoPE -> 因果 GQA
  Tensor h = ForwardRmsNorm(model, x, lw.attention_norm, c.rms_norm_eps);
  Tensor q_flat, k_flat, v_flat;
  ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);
  Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
  Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
  Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");
  ForwardRope(model, q, k, pos, c.rope_theta, c.rope_type);
  Tensor attn_out =
      AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache,
                       model.cuda_weights ? &ctx.cuda_kv_cache : nullptr, n_heads,
                       n_kv_heads, head_dim);

  // 拼接查询头输出，执行 O 投影并加回本层输入，形成第一次残差
  Tensor attn_out_flat =
      attn_out.ReshapeChecked({1, n_heads * head_dim}, "forward_layer attn_out");
  Tensor attn_proj = ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo);
  RequireShape(attn_proj, {1, dim}, "forward_layer attn_proj");
  Tensor attn_proj_1d =
      attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");
  Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

  // FFN：第二次 Pre-Norm -> Gate/Up -> SwiGLU -> Down -> 第二次残差
  Tensor h2 = ForwardRmsNorm(model, x_attn, lw.ffn_norm, c.rms_norm_eps);
  Tensor ff = FfnForward(model, layer_prefix, h2, lw);
  RequireShape(ff, {dim}, "forward_layer ffn");
  return ForwardAdd(model, x_attn, ff);
}

// 最终 RMSNorm + LM Head，返回 [vocab_size] logits
static Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                            const ModelConfig& c) {
  Tensor normed = ForwardRmsNorm(model, x, model.final_norm, c.rms_norm_eps);
  Tensor logits = ForwardLinear(model, "lm_head", normed, model.lm_head);
  return logits.ReshapeChecked({c.vocab_size}, "compute_logits");
}

// 在任何缓存写入之前校验完整快照；不允许缺失权重触发 CPU 回退
static const CudaUploadedWeight& RequireCudaWeight(const MiniLlamaModel& model,
                                                   const std::string& name,
                                                   const std::vector<int>& shape,
                                                   QuantType type, bool linear) {
  for (const auto& w : model.cuda_weights->weights) {
    if (w.name != name) {
      continue;
    }

    size_t count = 1;
    for (int d : shape) {
      count *= static_cast<size_t>(d);
    }

    size_t bytes = count * sizeof(float);
    size_t blocks = 0;

    if (linear && type != QuantType::kF32) {
      blocks = static_cast<size_t>(shape[0]) *
               ((static_cast<size_t>(shape[1]) + 31) / 32);
      switch (type) {
        case QuantType::kQ80:
          bytes = blocks * sizeof(BlockQ80);
          break;
        case QuantType::kQ40:
          bytes = blocks * sizeof(BlockQ40);
          break;
        case QuantType::kQ41:
          bytes = blocks * sizeof(BlockQ41);
          break;
        default:
          throw std::runtime_error("ForwardToken: unsupported CUDA type");
      }
    }

    if (w.shape != shape || w.type != type || w.linear != linear ||
        w.buffer.empty() || w.buffer.bytes() != bytes || w.block_count != blocks ||
        w.buffer.device_id() != model.cuda_weights->device_id) {
      throw std::runtime_error("ForwardToken: CUDA weight mismatch: " + name);
    }
    return w;
  }
  throw std::runtime_error("ForwardToken: missing CUDA weight: " + name);
}

static void ValidateCudaWeights(const MiniLlamaModel& model) {
  auto dense = [&](const std::string& name, const Tensor& w) {
    RequireCudaWeight(model, name, w.shape, QuantType::kF32, false);
  };

  auto linear = [&](const std::string& name, const QuantizedTensor& w) {
    RequireCudaWeight(model, name, w.shape, w.type, true);
  };

  dense("token_embedding", model.token_embedding);
  dense("final_norm", model.final_norm);
  linear("lm_head", model.lm_head);

  for (int l = 0; l < model.config.n_layers; ++l) {
    const auto& w = model.layers[l];
    const std::string p = "layers." + std::to_string(l) + ".";
    dense(p + "attention_norm", w.attention_norm);
    dense(p + "ffn_norm", w.ffn_norm);
    linear(p + "wq", w.wq);
    linear(p + "wk", w.wk);
    linear(p + "wv", w.wv);
    linear(p + "wo", w.wo);
    linear(p + "w_gate", w.w_gate);
    linear(p + "w_up", w.w_up);
    linear(p + "w_down", w.w_down);

    if (!w.bq.data.empty()) {
      dense(p + "bq", w.bq);
    }
    if (!w.bk.data.empty()) {
      dense(p + "bk", w.bk);
    }
    if (!w.bv.data.empty()) {
      dense(p + "bv", w.bv);
    }
  }
}

// 首次分配仅允许从零开始；已有缓存校验配置，避免重复 Reset 丢失历史
static void PrepareCudaCache(MiniLlamaContext& ctx, const MiniLlamaModel& model) {
  const auto& c = model.config;
  auto& cache = ctx.cuda_kv_cache;
  if (cache.empty()) {
    if (ctx.pos != 0) {
      throw std::runtime_error("ForwardToken: missing GPU history; start at zero");
    }
    cache.Reset(c.n_layers, c.max_seq_len, c.n_kv_heads, c.head_dim,
                CudaDeviceId(model));
  }

  if (cache.device_id() != CudaDeviceId(model) || cache.n_layers() != c.n_layers ||
      cache.max_seq_len() != c.max_seq_len || cache.n_kv_heads() != c.n_kv_heads ||
      cache.head_dim() != c.head_dim) {
    throw std::runtime_error("ForwardToken: CUDA cache configuration mismatch");
  }

  if (!ctx.token_history.empty() &&
      static_cast<size_t>(ctx.pos) > ctx.token_history.size()) {
    throw std::runtime_error("ForwardToken: CUDA cache has no history for position");
  }
}

static CudaTensor EmbedTokenDevice(const MiniLlamaModel& model, int token) {
  const auto& w = RequireCudaWeight(
      model, "token_embedding", model.token_embedding.shape, QuantType::kF32, false);
  auto result = CudaEmbeddingLookupDeviceWeight(w.buffer.data(), w.shape, token,
                                                model.cuda_weights->device_id);
  return result;
}

static CudaTensor ForwardLinearDevice(const MiniLlamaModel& model,
                                      const std::string& name, const CudaTensor& x,
                                      const QuantizedTensor& host) {
  if (!CudaLinearSupported(host)) {
    throw std::runtime_error("CUDA forward Linear unsupported weight type: " + name);
  }

  const auto& w = RequireCudaWeight(model, name, host.shape, host.type, true);
  const int device = model.cuda_weights->device_id;
  CudaTensor result;
  switch (w.type) {
    case QuantType::kF32:
      result = CudaLinearDeviceInput(x, w.buffer.data(), w.shape, device);
      break;
    case QuantType::kQ80:
      result = CudaQ80LinearDeviceInput(x, w.buffer.data(), w.block_count, w.shape,
                                        device);
      break;
    case QuantType::kQ40:
      result = CudaQ40LinearDeviceInput(x, w.buffer.data(), w.block_count, w.shape,
                                        device);
      break;
    case QuantType::kQ41:
      result = CudaQ41LinearDeviceInput(x, w.buffer.data(), w.block_count, w.shape,
                                        device);
      break;
    default:
      throw std::runtime_error("ForwardToken: unsupported CUDA Linear");
  }
  RecordCudaLinearCall(model);
  return result;
}

static CudaTensor ForwardRmsNormDevice(const MiniLlamaModel& model,
                                       const std::string& name, const CudaTensor& x,
                                       const Tensor& host, float eps) {
  const auto& w = RequireCudaWeight(model, name, host.shape, QuantType::kF32, false);
  auto result =
      CudaRmsNormDeviceWeight(x, w.buffer.data(), w.shape, eps, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return result;
}

static CudaTensor AddOptionalBiasDevice(const MiniLlamaModel& model,
                                        const std::string& name, CudaTensor x,
                                        const Tensor& bias) {
  if (bias.data.empty()) {
    return x;
  }
  const auto& w = RequireCudaWeight(model, name, bias.shape, QuantType::kF32, false);
  auto result = CudaElementwiseAddDeviceWeight(x, w.buffer.data(), w.shape,
                                               model.cuda_weights->device_id);
  RecordCudaActivationCall(model);
  return result;
}

// 设备 Q/K/V 投影及可选 Bias，返回独占显存的扁平向量
static void ForwardQkvProjectionDevice(const MiniLlamaModel& model,
                                       const std::string& layer_prefix,
                                       const CudaTensor& h, const LayerWeights& lw,
                                       CudaTensor& q, CudaTensor& k, CudaTensor& v) {
  q = ForwardLinearDevice(model, layer_prefix + "wq", h, lw.wq);
  k = ForwardLinearDevice(model, layer_prefix + "wk", h, lw.wk);
  v = ForwardLinearDevice(model, layer_prefix + "wv", h, lw.wv);
  q = AddOptionalBiasDevice(model, layer_prefix + "bq", std::move(q), lw.bq);
  k = AddOptionalBiasDevice(model, layer_prefix + "bk", std::move(k), lw.bk);
  v = AddOptionalBiasDevice(model, layer_prefix + "bv", std::move(v), lw.bv);
}

static CudaTensor ForwardAddDevice(const MiniLlamaModel& model, const CudaTensor& a,
                                   const CudaTensor& b) {
  auto y = CudaElementwiseAddDeviceInput(a, b, CudaDeviceId(model));
  RecordCudaActivationCall(model);
  return y;
}

// 同一次接口调用旋转 Q/K，使用已有 Normal/NeoX Kernel
static void ForwardRopeDevice(const MiniLlamaModel& model, CudaTensor& q,
                              CudaTensor& k, int n_heads, int n_kv_heads,
                              int head_dim, int pos, float theta,
                              RopeType rope_type) {
  CudaRopeDeviceInput(q, k, n_heads, n_kv_heads, head_dim, pos, theta, rope_type,
                      CudaDeviceId(model));
  RecordCudaActivationCall(model);
}

// 写入当前 K/V 后直接读取设备历史，不执行 CPU Attention 回退
static CudaTensor AttentionForwardDevice(const MiniLlamaModel& model,
                                         const CudaTensor& q, const CudaTensor& k,
                                         const CudaTensor& v, int pos, int layer,
                                         CudaKvCache& cuda_kv_cache, int n_heads,
                                         int n_kv_heads, int head_dim) {
  cuda_kv_cache.WriteDevice(layer, pos, k, v);
  RecordCudaKvCacheWrite(model, (k.size() + v.size()) * sizeof(float));
  auto out =
      CudaAttentionDecodeDeviceInput(q, cuda_kv_cache, layer, pos, n_heads,
                                     n_kv_heads, head_dim, CudaDeviceId(model));
  RecordCudaAttentionCall(model);

  // GQA 共享数据按各查询头逻辑访问计入，不等同于物理 DRAM 流量
  RecordCudaKvCacheRead(
      model, 2 * static_cast<size_t>(pos + 1) * n_heads * head_dim * sizeof(float));
  return out;
}

// FFN 的 Gate/Up、SiLU × Up 与 Down 的输入输出全程驻留 GPU
static CudaTensor FfnForwardDevice(const MiniLlamaModel& model,
                                   const std::string& layer_prefix,
                                   const CudaTensor& h, const LayerWeights& lw) {
  auto gate = ForwardLinearDevice(model, layer_prefix + "w_gate", h, lw.w_gate);
  auto up = ForwardLinearDevice(model, layer_prefix + "w_up", h, lw.w_up);
  auto silu_gate = CudaSiluDeviceInput(gate, CudaDeviceId(model));
  auto ff = CudaElementwiseMulDeviceInput(silu_gate, up, CudaDeviceId(model));
  RecordCudaActivationCall(model, 2);
  return ForwardLinearDevice(model, layer_prefix + "w_down", ff, lw.w_down);
}

// 单层 Pre-Norm Transformer，保留 Attention 与 FFN 的两次残差
static CudaTensor ForwardLayerDevice(MiniLlamaContext& ctx,
                                     const MiniLlamaModel& model,
                                     const CudaTensor& x, int layer,
                                     const LayerWeights& lw, const ModelConfig& c) {
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  auto h = ForwardRmsNormDevice(model, prefix + "attention_norm", x,
                                lw.attention_norm, c.rms_norm_eps);
  CudaTensor q, k, v;
  ForwardQkvProjectionDevice(model, prefix, h, lw, q, k, v);
  ForwardRopeDevice(model, q, k, c.n_heads, c.n_kv_heads, c.head_dim, ctx.pos,
                    c.rope_theta, c.rope_type);
  auto attention =
      AttentionForwardDevice(model, q, k, v, ctx.pos, layer, ctx.cuda_kv_cache,
                             c.n_heads, c.n_kv_heads, c.head_dim);
  auto projection = ForwardLinearDevice(model, prefix + "wo", attention, lw.wo);

  if (projection.size() != static_cast<size_t>(c.dim)) {
    throw std::runtime_error(
        "forward_layer_device: attention projection shape mismatch");
  }

  auto x_attn = ForwardAddDevice(model, x, projection);
  auto h2 = ForwardRmsNormDevice(model, prefix + "ffn_norm", x_attn, lw.ffn_norm,
                                 c.rms_norm_eps);
  auto ff = FfnForwardDevice(model, prefix, h2, lw);
  if (ff.size() != static_cast<size_t>(c.dim)) {
    throw std::runtime_error("forward_layer_device: FFN output shape mismatch");
  }
  return ForwardAddDevice(model, x_attn, ff);
}

// 最终 Norm 和 LM Head 保留在 GPU，只下载 [vocab_size] F32 Logits
static Tensor ComputeLogitsDevice(const CudaTensor& x, const MiniLlamaModel& model,
                                  const ModelConfig& c) {
  auto normed =
      ForwardRmsNormDevice(model, "final_norm", x, model.final_norm, c.rms_norm_eps);
  auto logits = ForwardLinearDevice(model, "lm_head", normed, model.lm_head);
  return DownloadCudaTensor(model, logits)
      .ReshapeChecked({c.vocab_size}, "compute_logits_device");
}

// 处理 ctx.pos 指定的单个 token，返回 [vocab_size] logits 并更新各层 K/V
// 位置推进和 token_history 维护交由 ForwardBatch 或直接调用方负责
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
  ValidateForwardInputs(ctx, model, token);
  if (model.cuda_weights) {
    ValidateCudaWeights(model);
    PrepareCudaCache(ctx, model);
    auto x = EmbedTokenDevice(model, token);
    for (int layer = 0; layer < model.config.n_layers; ++layer) {
      x = ForwardLayerDevice(ctx, model, x, layer, model.layers[layer],
                             model.config);
    }
    return ComputeLogitsDevice(x, model, model.config);
  }

  if (!ctx.cuda_kv_cache.empty()) {
    throw std::runtime_error(
        "ForwardToken: GPU history requires CUDA weights; create a CPU context");
  }

  const ModelConfig& c = model.config;
  Tensor x = EmbedToken(model, token);
  for (int layer = 0; layer < c.n_layers; ++layer) {
    x = ForwardLayer(ctx, model, x, layer, model.layers[layer], c);
  }
  return ComputeLogits(x, model, c);
}

}  // namespace mini_llama
