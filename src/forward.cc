#include "mini_llama/forward.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_matmul.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

namespace mini_llama {
namespace {

// 检查普通张量的逻辑形状，并保留调用阶段信息
static void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                         const char* caller) {
  tensor.AssertShape(expected, caller);
}

// 检查模型配置之间的约束，避免后续头映射、RoPE 和缓存访问越界
void ValidateConfig(const ModelConfig& config) {
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
void ValidateLayerWeights(const LayerWeights& weights, const ModelConfig& config,
                          int layer) {
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
void ValidateForwardInputs(const MiniLlamaContext& ctx, const MiniLlamaModel& model,
                           int token) {
  const ModelConfig& config = model.config;
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }

  // 先验证配置，再使用配置值检查 token、位置和各个张量
  ValidateConfig(config);
  if (token < 0 || token >= config.vocab_size) {
    throw std::out_of_range("ForwardToken token id out of range");
  }
  if (ctx.pos < 0 || ctx.pos >= config.max_seq_len) {
    throw std::out_of_range("ForwardToken position out of range");
  }
  if (model.layers.size() != static_cast<size_t>(config.n_layers)) {
    throw std::runtime_error("ForwardToken layer count does not match model config");
  }

  // 模型级张量和上下文缓存必须与配置完全一致
  RequireShape(model.token_embedding, {config.vocab_size, config.dim},
               "ForwardToken token_embedding");
  RequireShape(model.final_norm, {config.dim}, "ForwardToken final_norm");
  model.lm_head.AssertShape({config.vocab_size, config.dim}, "ForwardToken lm_head");
  RequireShape(
      ctx.kv_cache.keys,
      {config.n_layers, config.max_seq_len, config.n_kv_heads, config.head_dim},
      "ForwardToken key cache");
  RequireShape(
      ctx.kv_cache.values,
      {config.n_layers, config.max_seq_len, config.n_kv_heads, config.head_dim},
      "ForwardToken value cache");

  // 每层权重都在计算开始前检查，避免执行到一半才修改部分 KV Cache
  for (int layer = 0; layer < config.n_layers; ++layer) {
    ValidateLayerWeights(model.layers[layer], config, layer);
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

// 按上传时的名称和类型借用快照中的线性权重，不取得显存所有权
const CudaUploadedWeight* FindCudaWeight(const MiniLlamaModel& model,
                                         const std::string& name, QuantType type) {
  if (!model.cuda_weights) return nullptr;
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.linear && weight.name == name && weight.type == type) {
      return &weight;
    }
  }
  return nullptr;
}

int CudaDeviceId(const MiniLlamaModel& model) {
  return model.cuda_weights->device_id;
}

// 命中常驻 F32 权重时执行 CUDA Linear；未命中及量化权重复用 CPU Linear
static Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name,
                            const Tensor& x, const QuantizedTensor& weight) {
  const CudaUploadedWeight* cuda_weight = FindCudaWeight(model, name, weight.type);
  if (cuda_weight != nullptr && weight.type == QuantType::kF32) {
    if (cuda_weight->shape != weight.shape) {
      throw std::runtime_error("ForwardLinear: CUDA weight shape mismatch: " + name);
    }
    auto& stats = *model.cuda_weights;

    // 当前 Norm/Attention/FFN 辅助算子只有 CPU 实现，在边界上传/下载
    // 上传快照中的权重始终借用，不在 Decode 中重新上传
    CudaTensor device_x = CudaTensorFromHost(x, CudaDeviceId(model));
    ++stats.host_to_device_copies;
    stats.host_to_device_bytes += device_x.bytes();
    CudaTensor y = CudaLinearDeviceInput(device_x, cuda_weight->buffer.data(),
                                         cuda_weight->shape, CudaDeviceId(model));
    ++stats.linear_calls;
    Tensor result = y.Download();
    ++stats.device_to_host_copies;
    stats.device_to_host_bytes += y.bytes();
    return result;
  }
  return Linear(x, weight);
}

// Q/K/V 共用归一化输入，分别投影并添加可选偏置；输出保持一维以便按头 reshape
static void ForwardQkvProjection(const MiniLlamaModel& model,
                                 const std::string& layer_prefix, const Tensor& h,
                                 const LayerWeights& lw, Tensor& q_flat,
                                 Tensor& k_flat, Tensor& v_flat) {
  q_flat = AddOptionalBias(ForwardLinear(model, layer_prefix + "wq", h, lw.wq),
                           lw.bq, "forward_layer q");
  k_flat = AddOptionalBias(ForwardLinear(model, layer_prefix + "wk", h, lw.wk),
                           lw.bk, "forward_layer k");
  v_flat = AddOptionalBias(ForwardLinear(model, layer_prefix + "wv", h, lw.wv),
                           lw.bv, "forward_layer v");
}

// 执行现有 CPU RMSNorm，隐藏状态与缩放权重均为 [dim]；保留上层包装签名
static Tensor ForwardRmsNorm(const MiniLlamaModel& model, const std::string& name,
                             const Tensor& x, const Tensor& weight, float eps) {
  (void)model;
  (void)name;
  return RmsNorm(x, weight, eps);
}

// 融合同 shape 的 gate/up 分支，计算 SiLU(gate) * up，仍在 CPU 上执行
static Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                            const Tensor& up) {
  (void)model;
  return SwiGlu(gate, up);
}

// 残差逐元素相加；shape 必须一致，新建输出避免覆盖任一输入分支
static Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& a,
                         const Tensor& b) {
  (void)model;
  if (a.shape != b.shape) {
    throw std::runtime_error("forward_add: shape mismatch " + a.ShapeStringShort() +
                             " vs " + b.ShapeStringShort());
  }
  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.size(); ++i) y.data[i] = a.data[i] + b.data[i];
  return y;
}

// CPU 注意力使用现有 Softmax；当前没有独立的 CUDA Softmax 接口
static Tensor ForwardSoftmax(const Tensor& x) { return Softmax(x); }

// 按模型布局原地旋转 Q/K，pos 是绝对位置，V 不参与 RoPE
static void ForwardRope(const MiniLlamaModel& model, Tensor& q, Tensor& k, int pos,
                        float theta, RopeType rope_type) {
  (void)model;
  Rope(q, k, pos, theta, rope_type);
}

// Embedding lookup: token_id -> x[dim]
static Tensor EmbedToken(const MiniLlamaModel& model, int token_id) {
  const int dim = model.config.dim;
  const size_t offset = static_cast<size_t>(token_id) * static_cast<size_t>(dim);
  Tensor x({dim}, 0.0f);
  for (int i = 0; i < dim; ++i) x.data[i] = model.token_embedding.data[offset + i];
  return x;
}

// GQA: 同一组查询头共享一个 KV 头
static int MapQHeadToKvHead(int q_head, int n_heads, int n_kv_heads) {
  return q_head / (n_heads / n_kv_heads);
}

// q: [n_heads, head_dim], k/v: [n_kv_heads, head_dim]
// 上下文尚未持有 CUDA KV Cache，缓存和注意力目前在 CPU 上执行
static Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& q,
                               const Tensor& k, const Tensor& v, int pos, int layer,
                               KvCache& kv_cache, int n_heads, int n_kv_heads,
                               int head_dim) {
  // 写入当前位置后才读 [0, pos]，既包含当前 token，也保持因果约束
  kv_cache.Write(layer, pos, k, v);
  (void)model;
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
    Tensor probs = ForwardSoftmax(scores);
    for (int d = 0; d < head_dim; ++d) {
      float val = 0.0f;
      for (int t = 0; t <= pos; ++t) {
        const float* v_ptr = kv_cache.valuePtr(layer, t, kv_head);
        val += probs.data[t] * v_ptr[d];
      }
      attn_out.data[h * head_dim + d] = val;
    }
  };

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
  Tensor h = ForwardRmsNorm(model, layer_prefix + "attention_norm", x,
                            lw.attention_norm, c.rms_norm_eps);
  Tensor q_flat, k_flat, v_flat;
  const size_t linear_calls_before_qkv = ModelCudaLinearCalls(model);
  ForwardQkvProjection(model, layer_prefix, h, lw, q_flat, k_flat, v_flat);
  Tensor q = q_flat.ReshapeChecked({n_heads, head_dim}, "forward_layer q");
  Tensor k = k_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer k");
  Tensor v = v_flat.ReshapeChecked({n_kv_heads, head_dim}, "forward_layer v");
  ForwardRope(model, q, k, pos, c.rope_theta, c.rope_type);
  Tensor attn_out = AttentionForward(model, q, k, v, pos, layer, ctx.kv_cache,
                                     n_heads, n_kv_heads, head_dim);

  // Q/K/V 实际执行过 CUDA 投影时，后续 CPU Attention 才计作设备链路回退
  if (model.cuda_weights && ModelCudaLinearCalls(model) > linear_calls_before_qkv) {
    ++model.cuda_weights->attention_cpu_fallbacks;
  }
  // 拼接查询头输出，执行 O 投影并加回本层输入，形成第一次残差
  Tensor attn_out_flat =
      attn_out.ReshapeChecked({1, n_heads * head_dim}, "forward_layer attn_out");
  Tensor attn_proj = ForwardLinear(model, layer_prefix + "wo", attn_out_flat, lw.wo);
  RequireShape(attn_proj, {1, dim}, "forward_layer attn_proj");
  Tensor attn_proj_1d =
      attn_proj.ReshapeChecked({dim}, "forward_layer attn_proj 1d");
  Tensor x_attn = ForwardAdd(model, x, attn_proj_1d);

  // FFN：第二次 Pre-Norm -> Gate/Up -> SwiGLU -> Down -> 第二次残差
  Tensor h2 = ForwardRmsNorm(model, layer_prefix + "ffn_norm", x_attn, lw.ffn_norm,
                             c.rms_norm_eps);
  Tensor ff = FfnForward(model, layer_prefix, h2, lw);
  RequireShape(ff, {dim}, "forward_layer ffn");
  return ForwardAdd(model, x_attn, ff);
}

// 最终 RMSNorm + LM Head，返回 [vocab_size] logits
static Tensor ComputeLogits(const Tensor& x, const MiniLlamaModel& model,
                            const ModelConfig& c) {
  Tensor normed =
      ForwardRmsNorm(model, "final_norm", x, model.final_norm, c.rms_norm_eps);
  Tensor logits = ForwardLinear(model, "lm_head", normed, model.lm_head);
  return logits.ReshapeChecked({c.vocab_size}, "compute_logits");
}

}  // namespace

// 处理 ctx.pos 指定的单个 token，返回 [vocab_size] logits 并更新各层 K/V
// 位置推进和 token_history 维护交由 ForwardBatch 或直接调用方负责
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
  ValidateForwardInputs(ctx, model, token);
  // 不改变已有调用接口；各 Linear 按名称和类型查找常驻 F32 权重
  const ModelConfig& c = model.config;
  Tensor x = EmbedToken(model, token);
  for (int layer = 0; layer < c.n_layers; ++layer) {
    x = ForwardLayer(ctx, model, x, layer, model.layers[layer], c);
  }
  return ComputeLogits(x, model, c);
}

}  // namespace mini_llama
