#include "mini_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/ops.h"

namespace mini_llama {
namespace {

// 统一执行张量形状检查，使错误信息能够指出具体的前向计算阶段。
void RequireShape(const Tensor& tensor, const std::vector<int>& expected,
                  const char* caller) {
  tensor.AssertShape(expected, caller);
}

// 检查各配置项之间的约束，避免在注意力头映射或 RoPE 中发生除零和越界。
void ValidateConfig(const ModelConfig& config) {
  if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
      config.n_layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
      config.head_dim <= 0 || config.max_seq_len <= 0) {
    throw std::runtime_error("ForwardToken: model config values must be positive");
  }
  if (config.dim != config.n_heads * config.head_dim) {
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

// 检查单层权重的形状。Q 可以拥有更多头，而 K/V 使用 GQA 的 KV 头数。
void ValidateLayerWeights(const LayerWeights& weights, const ModelConfig& config,
                          int layer) {
  const std::string prefix = "ForwardToken layer " + std::to_string(layer);
  const int query_dim = config.n_heads * config.head_dim;
  const int kv_dim = config.n_kv_heads * config.head_dim;

  RequireShape(weights.attention_norm, {config.dim},
               (prefix + " attention_norm").c_str());
  RequireShape(weights.wq, {query_dim, config.dim}, (prefix + " wq").c_str());
  RequireShape(weights.wk, {kv_dim, config.dim}, (prefix + " wk").c_str());
  RequireShape(weights.wv, {kv_dim, config.dim}, (prefix + " wv").c_str());
  RequireShape(weights.wo, {config.dim, query_dim}, (prefix + " wo").c_str());
  if (!weights.bq.data.empty()) {
    RequireShape(weights.bq, {query_dim}, (prefix + " bq").c_str());
  }
  if (!weights.bk.data.empty()) {
    RequireShape(weights.bk, {kv_dim}, (prefix + " bk").c_str());
  }
  if (!weights.bv.data.empty()) {
    RequireShape(weights.bv, {kv_dim}, (prefix + " bv").c_str());
  }
  RequireShape(weights.ffn_norm, {config.dim}, (prefix + " ffn_norm").c_str());
  RequireShape(weights.w_gate, {config.hidden_dim, config.dim},
               (prefix + " w_gate").c_str());
  RequireShape(weights.w_up, {config.hidden_dim, config.dim},
               (prefix + " w_up").c_str());
  RequireShape(weights.w_down, {config.dim, config.hidden_dim},
               (prefix + " w_down").c_str());
}

// 在真正计算前集中验证模型、token、位置、缓存及所有权重。
// 这样后续算子可以直接按已知形状访问连续内存。
void ValidateForwardInputs(const MiniLlamaContext& ctx, const MiniLlamaModel& model,
                           int token) {
  const ModelConfig& config = model.config;
  if (!model.loaded) {
    throw std::runtime_error("ForwardToken called with an unloaded model");
  }
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

  RequireShape(model.token_embedding, {config.vocab_size, config.dim},
               "ForwardToken token_embedding");
  RequireShape(model.final_norm, {config.dim}, "ForwardToken final_norm");
  RequireShape(model.lm_head, {config.vocab_size, config.dim},
               "ForwardToken lm_head");
  RequireShape(
      ctx.kv_cache.keys,
      {config.n_layers, config.max_seq_len, config.n_kv_heads, config.head_dim},
      "ForwardToken key cache");
  RequireShape(
      ctx.kv_cache.values,
      {config.n_layers, config.max_seq_len, config.n_kv_heads, config.head_dim},
      "ForwardToken value cache");
  for (int layer = 0; layer < config.n_layers; ++layer) {
    ValidateLayerWeights(model.layers[layer], config, layer);
  }
}

// 部分模型的 Q/K/V 投影包含偏置；空 Tensor 表示该偏置不存在。
Tensor AddOptionalBias(const Tensor& input, const Tensor& bias, const char* caller) {
  if (bias.data.empty()) {
    return input;
  }
  if (input.num_dims() != 1 || bias.num_dims() != 1 ||
      input.shape[0] != bias.shape[0]) {
    throw std::runtime_error(std::string(caller) + ": bias shape mismatch");
  }
  Tensor output = input;
  for (size_t i = 0; i < output.size(); ++i) {
    output.data[i] += bias.data[i];
  }
  return output;
}

// 当前阶段统一调用 F32 Linear。保留模型和权重名称参数，后续可在此处
// 分发到 QuantizedTensor 或已上传到 GPU 的同名权重。
Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name,
                     const Tensor& input, const Tensor& weight) {
  (void)model;
  (void)name;
  return Linear(input, weight);
}

// RMSNorm 的前向包装层，后续 CUDA 阶段可在这里切换设备实现。
Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& input,
                      const Tensor& weight, float eps) {
  (void)model;
  return RmsNorm(input, weight, eps);
}

// SwiGLU 的前向包装层：SiLU(gate) 与 up 逐元素相乘。
Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                     const Tensor& up) {
  (void)model;
  return SwiGlu(gate, up);
}

// 残差连接要求两个分支形状完全相同，当前阶段在 CPU 上逐元素相加。
Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& left,
                  const Tensor& right) {
  (void)model;
  if (left.shape != right.shape) {
    throw std::runtime_error("forward_add: shape mismatch " +
                             left.ShapeStringShort() + " vs " +
                             right.ShapeStringShort());
  }
  Tensor output(left.shape, 0.0f);
  for (size_t i = 0; i < output.size(); ++i) {
    output.data[i] = left.data[i] + right.data[i];
  }
  return output;
}

// Softmax 包装层用于归一化单个注意力头在历史位置上的分数。
Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& input) {
  (void)model;
  return Softmax(input);
}

// 对当前 token 的 Q/K 应用旋转位置编码，V 不参与旋转。
void ForwardRope(const MiniLlamaModel& model, Tensor& query, Tensor& key, int pos,
                 float theta, RopeType rope_type) {
  (void)model;
  Rope(query, key, pos, theta, rope_type);
}

// 从 [vocab_size, dim] 的嵌入表复制指定 token 对应的一行。
Tensor EmbedToken(const MiniLlamaModel& model, int token) {
  const int dim = model.config.dim;
  Tensor embedding({dim}, 0.0f);
  const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(dim);
  for (int i = 0; i < dim; ++i) {
    embedding.data[i] = model.token_embedding.data[offset + i];
  }
  return embedding;
}

// GQA 中一组查询头共享同一个 KV 头。
// 例如 8 个查询头和 2 个 KV 头时，每连续 4 个查询头映射到同一 KV 头。
int MapQHeadToKvHead(int query_head, int n_heads, int n_kv_heads) {
  return query_head / (n_heads / n_kv_heads);
}

// 计算单层、单 token 的因果自注意力。
// query: [n_heads, head_dim]，key/value: [n_kv_heads, head_dim]。
Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& query,
                        const Tensor& key, const Tensor& value, int pos, int layer,
                        KvCache& kv_cache, int n_heads, int n_kv_heads,
                        int head_dim) {
  // 先缓存当前位置的 K/V，使当前位置也参与注意力计算。
  kv_cache.Write(layer, pos, key, value);
  Tensor output({n_heads, head_dim}, 0.0f);
  // 缩放点积可避免 head_dim 增大时 Softmax 过早饱和。
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int query_head = 0; query_head < n_heads; ++query_head) {
    const int kv_head = MapQHeadToKvHead(query_head, n_heads, n_kv_heads);
    Tensor scores({pos + 1}, 0.0f);

    // 只访问 [0, pos]，因此天然满足自回归推理的因果掩码要求。
    for (int token_pos = 0; token_pos <= pos; ++token_pos) {
      const float* cached_key = kv_cache.KeyPtr(layer, token_pos, kv_head);
      float dot = 0.0f;
      for (int dim = 0; dim < head_dim; ++dim) {
        dot += query.data[query_head * head_dim + dim] * cached_key[dim];
      }
      scores.data[token_pos] = dot * scale;
    }

    // 将历史位置分数归一化，再对缓存中的 V 做加权求和。
    const Tensor probabilities = ForwardSoftmax(model, scores);
    for (int dim = 0; dim < head_dim; ++dim) {
      float weighted_value = 0.0f;
      for (int token_pos = 0; token_pos <= pos; ++token_pos) {
        const float* cached_value = kv_cache.valuePtr(layer, token_pos, kv_head);
        weighted_value += probabilities.data[token_pos] * cached_value[dim];
      }
      output.data[query_head * head_dim + dim] = weighted_value;
    }
  }
  return output;
}

// SwiGLU 前馈网络：两个上投影产生 gate/up，激活后再投影回模型维度。
Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix,
                  const Tensor& input, const LayerWeights& weights) {
  Tensor gate = ForwardLinear(model, layer_prefix + "w_gate", input, weights.w_gate);
  Tensor up = ForwardLinear(model, layer_prefix + "w_up", input, weights.w_up);
  return ForwardLinear(model, layer_prefix + "w_down",
                       ForwardSwiGlu(model, gate, up), weights.w_down);
}

// 执行一个 Transformer 层，依次完成注意力子层和 FFN 子层。
// 两个子层都采用 Pre-Norm，并在子层输出后执行残差连接。
Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const Tensor& input, int layer, const LayerWeights& weights,
                    const ModelConfig& config) {
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  // ---- 注意力子层：RMSNorm -> QKV 投影 -> RoPE -> Attention ----
  Tensor normalized =
      ForwardRmsNorm(model, input, weights.attention_norm, config.rms_norm_eps);
  Tensor query_flat = AddOptionalBias(
      ForwardLinear(model, layer_prefix + "wq", normalized, weights.wq), weights.bq,
      "ForwardLayer query");
  Tensor key_flat = AddOptionalBias(
      ForwardLinear(model, layer_prefix + "wk", normalized, weights.wk), weights.bk,
      "ForwardLayer key");
  Tensor value_flat = AddOptionalBias(
      ForwardLinear(model, layer_prefix + "wv", normalized, weights.wv), weights.bv,
      "ForwardLayer value");
  Tensor query = query_flat.ReshapeChecked({config.n_heads, config.head_dim},
                                           "ForwardLayer query");
  Tensor key = key_flat.ReshapeChecked({config.n_kv_heads, config.head_dim},
                                       "ForwardLayer key");
  Tensor value = value_flat.ReshapeChecked({config.n_kv_heads, config.head_dim},
                                           "ForwardLayer value");
  ForwardRope(model, query, key, ctx.pos, config.rope_theta, config.rope_type);

  Tensor attention =
      AttentionForward(model, query, key, value, ctx.pos, layer, ctx.kv_cache,
                       config.n_heads, config.n_kv_heads, config.head_dim);
  // wo 接收二维输入，因此先合并所有查询头，再恢复为一维残差分支。
  Tensor attention_flat = attention.ReshapeChecked(
      {1, config.n_heads * config.head_dim}, "ForwardLayer attention");
  Tensor attention_projection =
      ForwardLinear(model, layer_prefix + "wo", attention_flat, weights.wo);
  RequireShape(attention_projection, {1, config.dim},
               "ForwardLayer attention projection");
  Tensor attention_projection_1d = attention_projection.ReshapeChecked(
      {config.dim}, "ForwardLayer attention projection 1d");
  Tensor attention_residual = ForwardAdd(model, input, attention_projection_1d);

  // ---- FFN 子层：RMSNorm -> SwiGLU FFN -> 残差连接 ----
  Tensor ffn_input = ForwardRmsNorm(model, attention_residual, weights.ffn_norm,
                                    config.rms_norm_eps);
  Tensor ffn_output = FfnForward(model, layer_prefix, ffn_input, weights);
  RequireShape(ffn_output, {config.dim}, "ForwardLayer FFN");
  return ForwardAdd(model, attention_residual, ffn_output);
}

// 最终隐藏状态经过 RMSNorm 和词表投影，得到每个 token 的未归一化 logits。
Tensor ComputeLogits(const Tensor& input, const MiniLlamaModel& model) {
  Tensor normalized =
      ForwardRmsNorm(model, input, model.final_norm, model.config.rms_norm_eps);
  Tensor logits = ForwardLinear(model, "lm_head", normalized, model.lm_head);
  return logits.ReshapeChecked({model.config.vocab_size}, "ComputeLogits");
}

}  // namespace

// 单 token 解码入口。当前位置由 ctx.pos 指定，函数会更新每层的 KV Cache，
// 但不推进位置；批处理或生成循环负责管理 ctx.pos。
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
  ValidateForwardInputs(ctx, model, token);
  Tensor hidden = EmbedToken(model, token);
  for (int layer = 0; layer < model.config.n_layers; ++layer) {
    hidden =
        ForwardLayer(ctx, model, hidden, layer, model.layers[layer], model.config);
  }
  return ComputeLogits(hidden, model);
}

}  // namespace mini_llama
