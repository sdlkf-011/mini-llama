#include "mini_llama/forward.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/quant.h"
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

// 为一维投影结果添加可选偏置；空张量表示该模型没有对应偏置
Tensor AddOptionalBias(const Tensor& input, const Tensor& bias, const char* caller) {
  if (bias.data.empty()) {
    return input;
  }
  if (input.num_dims() != 1 || bias.num_dims() != 1 ||
      input.shape[0] != bias.shape[0]) {
    throw std::runtime_error(std::string(caller) +
                             ": bias shape mismatch x=" + input.ShapeStringShort() +
                             " bias=" + bias.ShapeStringShort());
  }

  // 复制投影结果后逐元素累加，避免修改调用方持有的输入张量
  Tensor output = input;
  for (size_t index = 0; index < output.size(); ++index) {
    output.data[index] += bias.data[index];
  }
  return output;
}

// 根据权重标签分发 F32 或量化 Linear；量化分支不会展开完整权重矩阵
Tensor ForwardLinear(const MiniLlamaModel& model, const std::string& name,
                     const Tensor& input, const QuantizedTensor& weight) {
  if (!model.loaded) {
    throw std::runtime_error("ForwardLinear called with an unloaded model: " + name);
  }

  switch (weight.type) {
    case QuantType::kF32:
      return Linear(input, ToTensor(weight));
    case QuantType::kQ80:
      return LinearQ80(input, weight.q8_0_data, weight.shape);
    case QuantType::kQ40:
      return LinearQ40(input, weight.q4_0_data, weight.shape);
    case QuantType::kQ41:
      return LinearQ41(input, weight.q4_1_data, weight.shape);
  }
  throw std::runtime_error("ForwardLinear: unsupported weight type: " + name);
}

// 对两个同形状张量执行逐元素加法，用于 Attention 和 FFN 的残差连接
Tensor ForwardAdd(const MiniLlamaModel& model, const Tensor& left,
                  const Tensor& right) {
  (void)model;
  if (left.shape != right.shape) {
    throw std::runtime_error("ForwardAdd: shape mismatch " +
                             left.ShapeStringShort() + " vs " +
                             right.ShapeStringShort());
  }

  // 新建输出张量，确保残差计算不会原地覆盖任一输入分支
  Tensor output(left.shape, 0.0f);
  for (size_t index = 0; index < output.size(); ++index) {
    output.data[index] = left.data[index] + right.data[index];
  }
  return output;
}

// 一次性完成 Q/K/V 投影和可选偏置，为后续头维度重排提供扁平结果
void ForwardQkvProjection(const MiniLlamaModel& model,
                          const std::string& layer_prefix, const Tensor& normalized,
                          const LayerWeights& weights, Tensor& query, Tensor& key,
                          Tensor& value) {
  // 三个投影共享同一个归一化输入，但使用各自的权重和可选偏置
  query = AddOptionalBias(
      ForwardLinear(model, layer_prefix + "wq", normalized, weights.wq), weights.bq,
      "ForwardLayer query");
  key = AddOptionalBias(
      ForwardLinear(model, layer_prefix + "wk", normalized, weights.wk), weights.bk,
      "ForwardLayer key");
  value = AddOptionalBias(
      ForwardLinear(model, layer_prefix + "wv", normalized, weights.wv), weights.bv,
      "ForwardLayer value");
}

// 根据最终实现的后端入口执行 RMSNorm，当前仓库使用 CPU 算子
Tensor ForwardRmsNorm(const MiniLlamaModel& model, const Tensor& input,
                      const Tensor& weight, float eps) {
  (void)model;
  return RmsNorm(input, weight, eps);
}

// 根据最终实现的后端入口执行 SwiGLU，当前仓库使用 CPU 算子
Tensor ForwardSwiGlu(const MiniLlamaModel& model, const Tensor& gate,
                     const Tensor& up) {
  (void)model;
  return SwiGlu(gate, up);
}

// 根据最终实现的后端入口执行 Softmax，当前仓库使用 CPU 算子
Tensor ForwardSoftmax(const MiniLlamaModel& model, const Tensor& input) {
  (void)model;
  return Softmax(input);
}

// 根据最终实现的后端入口为 Q/K 应用 RoPE，当前仓库使用 CPU 算子
void ForwardRope(const MiniLlamaModel& model, Tensor& query, Tensor& key, int pos,
                 float theta, RopeType rope_type) {
  (void)model;
  Rope(query, key, pos, theta, rope_type);
}

// 从词嵌入表复制 token 对应的一行，形成当前 token 的初始隐藏状态
Tensor EmbedToken(const MiniLlamaModel& model, int token) {
  const int dim = model.config.dim;
  const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(dim);
  Tensor embedding({dim}, 0.0f);

  // 嵌入表按 [vocab_size, dim] 行主序保存，目标行在内存中连续
  for (int index = 0; index < dim; ++index) {
    embedding.data[index] = model.token_embedding.data[offset + index];
  }
  return embedding;
}

// 将查询头映射到共享的 KV 头，实现 grouped-query attention
int MapQHeadToKvHead(int query_head, int n_heads, int n_kv_heads) {
  // 配置校验已保证 n_heads 能够被 n_kv_heads 整除。
  return query_head / (n_heads / n_kv_heads);
}

// 计算单层、单 token 的因果自注意力，并读写该层的 KV Cache
Tensor AttentionForward(const MiniLlamaModel& model, const Tensor& query,
                        const Tensor& key, const Tensor& value, int pos, int layer,
                        KvCache& kv_cache, int n_heads, int n_kv_heads,
                        int head_dim) {
  // 先写入当前位置，使当前 token 本身也参与 [0, pos] 的注意力计算
  kv_cache.Write(layer, pos, key, value);
  Tensor output({n_heads, head_dim}, 0.0f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // 缓存写入已完成；各 head 只读缓存，并独占输出中对应的一行
  ParallelFor(n_heads, [&](int begin, int end) {
    for (int query_head = begin; query_head < end; ++query_head) {
      const int kv_head = MapQHeadToKvHead(query_head, n_heads, n_kv_heads);
      Tensor scores({pos + 1}, 0.0f);

      // 只遍历不晚于当前位置的缓存，因此自然满足自回归因果掩码
      for (int token_pos = 0; token_pos <= pos; ++token_pos) {
        const float* cached_key = kv_cache.KeyPtr(layer, token_pos, kv_head);
        float dot = 0.0f;
        for (int index = 0; index < head_dim; ++index) {
          dot += query.data[query_head * head_dim + index] * cached_key[index];
        }
        scores.data[token_pos] = dot * scale;
      }

      // Softmax 得到历史位置权重，再对相同 KV 头中的 V 做加权求和
      const Tensor probabilities = ForwardSoftmax(model, scores);
      for (int index = 0; index < head_dim; ++index) {
        float weighted_value = 0.0f;
        for (int token_pos = 0; token_pos <= pos; ++token_pos) {
          const float* cached_value = kv_cache.valuePtr(layer, token_pos, kv_head);
          weighted_value += probabilities.data[token_pos] * cached_value[index];
        }
        output.data[query_head * head_dim + index] = weighted_value;
      }
    }
  });
  return output;
}

// 执行 SwiGLU 前馈网络：gate/up 上投影、激活融合和 down 下投影
Tensor FfnForward(const MiniLlamaModel& model, const std::string& layer_prefix,
                  const Tensor& normalized, const LayerWeights& weights) {
  // FFN 第 2 步：Gate 与 Up 使用同一归一化输入分别执行线性投影
  const Tensor gate =
      ForwardLinear(model, layer_prefix + "w_gate", normalized, weights.w_gate);
  const Tensor up =
      ForwardLinear(model, layer_prefix + "w_up", normalized, weights.w_up);

  // FFN 第 3 步：SwiGLU 计算 SiLU(gate) * up，融合两条中间分支
  const Tensor activated = ForwardSwiGlu(model, gate, up);

  // FFN 第 4 步：Down 投影把 hidden_dim 恢复为模型维度 dim
  return ForwardLinear(model, layer_prefix + "w_down", activated, weights.w_down);
}

// 执行一个 Pre-Norm Transformer 层，并依次完成注意力和 FFN 残差更新
Tensor ForwardLayer(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const Tensor& input, int layer, const LayerWeights& weights,
                    const ModelConfig& config) {
  const std::string layer_prefix = "layers." + std::to_string(layer) + ".";

  // Attention 第 1 步：Pre-Norm，在进入注意力网络前执行 RMSNorm
  Tensor normalized =
      ForwardRmsNorm(model, input, weights.attention_norm, config.rms_norm_eps);

  // Attention 第 2 步：基于同一归一化输入执行 Q/K/V 线性投影
  Tensor query_flat;
  Tensor key_flat;
  Tensor value_flat;
  ForwardQkvProjection(model, layer_prefix, normalized, weights, query_flat,
                       key_flat, value_flat);

  // Attention 第 3 步：把扁平投影结果划分为查询头和 KV 头
  Tensor query = query_flat.ReshapeChecked({config.n_heads, config.head_dim},
                                           "ForwardLayer query");
  Tensor key = key_flat.ReshapeChecked({config.n_kv_heads, config.head_dim},
                                       "ForwardLayer key");
  Tensor value = value_flat.ReshapeChecked({config.n_kv_heads, config.head_dim},
                                           "ForwardLayer value");

  // Attention 第 4 步：向 Q/K 注入当前位置的旋转位置编码
  ForwardRope(model, query, key, ctx.pos, config.rope_theta, config.rope_type);

  // Attention 第 5 步：写入当前 K/V，并对 [0, pos] 执行因果注意力
  Tensor attention =
      AttentionForward(model, query, key, value, ctx.pos, layer, ctx.kv_cache,
                       config.n_heads, config.n_kv_heads, config.head_dim);

  // Attention 第 6 步：拼接全部查询头，再执行输出投影
  Tensor attention_flat = attention.ReshapeChecked(
      {1, config.n_heads * config.head_dim}, "ForwardLayer attention");
  Tensor attention_projection =
      ForwardLinear(model, layer_prefix + "wo", attention_flat, weights.wo);
  RequireShape(attention_projection, {1, config.dim},
               "ForwardLayer attention projection");
  Tensor attention_projection_1d = attention_projection.ReshapeChecked(
      {config.dim}, "ForwardLayer attention projection 1d");

  // Attention 第 7 步：将注意力输出加回本层输入，形成第一条残差
  Tensor attention_residual = ForwardAdd(model, input, attention_projection_1d);

  // FFN 第 1 步：对注意力残差执行第二次 Pre-Norm
  Tensor ffn_input = ForwardRmsNorm(model, attention_residual, weights.ffn_norm,
                                    config.rms_norm_eps);

  // FFN 第 2 至 4 步：Gate/Up 投影、SwiGLU 激活和 Down 投影
  Tensor ffn_output = FfnForward(model, layer_prefix, ffn_input, weights);
  RequireShape(ffn_output, {config.dim}, "ForwardLayer FFN");

  // FFN 第 5 步：将 FFN 输出加回注意力残差，得到本层最终输出
  return ForwardAdd(model, attention_residual, ffn_output);
}

// 对最终隐藏状态执行 RMSNorm 和词表投影，返回未归一化 logits
Tensor ComputeLogits(const Tensor& input, const MiniLlamaModel& model,
                     const ModelConfig& config) {
  // 最终归一化不改变隐藏状态维度，lm_head 将其映射到词表维度。
  Tensor normalized =
      ForwardRmsNorm(model, input, model.final_norm, config.rms_norm_eps);
  Tensor logits = ForwardLinear(model, "lm_head", normalized, model.lm_head);
  return logits.ReshapeChecked({config.vocab_size}, "ComputeLogits");
}

}  // namespace

// 执行单 token 前向传播；函数更新 KV Cache，但由调用方负责推进位置
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token) {
  ValidateForwardInputs(ctx, model, token);
  const ModelConfig& config = model.config;

  // 第 1 步：从词嵌入表取得当前 token 的初始隐藏状态
  Tensor hidden = EmbedToken(model, token);

  // 第 2 步：按层执行图示的 Attention 网络和 FFN 网络
  for (int layer = 0; layer < config.n_layers; ++layer) {
    hidden = ForwardLayer(ctx, model, hidden, layer, model.layers[layer], config);
  }

  // 第 3 步：最后一层隐藏状态经归一化和 lm_head 得到词表 logits
  return ComputeLogits(hidden, model, config);
}

}  // namespace mini_llama
