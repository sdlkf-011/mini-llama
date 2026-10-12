// 测试目的：验证 CPU 前向数值、批量与逐 token 等价性、KV 状态及后端分发。
#include "mini_llama/forward.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/quant.h"
#include "mini_llama/tensor.h"

namespace {

using mini_llama::ForwardBatch;
using mini_llama::ForwardToken;
using mini_llama::MiniBatch;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::Tensor;

// 构造可手算的小模型：恒等 Attention 投影、零 FFN，便于核对 logits 与缓存。
MiniLlamaModel MakeTestModel() {
  MiniLlamaModel model;
  model.config.vocab_size = 3;
  model.config.dim = 2;
  model.config.hidden_dim = 2;
  model.config.n_layers = 1;
  model.config.n_heads = 1;
  model.config.n_kv_heads = 1;
  model.config.head_dim = 2;
  model.config.max_seq_len = 4;
  model.config.rms_norm_eps = 0.0f;
  model.loaded = true;

  model.token_embedding = Tensor({3, 2}, 0.0f);
  model.token_embedding.data = {3.0f, 4.0f, 4.0f, 3.0f, 1.0f, -1.0f};

  model.layers.resize(1);
  auto& layer = model.layers[0];
  layer.attention_norm = Tensor({2}, 1.0f);
  // 测试模型仍使用精确 F32 数值，但通过 kF32 分支匹配生产模型的权重类型。
  layer.wq = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));
  layer.wq.f32_data = {1.0f, 0.0f, 0.0f, 1.0f};
  layer.wk = layer.wq;
  layer.wv = layer.wq;
  layer.wo = layer.wq;
  layer.ffn_norm = Tensor({2}, 1.0f);
  layer.w_gate = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));
  layer.w_up = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));
  layer.w_down = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));

  model.final_norm = Tensor({2}, 1.0f);
  model.lm_head = mini_llama::ToQuantizedTensor(Tensor({3, 2}, 0.0f));
  model.lm_head.f32_data = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  return model;
}

// 将测试中的 F32 线性权重按生产量化规则转换为 Q8_0，保持原始逻辑 shape。
mini_llama::QuantizedTensor ConvertToQ80(const mini_llama::QuantizedTensor& weight) {
  const Tensor dense = mini_llama::ToTensor(weight);
  mini_llama::QuantizedTensor quantized;
  quantized.type = mini_llama::QuantType::kQ80;
  quantized.shape = dense.shape;
  quantized.q8_0_data = mini_llama::QuantizeToQ80(dense);
  return quantized;
}

// 先核对 shape 与元素数，再逐元素使用绝对容差比较浮点结果。
void ExpectTensorNear(const Tensor& actual, const Tensor& expected,
                      float tolerance = 1e-5f) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.data.size(), expected.data.size());
  for (size_t index = 0; index < actual.data.size(); ++index) {
    EXPECT_NEAR(actual.data[index], expected.data[index], tolerance);
  }
}

// 测试单 token 前向链路依次完成 Embedding、Attention、FFN 和 LM Head，
// 输出可手算的词表 logits，并把当前位置的 K/V 写入缓存。
TEST(ForwardTest, ForwardTokenProducesLogitsAndWritesKvCache) {
  MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext context(&model);

  Tensor logits = ForwardToken(context, model, 0);

  const float inverse_rms = 1.0f / std::sqrt(12.5f);
  ASSERT_EQ(logits.shape, std::vector<int>({3}));
  EXPECT_NEAR(logits.data[0], 3.0f * inverse_rms, 1e-5f);
  EXPECT_NEAR(logits.data[1], 4.0f * inverse_rms, 1e-5f);
  EXPECT_NEAR(logits.data[2], 7.0f * inverse_rms, 1e-5f);
  EXPECT_NEAR(context.kv_cache.KeyPtr(0, 0, 0)[0], 3.0f * inverse_rms, 1e-5f);
  EXPECT_NEAR(context.kv_cache.valuePtr(0, 0, 0)[1], 4.0f * inverse_rms, 1e-5f);
}

// 量化模型必须走块级 Linear 完成整条前向，不能回退到 ToTensor 展开权重。
TEST(ForwardTest, ForwardTokenSupportsQ80Weights) {
  MiniLlamaModel dense_model = MakeTestModel();
  MiniLlamaModel quantized_model = dense_model;
  auto& layer = quantized_model.layers[0];
  layer.wq = ConvertToQ80(layer.wq);
  layer.wk = ConvertToQ80(layer.wk);
  layer.wv = ConvertToQ80(layer.wv);
  layer.wo = ConvertToQ80(layer.wo);
  layer.w_gate = ConvertToQ80(layer.w_gate);
  layer.w_up = ConvertToQ80(layer.w_up);
  layer.w_down = ConvertToQ80(layer.w_down);
  quantized_model.lm_head = ConvertToQ80(quantized_model.lm_head);

  MiniLlamaContext dense_context(&dense_model);
  MiniLlamaContext quantized_context(&quantized_model);
  const Tensor dense_logits = ForwardToken(dense_context, dense_model, 0);
  const Tensor quantized_logits =
      ForwardToken(quantized_context, quantized_model, 0);

  ExpectTensorNear(quantized_logits, dense_logits, 2e-3f);
}

// 测试 ForwardToken 只使用调用方指定的位置，不推进 ctx.pos，也不修改由
// ForwardBatch 或生成循环负责维护的 token_history。
TEST(ForwardTest, ForwardTokenLeavesSequenceStateToCaller) {
  MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext context(&model);
  context.pos = 1;
  context.token_history = {0};

  ForwardToken(context, model, 1);

  EXPECT_EQ(context.pos, 1);
  EXPECT_EQ(context.token_history, std::vector<int>({0}));
  EXPECT_NE(context.kv_cache.KeyPtr(0, 1, 0)[0], 0.0f);
  EXPECT_NE(context.kv_cache.valuePtr(0, 1, 0)[0], 0.0f);
}

// 测试 ForwardBatch 为 Prefill 构造连续位置并顺序调用 ForwardToken；其最终
// logits 与手工逐 token 执行一致，同时正确维护当前位置和 token 历史。
TEST(ForwardTest, ForwardBatchMatchesSequentialExecution) {
  MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext batch_context(&model);
  MiniLlamaContext sequential_context(&model);
  MiniBatch batch = MiniBatch::FromTokens({0, 1}, 0);

  ASSERT_EQ(batch.positions, std::vector<int>({0, 1}));
  Tensor batch_logits = ForwardBatch(batch_context, model, batch);

  sequential_context.pos = 0;
  ForwardToken(sequential_context, model, 0);
  sequential_context.pos = 1;
  Tensor sequential_logits = ForwardToken(sequential_context, model, 1);

  ExpectTensorNear(batch_logits, sequential_logits);
  EXPECT_EQ(batch_context.pos, 1);
  EXPECT_EQ(batch_context.token_history, std::vector<int>({0, 1}));

  MiniBatch decode = MiniBatch::Single(2, 2);
  ForwardBatch(batch_context, model, decode);
  EXPECT_EQ(batch_context.pos, 2);
  EXPECT_EQ(batch_context.token_history, std::vector<int>({0, 1, 2}));
}

// 测试 ForwardToken 拒绝未加载模型、越界 token、越界位置以及不符合配置的
// 权重形状，避免错误维度进入 Transformer 数值计算。
TEST(ForwardTest, RejectsInvalidModelTokenPositionAndWeights) {
  MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext context(&model);

  model.loaded = false;
  EXPECT_THROW(ForwardToken(context, model, 0), std::runtime_error);
  model.loaded = true;

  EXPECT_THROW(ForwardToken(context, model, -1), std::out_of_range);
  EXPECT_THROW(ForwardToken(context, model, model.config.vocab_size),
               std::out_of_range);

  context.pos = model.config.max_seq_len;
  EXPECT_THROW(ForwardToken(context, model, 0), std::out_of_range);
  context.pos = 0;

  // 即使存储类型改为 QuantizedTensor，错误逻辑 shape 仍必须在前向前被拒绝。
  model.layers[0].wq = mini_llama::ToQuantizedTensor(Tensor({1, 2}, 0.0f));
  EXPECT_THROW(ForwardToken(context, model, 0), std::runtime_error);
}

// 测试 ForwardBatch 拒绝空批次及 token/position 数量不一致的批次，确保
// 批处理入口不会留下只执行了一部分的上下文状态。
TEST(ForwardTest, RejectsInvalidBatch) {
  MiniLlamaModel model = MakeTestModel();
  MiniLlamaContext context(&model);

  EXPECT_THROW(ForwardBatch(context, model, MiniBatch()), std::runtime_error);

  MiniBatch mismatched;
  mismatched.tokens = {0, 1};
  mismatched.positions = {0};
  EXPECT_THROW(ForwardBatch(context, model, mismatched), std::runtime_error);
  EXPECT_EQ(context.pos, 0);
  EXPECT_TRUE(context.token_history.empty());
}

// 形状相同的不同模型不能共用上下文，拒绝时保留已有状态。
TEST(ForwardTest, ContextRejectsDifferentModelBeforeChangingState) {
  auto model = MakeTestModel();
  auto different_model = model;
  different_model.token_embedding.data[0] += 1.0f;
  MiniLlamaContext context(&model);
  ForwardBatch(context, model, MiniBatch::Single(0, 0));
  const auto keys = context.kv_cache.keys.data;
  const auto values = context.kv_cache.values.data;
  EXPECT_THROW(ForwardBatch(context, different_model, MiniBatch::Single(1, 1)),
               std::runtime_error);
  EXPECT_THROW(ForwardToken(context, different_model, 1), std::runtime_error);
  EXPECT_EQ(context.pos, 0);
  EXPECT_EQ(context.token_history, std::vector<int>({0}));
  EXPECT_EQ(context.kv_cache.keys.data, keys);
  EXPECT_EQ(context.kv_cache.values.data, values);
  MiniLlamaContext unbound;
  EXPECT_THROW(ForwardToken(unbound, model, 0), std::runtime_error);
  MiniLlamaContext fresh(&different_model);
  EXPECT_NO_THROW(ForwardBatch(fresh, different_model, MiniBatch::Single(1, 0)));
}

// 后段输入越界必须在任何前向执行之前被拒绝，保留已有历史和 KV 内容。
TEST(ForwardTest, InvalidBatchSuffixPreservesContext) {
  auto model = MakeTestModel();
  MiniLlamaContext context(&model);
  ForwardBatch(context, model, MiniBatch::Single(0, 0));
  const auto keys = context.kv_cache.keys.data;
  const auto values = context.kv_cache.values.data;
  const std::vector<MiniBatch> invalid_batches = {
      {{1, -1}, {1, 2}},
      {{1, model.config.vocab_size}, {1, 2}},
      {{1, 2}, {1, -1}},
      {{1, 2}, {1, model.config.max_seq_len}},
  };
  for (const auto& batch : invalid_batches) {
    EXPECT_THROW(ForwardBatch(context, model, batch), std::out_of_range);
    EXPECT_EQ(context.pos, 0);
    EXPECT_EQ(context.token_history, std::vector<int>({0}));
    EXPECT_EQ(context.kv_cache.keys.data, keys);
    EXPECT_EQ(context.kv_cache.values.data, values);
  }
}

// 回退后重放分支应与新上下文一致，并保留 KV 分配与累计计算计数。
TEST(ForwardTest, RetainedPrefixReplaysLikeFreshContext) {
  auto model = MakeTestModel();
  MiniLlamaContext context(&model);
  ForwardBatch(context, model, MiniBatch::FromTokens({0, 1, 2}));
  context.n_prefill_tokens = 3;
  context.n_decode_tokens = 2;
  const float* keys = context.kv_cache.keys.data.data();
  const float* values = context.kv_cache.values.data.data();

  context.token_history.resize(1);
  context.pos = 0;
  EXPECT_EQ(context.pos, 0);
  EXPECT_EQ(context.token_history, std::vector<int>({0}));
  EXPECT_EQ(context.n_prefill_tokens, 3);
  EXPECT_EQ(context.n_decode_tokens, 2);
  const auto actual = ForwardBatch(context, model, MiniBatch::FromTokens({2, 1}, 1));
  MiniLlamaContext reference(&model);
  const auto expected =
      ForwardBatch(reference, model, MiniBatch::FromTokens({0, 2, 1}));
  ExpectTensorNear(actual, expected);
  ExpectTensorNear(context.kv_cache.keys, reference.kv_cache.keys);
  ExpectTensorNear(context.kv_cache.values, reference.kv_cache.values);
  EXPECT_EQ(context.kv_cache.keys.data.data(), keys);
  EXPECT_EQ(context.kv_cache.values.data.data(), values);
  EXPECT_EQ(context.token_history, reference.token_history);
  EXPECT_EQ(context.pos, reference.pos);

  context = MiniLlamaContext(&model);
  MiniLlamaContext restarted(&model);
  ExpectTensorNear(ForwardBatch(context, model, MiniBatch::Single(2, 0)),
                   ForwardBatch(restarted, model, MiniBatch::Single(2, 0)));
}

// 使用非零 FFN、GQA、偏置和非完整 Q8_0 块，验证现有 CPU 前向。
MiniLlamaModel MakeExtendedTestModel(bool q80) {
  MiniLlamaModel model;
  auto& c = model.config;
  c.vocab_size = 7;
  c.dim = 16;
  c.hidden_dim = 35;  // Down 投影的输入跨过 32 元素量化块边界并包含尾块
  c.n_layers = 2;
  c.n_heads = 4;
  c.n_kv_heads = 2;  // 四个查询头共享两个 KV 头，覆盖 GQA 映射
  c.head_dim = 4;
  c.max_seq_len = 4;
  c.rope_type = q80 ? mini_llama::RopeType::kNeoX : mini_llama::RopeType::kNormal;
  model.loaded = true;
  int seed = 1;
  // 使用确定性的非零数值，避免随机模型引入不可复现的浮点差异。
  auto values = [&](const std::vector<int>& shape) {
    Tensor t(shape);
    for (size_t i = 0; i < t.size(); ++i) {
      t.data[i] = 0.12f * std::sin(static_cast<float>(i + seed) * 0.37f);
    }
    ++seed;
    return t;
  };
  // 所有 Linear 同步切换存储类型，稠密 embedding/norm/bias 继续保持 F32。
  auto weight = [&](const std::vector<int>& shape) {
    auto w = mini_llama::ToQuantizedTensor(values(shape));
    return q80 ? ConvertToQ80(w) : w;
  };
  model.token_embedding = values({c.vocab_size, c.dim});
  model.final_norm = Tensor({c.dim}, 1.0f);
  model.lm_head = weight({c.vocab_size, c.dim});
  model.layers.resize(c.n_layers);
  for (auto& lw : model.layers) {
    const int kv_dim = c.n_kv_heads * c.head_dim;
    lw.attention_norm = Tensor({c.dim}, 1.0f);
    lw.ffn_norm = Tensor({c.dim}, 1.0f);
    lw.wq = weight({c.dim, c.dim});
    lw.wk = weight({kv_dim, c.dim});
    lw.wv = weight({kv_dim, c.dim});
    lw.bq = values({c.dim});
    lw.bk = values({kv_dim});
    lw.bv = values({kv_dim});
    lw.wo = weight({c.dim, c.dim});
    lw.w_gate = weight({c.hidden_dim, c.dim});
    lw.w_up = weight({c.hidden_dim, c.dim});
    lw.w_down = weight({c.dim, c.hidden_dim});
  }
  // 前层保留非零偏置，末层移除偏置，覆盖两种投影分支。
  model.layers.back().bq = Tensor();
  model.layers.back().bk = Tensor();
  model.layers.back().bv = Tensor();
  return model;
}

// 参数 false/true 分别选择 F32/整组 Q8_0 权重；两种参数都执行 CPU 前向。
class ExtendedForwardTest : public ::testing::TestWithParam<bool> {};

// 批量 prefill/decode 应与手工逐 token 执行一致，包含 GQA、bias 与量化尾块。
TEST_P(ExtendedForwardTest, MatchesSequentialPrefillDecodeAndKvCache) {
  MiniLlamaModel model = MakeExtendedTestModel(GetParam());
  MiniLlamaContext batch_ctx(&model);
  MiniLlamaContext sequential_ctx(&model);
  const std::vector<int> tokens = {0, 2, 4, 6};
  Tensor sequential_logits;
  for (int pos = 0; pos < 3; ++pos) {
    sequential_ctx.pos = pos;
    sequential_logits = ForwardToken(sequential_ctx, model, tokens[pos]);
    sequential_ctx.token_history.push_back(tokens[pos]);
  }
  ExpectTensorNear(ForwardBatch(batch_ctx, model, MiniBatch::FromTokens({0, 2, 4})),
                   sequential_logits);
  sequential_ctx.pos = 3;
  sequential_logits = ForwardToken(sequential_ctx, model, tokens.back());
  sequential_ctx.token_history.push_back(tokens.back());
  const Tensor decode_logits =
      ForwardBatch(batch_ctx, model, MiniBatch::Single(6, 3));
  ExpectTensorNear(decode_logits, sequential_logits);
  for (float value : decode_logits.data) {
    EXPECT_TRUE(std::isfinite(value));
  }
  ExpectTensorNear(batch_ctx.kv_cache.keys, sequential_ctx.kv_cache.keys);
  ExpectTensorNear(batch_ctx.kv_cache.values, sequential_ctx.kv_cache.values);
  EXPECT_EQ(batch_ctx.pos, sequential_ctx.pos);
  EXPECT_EQ(batch_ctx.token_history, tokens);
  EXPECT_EQ(batch_ctx.token_history, sequential_ctx.token_history);
}

// 有快照即选择 CUDA；空快照必须拒绝，不能静默执行 CPU。
TEST_P(ExtendedForwardTest, EmptyCudaSnapshotRejectsWithoutChangingStats) {
  MiniLlamaModel model = MakeExtendedTestModel(GetParam());
  model.cuda_weights = std::make_shared<mini_llama::CudaModelWeights>();
  model.cuda_weights->linear_calls = 7;
  MiniLlamaContext ctx(&model);
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
  EXPECT_EQ(mini_llama::ModelCudaLinearCalls(model), 7u);
  EXPECT_EQ(mini_llama::ModelCudaActivationCalls(model), 0u);
  EXPECT_EQ(mini_llama::ModelCudaAttentionCpuFallbacks(model), 0u);
  EXPECT_EQ(mini_llama::ModelCudaHostToDeviceCopies(model), 0u);
  EXPECT_EQ(mini_llama::ModelCudaDeviceToHostCopies(model), 0u);
  EXPECT_TRUE(ctx.cuda_kv_cache.empty());
}

// 测试目的：验证上传保留权重，F32 与 Q8_0 均完整执行设备前向。
TEST_P(ExtendedForwardTest, UploadSnapshotPreservesWeightsAndForwardResults) {
  if (!mini_llama::CudaRuntimeBuilt()) {
    GTEST_SKIP() << "CUDA was not built; model upload was not executed";
  }
  try {
    if (mini_llama::CudaDeviceCount() == 0) {
      GTEST_SKIP() << "no CUDA devices; model upload was not executed";
    }
    mini_llama::CudaGetDeviceInfo(0);
  } catch (const std::runtime_error& error) {
    GTEST_SKIP() << "CUDA device query unavailable: " << error.what();
  }
  MiniLlamaModel model = MakeExtendedTestModel(GetParam());
  MiniLlamaContext before_ctx(&model);
  const auto prefill = MiniBatch::FromTokens({0, 2, 4});
  const Tensor expected = ForwardBatch(before_ctx, model, prefill);
  mini_llama::UploadModelWeightsToCuda(model);
  ASSERT_TRUE(mini_llama::ModelHasCudaWeights(model));
  EXPECT_EQ(mini_llama::ModelCudaUploadedWeightCount(model), 24u);
  EXPECT_EQ(mini_llama::ModelCudaMemoryBytes(model),
            mini_llama::ModelWeightBytes(model));
  EXPECT_EQ(mini_llama::ModelCudaUploadedF32WeightCount(model),
            GetParam() ? 0u : 15u);
  EXPECT_EQ(mini_llama::ModelCudaUploadedQ80WeightCount(model),
            GetParam() ? 15u : 0u);
  EXPECT_EQ(mini_llama::ModelCudaUploadedQ40WeightCount(model), 0u);
  EXPECT_EQ(mini_llama::ModelCudaUploadedQ41WeightCount(model), 0u);
  for (const auto& weight : model.cuda_weights->weights) {
    if (weight.name != "lm_head") {
      continue;
    }
    std::vector<unsigned char> downloaded(weight.buffer.bytes());
    weight.buffer.Download(downloaded.data(), downloaded.size());
    const void* original =
        GetParam() ? static_cast<const void*>(model.lm_head.q8_0_data.data())
                   : static_cast<const void*>(model.lm_head.f32_data.data());
    EXPECT_EQ(std::memcmp(downloaded.data(), original, downloaded.size()), 0);
  }
  MiniLlamaContext after_ctx(&model);
  ExpectTensorNear(ForwardBatch(after_ctx, model, prefill), expected);
  const auto& c = model.config;
  for (int l = 0; l < c.n_layers; ++l) {
    for (int p = 0; p < prefill.num_tokens(); ++p) {
      Tensor key({c.n_kv_heads, c.head_dim});
      std::copy_n(before_ctx.kv_cache.KeyPtr(l, p, 0), key.size(), key.data.begin());
      ExpectTensorNear(after_ctx.cuda_kv_cache.ReadKey(l, p), key, 1e-4f);
    }
  }
  const size_t tokens = prefill.num_tokens();
  EXPECT_EQ(mini_llama::ModelCudaLinearCalls(model), tokens * (7 * c.n_layers + 1));
  EXPECT_EQ(mini_llama::ModelCudaHostToDeviceCopies(model), 0u);
  EXPECT_EQ(mini_llama::ModelCudaDeviceToHostCopies(model), tokens);
  EXPECT_EQ(mini_llama::ModelCudaAttentionCalls(model), tokens * c.n_layers);
  EXPECT_EQ(mini_llama::ModelCudaAttentionCpuFallbacks(model), 0u);
  const auto snapshot = model.cuda_weights;
  model.lm_head.shape[0] += 1;
  EXPECT_THROW(mini_llama::UploadModelWeightsToCuda(model), std::runtime_error);
  EXPECT_EQ(model.cuda_weights, snapshot);
}

// 同时注册 F32 与 Q8_0 参数；仅上传用例需要真实 GPU。
INSTANTIATE_TEST_SUITE_P(F32AndQ80, ExtendedForwardTest,
                         ::testing::Values(false, true));

}  // namespace
