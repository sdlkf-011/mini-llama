#include "mini_llama/forward.h"

#include <gtest/gtest.h>

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
  for (float value : decode_logits.data) EXPECT_TRUE(std::isfinite(value));
  ExpectTensorNear(batch_ctx.kv_cache.keys, sequential_ctx.kv_cache.keys);
  ExpectTensorNear(batch_ctx.kv_cache.values, sequential_ctx.kv_cache.values);
  EXPECT_EQ(batch_ctx.pos, sequential_ctx.pos);
  EXPECT_EQ(batch_ctx.token_history, tokens);
  EXPECT_EQ(batch_ctx.token_history, sequential_ctx.token_history);
}

// 上传接口只管理权重；快照存在时仍使用 CPU 前向，并保留设备运行统计。
TEST_P(ExtendedForwardTest, CudaSnapshotDoesNotSelectInferenceBackend) {
  MiniLlamaModel cpu_model = MakeExtendedTestModel(GetParam());
  MiniLlamaModel model_with_snapshot = cpu_model;
  model_with_snapshot.cuda_weights =
      std::make_shared<mini_llama::CudaModelWeights>();
  // 手工快照只验证后端选择契约，不执行实际上传；非零哨兵验证统计未被清零。
  model_with_snapshot.cuda_weights->linear_calls = 7;
  const auto snapshot = model_with_snapshot.cuda_weights;
  MiniLlamaContext cpu_ctx(&cpu_model);
  MiniLlamaContext snapshot_ctx(&model_with_snapshot);
  const MiniBatch prefill = MiniBatch::FromTokens({0, 2, 4});
  ExpectTensorNear(ForwardBatch(snapshot_ctx, model_with_snapshot, prefill),
                   ForwardBatch(cpu_ctx, cpu_model, prefill));
  const MiniBatch decode = MiniBatch::Single(6, 3);
  ExpectTensorNear(ForwardBatch(snapshot_ctx, model_with_snapshot, decode),
                   ForwardBatch(cpu_ctx, cpu_model, decode));
  ExpectTensorNear(snapshot_ctx.kv_cache.keys, cpu_ctx.kv_cache.keys);
  ExpectTensorNear(snapshot_ctx.kv_cache.values, cpu_ctx.kv_cache.values);
  EXPECT_EQ(snapshot_ctx.pos, cpu_ctx.pos);
  EXPECT_EQ(snapshot_ctx.token_history, cpu_ctx.token_history);
  EXPECT_EQ(model_with_snapshot.cuda_weights, snapshot);
  EXPECT_EQ(mini_llama::ModelCudaLinearCalls(model_with_snapshot), 7u);
  EXPECT_EQ(mini_llama::ModelCudaActivationCalls(model_with_snapshot), 0u);
  EXPECT_EQ(mini_llama::ModelCudaAttentionCpuFallbacks(model_with_snapshot), 0u);
  EXPECT_EQ(mini_llama::ModelCudaHostToDeviceCopies(model_with_snapshot), 0u);
  EXPECT_EQ(mini_llama::ModelCudaDeviceToHostCopies(model_with_snapshot), 0u);
}

// 真实 GPU 上校验上传快照的字节与统计，上传后仍执行相同的 CPU 前向。
TEST_P(ExtendedForwardTest, UploadSnapshotPreservesWeightsAndCpuForward) {
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
    if (weight.name != "lm_head") continue;
    std::vector<unsigned char> downloaded(weight.buffer.bytes());
    weight.buffer.Download(downloaded.data(), downloaded.size());
    const void* original =
        GetParam() ? static_cast<const void*>(model.lm_head.q8_0_data.data())
                   : static_cast<const void*>(model.lm_head.f32_data.data());
    EXPECT_EQ(std::memcmp(downloaded.data(), original, downloaded.size()), 0);
  }
  MiniLlamaContext after_ctx(&model);
  ExpectTensorNear(ForwardBatch(after_ctx, model, prefill), expected);
  ExpectTensorNear(after_ctx.kv_cache.keys, before_ctx.kv_cache.keys);
  EXPECT_EQ(mini_llama::ModelCudaLinearCalls(model), 0u);
  const auto snapshot = model.cuda_weights;
  model.lm_head.shape[0] += 1;
  EXPECT_THROW(mini_llama::UploadModelWeightsToCuda(model), std::runtime_error);
  EXPECT_EQ(model.cuda_weights, snapshot);
}

// 同时注册 F32 与 Q8_0 参数；仅上传用例需要真实 GPU。
INSTANTIATE_TEST_SUITE_P(F32AndQ80, ExtendedForwardTest,
                         ::testing::Values(false, true));

}  // namespace
