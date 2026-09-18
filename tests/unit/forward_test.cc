#include "mini_llama/forward.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace {

using mini_llama::ForwardBatch;
using mini_llama::ForwardToken;
using mini_llama::MiniBatch;
using mini_llama::MiniLlamaContext;
using mini_llama::MiniLlamaModel;
using mini_llama::Tensor;

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
  layer.wq = Tensor({2, 2}, 0.0f);
  layer.wq.data = {1.0f, 0.0f, 0.0f, 1.0f};
  layer.wk = layer.wq;
  layer.wv = layer.wq;
  layer.wo = layer.wq;
  layer.ffn_norm = Tensor({2}, 1.0f);
  layer.w_gate = Tensor({2, 2}, 0.0f);
  layer.w_up = Tensor({2, 2}, 0.0f);
  layer.w_down = Tensor({2, 2}, 0.0f);

  model.final_norm = Tensor({2}, 1.0f);
  model.lm_head = Tensor({3, 2}, 0.0f);
  model.lm_head.data = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  return model;
}

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

  model.layers[0].wq = Tensor({1, 2}, 0.0f);
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

}  // namespace
