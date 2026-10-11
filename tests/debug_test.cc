// 测试目的：验证调试输出格式、logits 排序边界及 benchmark 的统计和上下文限制。
#include "mini_llama/debug.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace {

using mini_llama::BenchmarkResult;
using mini_llama::DumpKvCacheInfo;
using mini_llama::DumpLogitsTopK;
using mini_llama::DumpTensorShape;
using mini_llama::KvCache;
using mini_llama::MiniLlamaModel;
using mini_llama::RunBenchmark;
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

// 测试目的：整体吞吐量和解码吞吐量应使用各自对应的词元数量及耗时，
// 无有效耗时时应安全返回零。
TEST(DebugTest, BenchmarkResultCalculatesThroughput) {
  BenchmarkResult result;
  result.n_generated_tokens = 6;
  result.n_decode_tokens = 4;
  result.prefill_ms = 20.0;
  result.decode_ms = 10.0;

  EXPECT_DOUBLE_EQ(result.tokens_per_sec(), 200.0);
  EXPECT_DOUBLE_EQ(result.decode_tokens_per_sec(), 400.0);

  result.prefill_ms = 0.0;
  result.decode_ms = 0.0;
  EXPECT_DOUBLE_EQ(result.tokens_per_sec(), 0.0);
  EXPECT_DOUBLE_EQ(result.decode_tokens_per_sec(), 0.0);
}

// 测试目的：张量形状输出应包含名称、完整维度、数据类型和元素总数，
// 便于定位前向过程中的维度问题。
TEST(DebugTest, DumpTensorShapePrintsTensorMetadata) {
  const Tensor tensor({2, 3}, 1.0f);

  testing::internal::CaptureStdout();
  DumpTensorShape(tensor, "hidden");
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "  tensor: hidden shape=[2, 3] dtype=f32 size=6\n");
}

// 测试目的：Top-k 输出应只保留指定数量的最大 logits，并按值从大到小
// 展示对应的 token 编号。
TEST(DebugTest, DumpLogitsTopKPrintsLargestValuesInOrder) {
  Tensor logits({4}, 0.0f);
  logits.data = {0.5f, 3.0f, -1.0f, 2.0f};

  testing::internal::CaptureStdout();
  DumpLogitsTopK(logits, 2);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("logits top-2"), std::string::npos);
  const size_t first = output.find("id=   1 value=3.000000");
  const size_t second = output.find("id=   3 value=2.000000");
  ASSERT_NE(first, std::string::npos);
  ASSERT_NE(second, std::string::npos);
  EXPECT_LT(first, second);
  EXPECT_EQ(output.find("id=   0"), std::string::npos);
}

// 测试目的：Top-k 只接受一维 logits；对于非一维输入应输出明确提示，
// 而不是把多维数据误当作词表。
TEST(DebugTest, DumpLogitsTopKRejectsNonVectorTensor) {
  const Tensor logits({2, 2}, 0.0f);

  testing::internal::CaptureStdout();
  DumpLogitsTopK(logits, 2);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output, "  logits: invalid shape for top-k dump\n");
}

// 测试目的：KV 缓存输出应完整展示层数、序列长度、KV 头数、头维度、
// 当前位置以及键值张量形状。
TEST(DebugTest, DumpKvCacheInfoPrintsConfigurationAndPosition) {
  const KvCache cache(2, 8, 3, 4);

  testing::internal::CaptureStdout();
  DumpKvCacheInfo(cache, 5);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("layers=2 seq_len=8 n_kv_heads=3 head_dim=4"),
            std::string::npos);
  EXPECT_NE(output.find("current_pos: 5"), std::string::npos);
  EXPECT_NE(output.find("keys shape: [2, 8, 3, 4]"), std::string::npos);
  EXPECT_NE(output.find("values shape: [2, 8, 3, 4]"), std::string::npos);
}

// 测试目的：基准运行器应完成一次预填充和指定次数的生成，并分别记录
// 提示词元、生成词元和实际逐 token 解码次数。
TEST(DebugTest, RunBenchmarkCollectsPrefillAndDecodeStatistics) {
  const MiniLlamaModel model = MakeTestModel();

  const BenchmarkResult result = RunBenchmark(model, {0}, 3, 42, false);

  EXPECT_EQ(result.n_prompt_tokens, 1);
  EXPECT_EQ(result.n_generated_tokens, 3);
  EXPECT_EQ(result.n_decode_tokens, 2);
  EXPECT_GE(result.prefill_ms, 0.0);
  EXPECT_GE(result.decode_ms, 0.0);
}

// 测试目的：生成数量应受模型最大序列长度限制，提示已经占满上下文时
// 不应再采样或执行解码。
TEST(DebugTest, RunBenchmarkStopsAtMaximumSequenceLength) {
  const MiniLlamaModel model = MakeTestModel();

  const BenchmarkResult result = RunBenchmark(model, {0, 1, 2, 0}, 3, 42, false);

  EXPECT_EQ(result.n_prompt_tokens, 4);
  EXPECT_EQ(result.n_generated_tokens, 0);
  EXPECT_EQ(result.n_decode_tokens, 0);
  EXPECT_DOUBLE_EQ(result.decode_ms, 0.0);
}

// 测试目的：详细模式应输出预填充、每步解码、logits 和 KV 缓存信息，
// 确认调试开关覆盖完整的基准执行过程。
TEST(DebugTest, RunBenchmarkVerboseModePrintsDebugInformation) {
  const MiniLlamaModel model = MakeTestModel();

  testing::internal::CaptureStdout();
  const BenchmarkResult result = RunBenchmark(model, {0}, 2, 42, true);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(result.n_generated_tokens, 2);
  EXPECT_NE(output.find("[verbose] prefill 1 tokens"), std::string::npos);
  EXPECT_NE(output.find("tensor: logits shape=[3]"), std::string::npos);
  EXPECT_NE(output.find("logits top-3"), std::string::npos);
  EXPECT_NE(output.find("[verbose] Decode step 0"), std::string::npos);
  EXPECT_NE(output.find("[verbose] Decode step 1"), std::string::npos);
  EXPECT_NE(output.find("KV cache: layers=1"), std::string::npos);
}

// 测试非正数或超过词表大小的 k 会展示完整词表，并按 logits 降序输出。
TEST(DebugUnitTest, InvalidTopKShowsCompleteVocabulary) {
  mini_llama::Tensor logits({3}, 0.0f);
  logits.data = {1.0f, 3.0f, 2.0f};

  for (int k : {-1, 0, 4}) {
    SCOPED_TRACE(k);
    testing::internal::CaptureStdout();
    mini_llama::DumpLogitsTopK(logits, k);
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("logits top-3"), std::string::npos);
    const size_t first = output.find("id=   1");
    const size_t second = output.find("id=   2");
    const size_t third = output.find("id=   0");
    ASSERT_NE(first, std::string::npos);
    ASSERT_NE(second, std::string::npos);
    ASSERT_NE(third, std::string::npos);
    EXPECT_LT(first, second);
    EXPECT_LT(second, third);
  }
}

// 测试 Top-k 临时设置的小数格式不会改变调用方输出流的 flags 和 precision。
TEST(DebugUnitTest, TopKPreservesOutputStreamFormatting) {
  const mini_llama::Tensor logits({1}, 1.0f);
  const auto flags = std::cout.flags();
  const auto precision = std::cout.precision();

  testing::internal::CaptureStdout();
  mini_llama::DumpLogitsTopK(logits, 1);
  testing::internal::GetCapturedStdout();

  EXPECT_EQ(std::cout.flags(), flags);
  EXPECT_EQ(std::cout.precision(), precision);
}

// 测试未提供当前位置时省略位置行，但仍保留 KV 缓存的结构信息。
TEST(DebugUnitTest, KvCacheOmitsUnspecifiedPosition) {
  const mini_llama::KvCache cache(1, 2, 1, 2);

  testing::internal::CaptureStdout();
  mini_llama::DumpKvCacheInfo(cache, -1);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output.find("current_pos:"), std::string::npos);
  EXPECT_NE(output.find("keys shape: [1, 2, 1, 2]"), std::string::npos);
  EXPECT_NE(output.find("values shape: [1, 2, 1, 2]"), std::string::npos);
}

}  // namespace
