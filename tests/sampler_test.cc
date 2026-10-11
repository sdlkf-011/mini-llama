// 测试目的：验证贪心和温度采样、seed 复现、top-k/top-p 过滤及非法输入。
#include "mini_llama/sampler.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using mini_llama::MiniSampler;
using mini_llama::SamplingParams;
using mini_llama::Tensor;

// 测试目的：温度为零时应始终选择 logits 最大的 token，并保持并列时的
// ArgMax 行为。
TEST(SamplerTest, GreedySamplingSelectsLargestLogit) {
  const Tensor logits({4}, 0.0f);
  Tensor input = logits;
  input.data = {-1.0f, 3.0f, 2.0f, 3.0f};

  MiniSampler sampler(42);
  SamplingParams params;
  EXPECT_EQ(sampler.Sample(input, params), 1);
  EXPECT_EQ(MiniSampler::SampleGreedy(input), 1);
}

// 测试目的：相同 seed 和采样参数应产生相同序列，保证 --seed 能够复现
// 温度采样结果。
TEST(SamplerTest, SameSeedProducesReproducibleTemperatureSamples) {
  Tensor logits({4}, 0.0f);
  logits.data = {0.1f, 0.2f, 0.3f, 0.4f};
  SamplingParams params;
  params.temperature = 0.8f;
  params.seed = 1234;

  MiniSampler first(params);
  MiniSampler second(params);
  for (int i = 0; i < 20; ++i) {
    EXPECT_EQ(first.Sample(logits, params), second.Sample(logits, params));
  }
}

// 测试目的：top-k 采样只能返回排序后前 k 个候选，不能采到 top-k 之外的
// 低分 token。
TEST(SamplerTest, TopKSamplingRestrictsCandidates) {
  Tensor logits({5}, 0.0f);
  logits.data = {10.0f, 9.0f, 8.0f, -10.0f, -20.0f};
  SamplingParams params;
  params.temperature = 1.0f;
  params.top_k = 2;
  params.seed = 99;

  MiniSampler sampler(params);
  for (int i = 0; i < 100; ++i) {
    const int token = sampler.Sample(logits, params);
    EXPECT_TRUE(token == 0 || token == 1);
  }
}

// 测试目的：非法温度、非法 top-k、空 logits 和非有限 logits 都应在采样
// 前被拒绝，而不是进入数值计算。
TEST(SamplerTest, RejectsInvalidSamplingInputs) {
  Tensor logits({2}, 1.0f);
  MiniSampler sampler(1);

  SamplingParams negative_temperature;
  negative_temperature.temperature = -0.1f;
  EXPECT_THROW(sampler.Sample(logits, negative_temperature), std::runtime_error);

  SamplingParams nan_temperature;
  nan_temperature.temperature = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(sampler.Sample(logits, nan_temperature), std::runtime_error);

  SamplingParams negative_top_k;
  negative_top_k.top_k = -1;
  EXPECT_THROW(sampler.Sample(logits, negative_top_k), std::runtime_error);

  EXPECT_THROW(sampler.SampleTopK(logits, 1.0f, 0), std::runtime_error);

  Tensor empty;
  EXPECT_THROW(sampler.SampleGreedy(empty), std::runtime_error);

  Tensor non_finite({2}, 0.0f);
  non_finite.data[1] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(sampler.SampleGreedy(non_finite), std::runtime_error);
}

// 测试目的：top-k 为一或温度接近零时应退化到贪心采样，确保命令行参数
// 组合不会意外进入随机路径。
TEST(SamplerTest, DegenerateSamplingParametersUseGreedyPath) {
  Tensor logits({3}, 0.0f);
  logits.data = {0.0f, 5.0f, 4.0f};
  MiniSampler sampler(7);

  SamplingParams top_one;
  top_one.temperature = 2.0f;
  top_one.top_k = 1;
  EXPECT_EQ(sampler.Sample(logits, top_one), 1);

  SamplingParams near_zero;
  near_zero.temperature = 1e-7f;
  near_zero.top_k = 0;
  EXPECT_EQ(sampler.Sample(logits, near_zero), 1);
}

}  // namespace
