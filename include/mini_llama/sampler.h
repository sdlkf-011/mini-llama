#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_SAMPLER_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_SAMPLER_H_

#include <random>

#include "mini_llama/tensor.h"

namespace mini_llama {

// 核心配置参数
struct SamplingParams {
  float temperature = 0.0f;
  int top_k = 0;          // 0 表示默认禁用
  unsigned int seed = 0;  // 0 表示使用系统随机源
};

// MiniSampler 支持贪心采样、温度采样和 top-k 采样
class MiniSampler {
 public:
  explicit MiniSampler(unsigned int seed = 0);
  explicit MiniSampler(const SamplingParams& params);

  // 使用给定参数从 logits 中采样一个 token
  int Sample(const Tensor& logits, const SamplingParams& params);

  // 贪心采样（始终返回 ArgMax）
  static int SampleGreedy(const Tensor& logits);

  // 温度采样
  int SampleTemperature(const Tensor& logits, float temperature);

  // Top-k 采样（在 top-k 范围内应用温度）
  int SampleTopK(const Tensor& logits, float temperature, int top_k);

 private:
  std::mt19937 rng_;
};

// 为向后兼容保留的旧版自由函数
int SampleGreedy(const Tensor& logits);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_SAMPLER_H_
