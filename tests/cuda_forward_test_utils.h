#ifndef MINI_LLAMA_TESTS_CUDA_FORWARD_TEST_UTILS_H_
#define MINI_LLAMA_TESTS_CUDA_FORWARD_TEST_UTILS_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/model.h"

namespace mini_llama::test {
// 测试自包含；同时检查 shape、有限值和按参考值缩放的容差，避免只看最终 token
inline void RequireCloseTensor(const Tensor& actual, const Tensor& golden,
                               float tolerance = 2e-5f) {
  ASSERT_EQ(actual.shape, golden.shape);
  ASSERT_EQ(actual.size(), golden.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.data[i])) << "element " << i;
    EXPECT_NEAR(actual.data[i], golden.data[i],
                tolerance * std::max(1.0f, std::abs(golden.data[i])))
        << "element " << i;
  }
}

// 确定性的非零权重让漏投影、错误层权重和残差路径更容易在数值对照中暴露
inline Tensor Values(const std::vector<int>& shape, float phase = 0) {
  Tensor result(shape);
  for (size_t i = 0; i < result.size(); ++i) {
    result.data[i] = 0.3f * std::sin(0.37f * static_cast<float>(i) + phase);
  }
  return result;
}

// GPU 不可用时明确跳过；查询成功后的上传、计算和下载异常仍使测试失败
class CudaGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaRuntimeBuilt()) {
      GTEST_SKIP() << "CUDA was not built; GPU computations were not executed";
    }
    try {
      if (CudaDeviceCount() == 0) {
        GTEST_SKIP() << "no CUDA devices; GPU computations were not executed";
      }
      CudaGetDeviceInfo(0);
    } catch (const std::runtime_error& error) {
      GTEST_SKIP() << "CUDA device query unavailable: " << error.what();
    }
  }
};

// 小模型包含 GQA、非零 FFN 与可选 Q/K/V bias；隐藏维度 35 也覆盖量化尾块
inline MiniLlamaModel MakeModel(int layers = 2, bool bias = false) {
  MiniLlamaModel model;
  auto& c = model.config;
  c.vocab_size = 7;
  c.dim = 16;
  c.hidden_dim = 35;
  c.n_layers = layers;
  c.n_heads = 4;
  c.n_kv_heads = 2;
  c.head_dim = 4;
  c.max_seq_len = 4;
  model.loaded = true;
  model.token_embedding = Values({c.vocab_size, c.dim}, 0.3f);
  model.final_norm = Tensor({c.dim}, 1.0f);
  int seed = 1;
  auto weight = [&](const std::vector<int>& shape) {
    return ToQuantizedTensor(Values(shape, 0.19f * seed++));
  };
  model.lm_head = weight({c.vocab_size, c.dim});
  model.layers.resize(layers);
  for (auto& lw : model.layers) {
    const int kv_dim = c.n_kv_heads * c.head_dim;
    lw.attention_norm = Tensor({c.dim}, 1.0f);
    lw.ffn_norm = Tensor({c.dim}, 1.0f);
    lw.wq = weight({c.dim, c.dim});
    lw.wk = weight({kv_dim, c.dim});
    lw.wv = weight({kv_dim, c.dim});
    lw.wo = weight({c.dim, c.dim});
    lw.w_gate = weight({c.hidden_dim, c.dim});
    lw.w_up = weight({c.hidden_dim, c.dim});
    lw.w_down = weight({c.dim, c.hidden_dim});
    if (bias) {
      lw.bq = Values({c.dim}, 0.7f);
      lw.bk = Values({kv_dim}, 0.9f);
      lw.bv = Values({kv_dim}, 1.1f);
    }
  }
  return model;
}

// 完整设备前向；Final Norm 一次，每层七个辅助算子；Embedding 单独不计激活。
inline void RequireDeviceStats(const MiniLlamaModel& model, size_t tokens) {
  const auto& c = model.config;
  EXPECT_EQ(ModelCudaLinearCalls(model), (7 * c.n_layers + 1) * tokens);
  size_t bias_calls = 0;
  for (const auto& w : model.layers) {
    bias_calls += static_cast<size_t>(!w.bq.data.empty()) +
                  static_cast<size_t>(!w.bk.data.empty()) +
                  static_cast<size_t>(!w.bv.data.empty());
  }
  EXPECT_EQ(ModelCudaActivationCalls(model),
            (7 * c.n_layers + 1 + bias_calls) * tokens);
  EXPECT_EQ(ModelCudaAttentionCalls(model), c.n_layers * tokens);
  EXPECT_EQ(ModelCudaAttentionCpuFallbacks(model), 0u);
  EXPECT_EQ(ModelCudaHostToDeviceCopies(model), 0u);
  EXPECT_EQ(ModelCudaHostToDeviceBytes(model), 0u);
  EXPECT_EQ(ModelCudaDeviceToHostCopies(model), tokens);
  EXPECT_EQ(ModelCudaDeviceToHostBytes(model),
            c.vocab_size * sizeof(float) * tokens);
  EXPECT_EQ(ModelCudaKvCacheWriteBytes(model),
            2 * c.n_layers * c.n_kv_heads * c.head_dim * sizeof(float) * tokens);
}

}  // namespace mini_llama::test
#endif  // MINI_LLAMA_TESTS_CUDA_FORWARD_TEST_UTILS_H_
