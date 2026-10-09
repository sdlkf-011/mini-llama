#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/forward.h"
#include "mini_llama/model.h"
#include "mini_llama/quant.h"

namespace {
using namespace mini_llama;

// 测试自包含：不依赖额外的 CUDA 测试辅助头文件。
void RequireCloseTensor(const Tensor& actual, const Tensor& golden,
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

Tensor Values(const std::vector<int>& shape, float phase = 0) {
  Tensor result(shape);
  for (size_t i = 0; i < result.size(); ++i) {
    result.data[i] = 0.3f * std::sin(0.37f * static_cast<float>(i) + phase);
  }
  return result;
}

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

using CudaForwardLinearTest = CudaGpuTest;

MiniLlamaModel MakeModel(int layers = 2, bool bias = false) {
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

// 当前链路中只有 Linear 在 GPU，所有统计对应实际 fallback 边界。
// 这组断言不能替代尚缺 CUDA Attention/激活链路的 0 H2D / 1 D2H 验收。
void RequireFallbackStats(const MiniLlamaModel& model, size_t tokens) {
  const auto& c = model.config;
  const size_t linear_calls = static_cast<size_t>(7 * c.n_layers + 1) * tokens;
  EXPECT_EQ(ModelCudaLinearCalls(model), linear_calls);
  EXPECT_EQ(ModelCudaAttentionCalls(model), 0u);
  EXPECT_EQ(ModelCudaAttentionCpuFallbacks(model), c.n_layers * tokens);
  EXPECT_EQ(ModelCudaHostToDeviceCopies(model), linear_calls);
  EXPECT_EQ(ModelCudaDeviceToHostCopies(model), linear_calls);
  const size_t input_elements = c.n_layers * (6 * c.dim + c.hidden_dim) + c.dim;
  const size_t output_elements =
      c.n_layers * (3 * c.dim + 2 * c.n_kv_heads * c.head_dim + 2 * c.hidden_dim) +
      c.vocab_size;
  EXPECT_EQ(ModelCudaHostToDeviceBytes(model),
            input_elements * sizeof(float) * tokens);
  EXPECT_EQ(ModelCudaDeviceToHostBytes(model),
            output_elements * sizeof(float) * tokens);
}

TEST_F(CudaForwardLinearTest, ResidentF32ProjectionsMatchCpuPrefillAndDecode) {
  MiniLlamaModel cpu_model = MakeModel();
  MiniLlamaModel cuda_model = cpu_model;
  UploadModelWeightsToCuda(cuda_model, 0);
  ASSERT_EQ(ModelCudaUploadedWeightCount(cuda_model), 21u);
  ASSERT_EQ(ModelCudaUploadedF32WeightCount(cuda_model), 15u);
  ASSERT_EQ(ModelCudaMemoryBytes(cuda_model), ModelWeightBytes(cpu_model));
  const auto snapshot = cuda_model.cuda_weights;
  std::vector<const void*> pointers;
  for (const auto& weight : snapshot->weights)
    pointers.push_back(weight.buffer.data());
  MiniLlamaContext cpu_ctx(&cpu_model), cuda_ctx(&cuda_model);
  ResetModelCudaRuntimeStats(cuda_model);
  const auto prefill = MiniBatch::FromTokens({0, 2});
  RequireCloseTensor(ForwardBatch(cuda_ctx, cuda_model, prefill),
                     ForwardBatch(cpu_ctx, cpu_model, prefill), 1e-4f);
  RequireFallbackStats(cuda_model, 2);
  // 将 CPU 投影副本置零作为上传探针：Decode 仍应读取已有显存快照。
  // 如果重新上传 CPU 权重或回退到 CPU，结果会偏离未修改的 Golden。
  // 生产中修改权重后须显式重新上传，此处故意保持旧快照验证所有权契约。
  for (auto& layer : cuda_model.layers) {
    for (auto* w : {&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.w_gate,
                    &layer.w_up, &layer.w_down}) {
      std::fill(w->f32_data.begin(), w->f32_data.end(), 0.0f);
    }
  }
  std::fill(cuda_model.lm_head.f32_data.begin(), cuda_model.lm_head.f32_data.end(),
            0.0f);
  // Decode 单步独立计数，初始化与静态权重上传均不在此范围。
  ResetModelCudaRuntimeStats(cuda_model);
  const auto decode = MiniBatch::Single(4, 2);
  RequireCloseTensor(ForwardBatch(cuda_ctx, cuda_model, decode),
                     ForwardBatch(cpu_ctx, cpu_model, decode), 1e-4f);
  RequireCloseTensor(cuda_ctx.kv_cache.keys, cpu_ctx.kv_cache.keys, 1e-4f);
  RequireCloseTensor(cuda_ctx.kv_cache.values, cpu_ctx.kv_cache.values, 1e-4f);
  RequireFallbackStats(cuda_model, 1);
  EXPECT_EQ(cuda_ctx.token_history, cpu_ctx.token_history);
  ASSERT_EQ(cuda_model.cuda_weights, snapshot);
  ASSERT_EQ(snapshot->weights.size(), pointers.size());
  for (size_t i = 0; i < pointers.size(); ++i) {
    EXPECT_EQ(snapshot->weights[i].buffer.data(), pointers[i]);
  }
  EXPECT_EQ(ModelCudaUploadedWeightCount(cuda_model), 21u);
  EXPECT_EQ(ModelCudaUploadedF32WeightCount(cuda_model), 15u);
  std::cout << "Decode stats: uploaded=21, F32=15, linear="
            << ModelCudaLinearCalls(cuda_model)
            << ", attention=" << ModelCudaAttentionCalls(cuda_model)
            << ", attention_cpu_fallbacks="
            << ModelCudaAttentionCpuFallbacks(cuda_model)
            << ", H2D=" << ModelCudaHostToDeviceCopies(cuda_model)
            << ", D2H=" << ModelCudaDeviceToHostCopies(cuda_model) << '\n';
}

TEST_F(CudaForwardLinearTest, CoversAllLayersGqaAndOptionalBias) {
  for (int layers : {1, 3}) {
    MiniLlamaModel cpu_model = MakeModel(layers, true);
    cpu_model.config.rope_type = RopeType::kNeoX;
    MiniLlamaModel cuda_model = cpu_model;
    UploadModelWeightsToCuda(cuda_model);
    EXPECT_EQ(ModelCudaUploadedWeightCount(cuda_model), 3u + 12u * layers);
    EXPECT_EQ(ModelCudaUploadedF32WeightCount(cuda_model), 1u + 7u * layers);
    MiniLlamaContext cpu_ctx(&cpu_model), cuda_ctx(&cuda_model);
    ResetModelCudaRuntimeStats(cuda_model);
    const auto batch = MiniBatch::FromTokens({0, 2, 4});
    RequireCloseTensor(ForwardBatch(cuda_ctx, cuda_model, batch),
                       ForwardBatch(cpu_ctx, cpu_model, batch), 1e-4f);
    RequireFallbackStats(cuda_model, 3);
  }
}

TEST_F(CudaForwardLinearTest, ResidentWeightsAreUsedUntilSnapshotIsCleared) {
  MiniLlamaModel cpu_model = MakeModel();
  MiniLlamaModel uploaded_model = cpu_model;
  UploadModelWeightsToCuda(uploaded_model);
  MiniLlamaContext cpu_ctx(&cpu_model), uploaded_ctx(&uploaded_model);
  ResetModelCudaRuntimeStats(uploaded_model);
  const auto batch = MiniBatch::Single(1, 0);
  RequireCloseTensor(ForwardBatch(uploaded_ctx, uploaded_model, batch),
                     ForwardBatch(cpu_ctx, cpu_model, batch));
  RequireFallbackStats(uploaded_model, 1);
  ClearModelCudaWeights(uploaded_model);
  ASSERT_FALSE(ModelHasCudaWeights(uploaded_model));
  MiniLlamaContext cleared_ctx(&uploaded_model);
  RequireCloseTensor(ForwardBatch(cleared_ctx, uploaded_model, batch),
                     ForwardBatch(cpu_ctx, cpu_model, batch));
  EXPECT_EQ(ModelCudaLinearCalls(uploaded_model), 0u);
  EXPECT_EQ(ModelCudaHostToDeviceCopies(uploaded_model), 0u);
  EXPECT_EQ(ModelCudaDeviceToHostCopies(uploaded_model), 0u);
}

TEST_F(CudaForwardLinearTest, MissingOrQuantizedWeightsUseCpuLinearFallback) {
  MiniLlamaModel cpu_model = MakeModel();
  auto& wk = cpu_model.layers[0].wk;
  wk.q8_0_data = QuantizeToQ80(ToTensor(wk));
  wk.f32_data.clear();
  wk.type = QuantType::kQ80;
  MiniLlamaModel cuda_model = cpu_model;
  UploadModelWeightsToCuda(cuda_model);
  ASSERT_EQ(ModelCudaUploadedF32WeightCount(cuda_model), 14u);
  // 改名模拟一次查找未命中，不改上传数或运行计数。
  for (auto& w : cuda_model.cuda_weights->weights) {
    if (w.name == "layers.1.wq") w.name = "missing.wq";
  }
  MiniLlamaContext cpu_ctx(&cpu_model), cuda_ctx(&cuda_model);
  ResetModelCudaRuntimeStats(cuda_model);
  const auto batch = MiniBatch::Single(1, 0);
  RequireCloseTensor(ForwardBatch(cuda_ctx, cuda_model, batch),
                     ForwardBatch(cpu_ctx, cpu_model, batch), 1e-4f);
  EXPECT_EQ(ModelCudaLinearCalls(cuda_model), 13u);
  EXPECT_EQ(ModelCudaHostToDeviceCopies(cuda_model), 13u);
  EXPECT_EQ(ModelCudaDeviceToHostCopies(cuda_model), 13u);
  EXPECT_EQ(ModelCudaAttentionCalls(cuda_model), 0u);
  EXPECT_EQ(ModelCudaAttentionCpuFallbacks(cuda_model), 2u);
}
}  // namespace
