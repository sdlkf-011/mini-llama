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

// 测试自包含；同时检查 shape、有限值和按参考值缩放的容差，避免只看最终 token
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

// 确定性的非零权重让漏投影、错误层权重和残差路径更容易在数值对照中暴露
Tensor Values(const std::vector<int>& shape, float phase = 0) {
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

using CudaForwardLinearTest = CudaGpuTest;

// 小模型包含 GQA、非零 FFN 与可选 Q/K/V bias；隐藏维度 35 也覆盖量化尾块
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

// 必需稠密权重被清空时上传必须失败，并保留完整的旧设备快照。
// 测试目的：逐项清空必需稠密权重，验证上传失败且保留完整旧快照。
TEST_F(CudaForwardLinearTest, EmptyRequiredDenseStoragePreservesSnapshot) {
  auto model = MakeModel();
  UploadModelWeightsToCuda(model);
  const auto snapshot = model.cuda_weights;
  for (Tensor* weight :
       {&model.token_embedding, &model.final_norm, &model.layers[0].attention_norm,
        &model.layers[1].ffn_norm}) {
    const Tensor original = *weight;
    weight->data.clear();
    EXPECT_THROW(UploadModelWeightsToCuda(model), std::runtime_error);
    EXPECT_EQ(model.cuda_weights, snapshot);
    *weight = original;
  }
}

// 已有 shape 的空 bias 仍表示省略，不应按必需权重处理。
// 测试目的：验证空的可选 QKV 偏置被省略，不影响其他权重上传及容量统计。
TEST_F(CudaForwardLinearTest, EmptyOptionalBiasStorageIsOmitted) {
  auto model = MakeModel(1, true);
  for (Tensor* bias :
       {&model.layers[0].bq, &model.layers[0].bk, &model.layers[0].bv}) {
    bias->data.clear();
  }
  UploadModelWeightsToCuda(model);
  EXPECT_EQ(ModelCudaUploadedWeightCount(model), 12u);
  EXPECT_EQ(ModelCudaMemoryBytes(model), ModelWeightBytes(model));
}

// QKV/RoPE/Attention 留在设备；其余模块边界仍上传和下载激活。
void RequireAttentionStageStats(const MiniLlamaModel& model, size_t tokens) {
  const auto& c = model.config;
  const size_t linear_calls = static_cast<size_t>(7 * c.n_layers + 1) * tokens;
  EXPECT_EQ(ModelCudaLinearCalls(model), linear_calls);
  EXPECT_EQ(ModelCudaAttentionCalls(model), c.n_layers * tokens);
  EXPECT_EQ(ModelCudaAttentionCpuFallbacks(model), 0u);
  EXPECT_EQ(ModelCudaHostToDeviceCopies(model), (5 * c.n_layers + 1) * tokens);
  EXPECT_EQ(ModelCudaDeviceToHostCopies(model), (5 * c.n_layers + 1) * tokens);
  const size_t input_elements = c.n_layers * (4 * c.dim + c.hidden_dim) + c.dim;
  const size_t output_elements =
      c.n_layers * (3 * c.dim + 2 * c.hidden_dim) + c.vocab_size;
  EXPECT_EQ(ModelCudaHostToDeviceBytes(model),
            input_elements * sizeof(float) * tokens);
  EXPECT_EQ(ModelCudaDeviceToHostBytes(model),
            output_elements * sizeof(float) * tokens);
}

// 测试目的：对照设备 QKV/Attention 链路与 CPU 的 prefill/decode、GPU KV 和序列状态
// 上传后置零 CPU Linear 权重作为探针，确认 decode 仍借用原设备快照和指针
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
  RequireAttentionStageStats(cuda_model, 2);
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
  for (int l = 0; l < cpu_model.config.n_layers; ++l) {
    for (int p = 0; p <= 2; ++p) {
      Tensor k({2, 4}), v(k.shape);
      std::copy_n(cpu_ctx.kv_cache.KeyPtr(l, p, 0), 8, k.data.begin());
      std::copy_n(cpu_ctx.kv_cache.valuePtr(l, p, 0), 8, v.data.begin());
      RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadKey(l, p), k, 1e-4f);
      RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadValue(l, p), v, 1e-4f);
    }
  }
  RequireAttentionStageStats(cuda_model, 1);
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

// 测试目的：验证不同层数下所有投影均被分发，并与 CPU 的 GQA、NeoX RoPE
// 和可选 bias 结果一致；运行计数须随真实层数和 token 数变化
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
    RequireAttentionStageStats(cuda_model, 3);
  }
}

// 测试目的：上传快照存在时使用 F32 CUDA Linear，清除后恢复 CPU 前向且查询计数为零
TEST_F(CudaForwardLinearTest, ResidentWeightsAreUsedUntilSnapshotIsCleared) {
  MiniLlamaModel cpu_model = MakeModel();
  MiniLlamaModel uploaded_model = cpu_model;
  UploadModelWeightsToCuda(uploaded_model);
  MiniLlamaContext cpu_ctx(&cpu_model), uploaded_ctx(&uploaded_model);
  ResetModelCudaRuntimeStats(uploaded_model);
  const auto batch = MiniBatch::Single(1, 0);
  RequireCloseTensor(ForwardBatch(uploaded_ctx, uploaded_model, batch),
                     ForwardBatch(cpu_ctx, cpu_model, batch));
  RequireAttentionStageStats(uploaded_model, 1);
  ClearModelCudaWeights(uploaded_model);
  ASSERT_FALSE(ModelHasCudaWeights(uploaded_model));
  MiniLlamaContext cleared_ctx(&uploaded_model);
  RequireCloseTensor(ForwardBatch(cleared_ctx, uploaded_model, batch),
                     ForwardBatch(cpu_ctx, cpu_model, batch));
  EXPECT_EQ(ModelCudaLinearCalls(uploaded_model), 0u);
  EXPECT_EQ(ModelCudaHostToDeviceCopies(uploaded_model), 0u);
  EXPECT_EQ(ModelCudaDeviceToHostCopies(uploaded_model), 0u);
}

// 测试目的：重复上传替换并释放旧快照；清除一个模型不释放副本仍共享的显存
// 最后一个持有者清除引用后，弱引用才失效，验证共享快照的资源生命周期
TEST_F(CudaForwardLinearTest, ReuploadAndClearRespectSharedOwnership) {
  MiniLlamaModel model = MakeModel();
  UploadModelWeightsToCuda(model);
  std::weak_ptr<CudaModelWeights> first = model.cuda_weights;
  const size_t uploaded_bytes = ModelCudaMemoryBytes(model);

  UploadModelWeightsToCuda(model);
  EXPECT_TRUE(first.expired());
  EXPECT_EQ(ModelCudaUploadedWeightCount(model), 21u);
  EXPECT_EQ(ModelCudaMemoryBytes(model), uploaded_bytes);

  MiniLlamaModel shared = model;
  std::weak_ptr<CudaModelWeights> second = model.cuda_weights;
  ClearModelCudaWeights(model);
  EXPECT_FALSE(ModelHasCudaWeights(model));
  EXPECT_FALSE(second.expired());
  EXPECT_TRUE(ModelHasCudaWeights(shared));
  ClearModelCudaWeights(shared);
  EXPECT_TRUE(second.expired());
}

// 测试目的：当前模型前向中的 Q8_0 和未命中 F32 权重分别走 CPU Linear
// CUDA 调用及拷贝只统计实际命中的 F32 投影，混合结果仍须与 CPU 一致
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
