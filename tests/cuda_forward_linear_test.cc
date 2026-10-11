// 测试目的：验证设备权重快照、F32/Q8/Q4 投影与 CPU 一致，并检查上传和所有权。
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
#include "names_test.h"

namespace {
using namespace mini_llama;

using namespace mini_llama::test;
using CudaForwardLinearTest = CudaGpuTest;

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

// 测试目的：对照设备 QKV/Attention 链路与 CPU 的 prefill/decode、GPU KV 和序列状态
// 上传后置零 CPU 权重作为探针，确认 decode 仍借用原设备快照和指针
TEST_F(CudaForwardLinearTest, ResidentF32ProjectionsMatchCpuPrefillAndDecode) {
  MiniLlamaModel cpu_model = MakeModel();
  MiniLlamaModel cuda_model = cpu_model;
  MiniLlamaContext cpu_ctx(&cpu_model), cuda_ctx(&cuda_model);
  const auto prefill = MiniBatch::FromTokens({0, 2});
  const auto prefill_expected = ForwardBatch(cpu_ctx, cpu_model, prefill);
  UploadModelWeightsToCuda(cuda_model, 0);
  ASSERT_EQ(ModelCudaUploadedWeightCount(cuda_model), 21u);
  ASSERT_EQ(ModelCudaUploadedF32WeightCount(cuda_model), 15u);
  ASSERT_EQ(ModelCudaMemoryBytes(cuda_model), ModelWeightBytes(cpu_model));
  const auto snapshot = cuda_model.cuda_weights;
  std::vector<const void*> pointers;
  for (const auto& weight : snapshot->weights) {
    pointers.push_back(weight.buffer.data());
  }
  ResetModelCudaRuntimeStats(cuda_model);
  RequireCloseTensor(ForwardBatch(cuda_ctx, cuda_model, prefill), prefill_expected,
                     1e-4f, 1e-4f, "prefill logits");
  RequireDeviceStats(cuda_model, 2);
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
  std::fill(cuda_model.token_embedding.data.begin(),
            cuda_model.token_embedding.data.end(), 0.0f);
  std::fill(cuda_model.final_norm.data.begin(), cuda_model.final_norm.data.end(),
            0.0f);
  for (auto& layer : cuda_model.layers) {
    std::fill(layer.attention_norm.data.begin(), layer.attention_norm.data.end(),
              0.0f);
    std::fill(layer.ffn_norm.data.begin(), layer.ffn_norm.data.end(), 0.0f);
  }
  // Decode 单步独立计数，初始化与静态权重上传均不在此范围。
  const auto decode = MiniBatch::Single(4, 2);
  const auto decode_expected = ForwardBatch(cpu_ctx, cpu_model, decode);
  ResetModelCudaRuntimeStats(cuda_model);
  RequireCloseTensor(ForwardBatch(cuda_ctx, cuda_model, decode), decode_expected,
                     1e-4f, 1e-4f, "decode logits");
  for (int l = 0; l < cpu_model.config.n_layers; ++l) {
    for (int p = 0; p <= 2; ++p) {
      Tensor k({2, 4}), v(k.shape);
      std::copy_n(cpu_ctx.kv_cache.KeyPtr(l, p, 0), 8, k.data.begin());
      std::copy_n(cpu_ctx.kv_cache.valuePtr(l, p, 0), 8, v.data.begin());
      RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadKey(l, p), k, 1e-4f);
      RequireCloseTensor(cuda_ctx.cuda_kv_cache.ReadValue(l, p), v, 1e-4f);
    }
  }
  RequireDeviceStats(cuda_model, 1);
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

// 测试目的：上传快照存在时使用 F32 CUDA Linear，清除后恢复 CPU 前向且查询计数为零
TEST_F(CudaForwardLinearTest, ResidentWeightsAreUsedUntilSnapshotIsCleared) {
  MiniLlamaModel cpu_model = MakeModel();
  MiniLlamaModel uploaded_model = cpu_model;
  UploadModelWeightsToCuda(uploaded_model);
  MiniLlamaContext cpu_ctx(&cpu_model), uploaded_ctx(&uploaded_model);
  ResetModelCudaRuntimeStats(uploaded_model);
  const auto batch = MiniBatch::Single(1, 0);
  const auto expected = ForwardBatch(cpu_ctx, cpu_model, batch);
  RequireCloseTensor(ForwardBatch(uploaded_ctx, uploaded_model, batch), expected);
  RequireDeviceStats(uploaded_model, 1);
  ClearModelCudaWeights(uploaded_model);
  ASSERT_FALSE(ModelHasCudaWeights(uploaded_model));
  MiniLlamaContext cleared_ctx(&uploaded_model);
  RequireCloseTensor(ForwardBatch(cleared_ctx, uploaded_model, batch), expected);
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

// 同一量化权重在 CPU/GPU 上对照，覆盖逐行尾块及所有 Transformer 投影。
void VerifyQuantizedForward(QuantType type) {
  SCOPED_TRACE(static_cast<int>(type));
  auto cpu_model = MakeModel(2, true);
  auto quantize = [&](QuantizedTensor& w) {
    const Tensor dense = ToTensor(w);
    w.f32_data.clear();
    w.type = type;
    if (type == QuantType::kQ80) {
      w.q8_0_data = QuantizeToQ80(dense);
    }
    if (type == QuantType::kQ40) {
      w.q4_0_data = QuantizeToQ40(dense);
    }
    if (type == QuantType::kQ41) {
      // 复用打包值，指定非零 FP16 缩放与偏移，验证 d*q+m。
      auto q40 = QuantizeToQ40(dense);
      w.q4_1_data.resize(q40.size());
      for (size_t i = 0; i < q40.size(); ++i) {
        auto& q = w.q4_1_data[i];
        q.d = 0x2800;  // FP16 0.03125
        q.m = 0xb000;  // FP16 -0.125
        std::copy_n(q40[i].qs, 16, q.qs);
      }
    }
  };
  quantize(cpu_model.lm_head);
  for (auto& l : cpu_model.layers) {
    for (auto* w : {&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down}) {
      quantize(*w);
    }
  }
  auto cuda_model = cpu_model;
  MiniLlamaContext cpu(&cpu_model), gpu(&cuda_model);
  std::vector<Tensor> expected;
  for (int pos = 0; pos < 4; ++pos) {
    expected.push_back(ForwardBatch(cpu, cpu_model, MiniBatch::Single(pos, pos)));
  }
  UploadModelWeightsToCuda(cuda_model);
  EXPECT_EQ(ModelCudaUploadedF32WeightCount(cuda_model), 0u);
  EXPECT_EQ(ModelCudaUploadedQ80WeightCount(cuda_model),
            type == QuantType::kQ80 ? 15u : 0u);
  EXPECT_EQ(ModelCudaUploadedQ40WeightCount(cuda_model),
            type == QuantType::kQ40 ? 15u : 0u);
  EXPECT_EQ(ModelCudaUploadedQ41WeightCount(cuda_model),
            type == QuantType::kQ41 ? 15u : 0u);
  EXPECT_EQ(ModelCudaMemoryBytes(cuda_model), ModelWeightBytes(cpu_model));
  ResetModelCudaRuntimeStats(cuda_model);
  for (int pos = 0; pos < 4; ++pos) {
    SCOPED_TRACE(pos);
    auto batch = MiniBatch::Single(pos, pos);
    RequireCloseTensor(ForwardBatch(gpu, cuda_model, batch), expected[pos], 3e-4f,
                       3e-4f, "quantized logits");
  }
  RequireDeviceStats(cuda_model, 4);
}

using cuda_forward_quant = CudaGpuTest;
using cuda_forward_q4 = CudaGpuTest;

// 测试目的：验证 Q8_0 前向投影保持权重驻留并与 CPU 数值一致。
TEST_F(cuda_forward_quant, Q80ProjectionsRemainResident) {
  VerifyQuantizedForward(QuantType::kQ80);
}
// 测试目的：验证 Q4_0 前向投影保持权重驻留并覆盖尾块数值。
TEST_F(cuda_forward_q4, Q40ProjectionsRemainResident) {
  VerifyQuantizedForward(QuantType::kQ40);
}
// 测试目的：验证 Q4_1 前向投影保持权重驻留并覆盖带偏移量化块。
TEST_F(cuda_forward_q4, Q41ProjectionsRemainResident) {
  VerifyQuantizedForward(QuantType::kQ41);
}
}  // namespace
