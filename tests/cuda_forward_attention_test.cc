#include <gtest/gtest.h>

#include "cuda_forward_test_utils.h"
#include "mini_llama/forward.h"

namespace {
using namespace mini_llama;
using namespace mini_llama::test;
using CudaForwardAttentionTest = CudaGpuTest;

// 测试目的：验证不同层数下所有投影均被分发，并与 CPU 的 GQA、NeoX RoPE
// 和可选 bias 结果一致；运行计数须随真实层数和 token 数变化
TEST_F(CudaForwardAttentionTest, CoversAllLayersGqaAndOptionalBias) {
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
    RequireDeviceStats(cuda_model, 3);
  }
}

// 测试目的：验证 Normal/NeoX RoPE 与不同 GQA 头数的连续 Decode 和 KV 逻辑读取量；
// 重放前缀复用已有缓存，缺失历史的位置应报错，合法后续位置仍与 CPU 一致。
TEST_F(CudaForwardAttentionTest, HistoryReadsRespectGqaAndPrefixReplay) {
  for (RopeType rope : {RopeType::kNormal, RopeType::kNeoX}) {
    for (int kv_heads : {1, 2, 4}) {
      auto cpu = MakeModel(2, true);
      cpu.config.n_kv_heads = kv_heads;
      cpu.config.rope_type = rope;
      const int kv_dim = kv_heads * cpu.config.head_dim;
      for (auto& l : cpu.layers) {
        l.wk = ToQuantizedTensor(Values({kv_dim, cpu.config.dim}, 0.4f));
        l.wv = ToQuantizedTensor(Values({kv_dim, cpu.config.dim}, 0.8f));
        l.bk = Values({kv_dim}, 0.9f);
        l.bv = Values({kv_dim}, 1.1f);
      }
      auto gpu = cpu;
      UploadModelWeightsToCuda(gpu);
      MiniLlamaContext c(&cpu), g(&gpu);
      ResetModelCudaRuntimeStats(gpu);
      size_t positions = 0;
      for (int p = 0; p < 4; ++p) {
        const auto batch = MiniBatch::Single(p, p);
        RequireCloseTensor(ForwardBatch(g, gpu, batch), ForwardBatch(c, cpu, batch),
                           2e-4f);
        positions += p + 1;
      }
      RequireDeviceStats(gpu, 4);
      EXPECT_EQ(ModelCudaKvCacheReadBytes(gpu),
                2 * cpu.config.n_layers * cpu.config.n_heads * cpu.config.head_dim *
                    sizeof(float) * positions);
      const void* keys = g.cuda_kv_cache.keys_data();
      // 回退重放分支，旧后缀不能作为未计算位置的有效历史。
      g.token_history.resize(1);
      c.token_history.resize(1);
      RequireCloseTensor(ForwardBatch(g, gpu, MiniBatch::Single(6, 1)),
                         ForwardBatch(c, cpu, MiniBatch::Single(6, 1)), 2e-4f);
      EXPECT_EQ(g.cuda_kv_cache.keys_data(), keys);
      EXPECT_EQ(g.token_history.size(), 2u);
      EXPECT_THROW(ForwardBatch(g, gpu, MiniBatch::Single(3, 3)),
                   std::runtime_error);
      RequireCloseTensor(ForwardBatch(g, gpu, MiniBatch::Single(5, 2)),
                         ForwardBatch(c, cpu, MiniBatch::Single(5, 2)), 2e-4f);
    }
  }
}
}  // namespace
