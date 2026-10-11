// 测试目的：验证完整 CUDA 前向、GQA/RoPE/bias、连续解码及设备传输和算子计数。
#include <gtest/gtest.h>

#include "mini_llama/forward.h"
#include "names_test.h"

namespace {
using namespace mini_llama;
using namespace mini_llama::test;
using CudaForwardTest = CudaGpuTest;

// 测试目的：对照单步和连续 CUDA Forward 的 Logits 与 CPU
// 结果，并验证统计重置后的累加； 每次仅下载 128 个 F32 Logits（512 字节），没有 H2D
// 或 CPU Attention 回退。
TEST_F(CudaForwardTest, SingleAndMultipleTokensDownloadOnly128Logits) {
  auto cpu = MakeModel();
  cpu.config.vocab_size = 128;
  cpu.token_embedding = Values({128, cpu.config.dim}, 0.3f);
  cpu.lm_head = ToQuantizedTensor(Values({128, cpu.config.dim}, 0.19f));
  auto gpu = cpu;
  MiniLlamaContext c(&cpu);
  std::vector<Tensor> expected;
  for (int p = 0; p < 3; ++p) {
    expected.push_back(ForwardBatch(c, cpu, MiniBatch::Single(p, p)));
  }
  UploadModelWeightsToCuda(gpu);
  MiniLlamaContext g(&gpu);
  ResetModelCudaRuntimeStats(gpu);
  for (int p = 0; p < 3; ++p) {
    SCOPED_TRACE(p);
    if (p == 1) {
      ResetModelCudaRuntimeStats(gpu);
    }
    RequireCloseTensor(ForwardBatch(g, gpu, MiniBatch::Single(p, p)), expected[p],
                       2e-4f, 2e-4f, "logits");
    const size_t count = p == 2 ? 2 : 1;
    RequireDeviceStats(gpu, count);
    EXPECT_EQ(ModelCudaDeviceToHostBytes(gpu), count * 512);
  }
}

// 测试目的：验证两层、三 Token 前向的 Linear/Activation/Attention 调用数为 45/45/6；
// 三次最终 Logits 下载共 1536 字节，计数与实际模型结构和 Forward 次数一致。
TEST_F(CudaForwardTest, ThreeTokensCountersReflectOperatorStructure) {
  auto model = MakeModel();
  model.config.vocab_size = 128;
  model.token_embedding = Values({128, model.config.dim}, 0.3f);
  model.lm_head = ToQuantizedTensor(Values({128, model.config.dim}, 0.19f));
  auto cpu = model;
  MiniLlamaContext reference(&cpu);
  const auto batch = MiniBatch::FromTokens({0, 1, 2});
  const auto expected = ForwardBatch(reference, cpu, batch);
  UploadModelWeightsToCuda(model);
  MiniLlamaContext ctx(&model);
  ResetModelCudaRuntimeStats(model);
  RequireCloseTensor(ForwardBatch(ctx, model, batch), expected, 2e-4f, 2e-4f,
                     "batch logits");
  RequireDeviceStats(model, 3);
  EXPECT_EQ(ModelCudaLinearCalls(model), 45u);
  EXPECT_EQ(ModelCudaActivationCalls(model), 45u);
  EXPECT_EQ(ModelCudaAttentionCalls(model), 6u);
  EXPECT_EQ(ModelCudaDeviceToHostBytes(model), 1536u);
}

// 测试目的：验证非法 Token ID、上下文位置和头配置在计算前被拒绝；
// 失败时没有 Linear 调用、Logits 下载或 GPU KV Cache 分配。
TEST_F(CudaForwardTest, InvalidTokenPositionAndConfigDoNotExecuteOperators) {
  auto model = MakeModel();
  UploadModelWeightsToCuda(model);
  MiniLlamaContext ctx(&model);
  for (int token : {-1, model.config.vocab_size}) {
    EXPECT_THROW(ForwardToken(ctx, model, token), std::out_of_range);
  }
  for (int pos : {-1, model.config.max_seq_len}) {
    ctx.pos = pos;
    EXPECT_THROW(ForwardToken(ctx, model, 0), std::out_of_range);
  }
  ctx.pos = 0;
  model.config.n_heads = 3;
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
  EXPECT_EQ(ModelCudaLinearCalls(model), 0u);
  EXPECT_EQ(ModelCudaDeviceToHostCopies(model), 0u);
  EXPECT_TRUE(ctx.cuda_kv_cache.empty());
}

// 测试目的：逐项破坏 CUDA 权重名称或形状，并设置不支持的量化类型，验证前向明确报错；
// 错误在算子执行及缓存写入前发生，不通过 CPU Attention 回退掩盖缺陷。
TEST_F(CudaForwardTest, MissingCorruptedAndUnsupportedWeightsFailBeforeCacheWrite) {
  auto model = MakeModel(2, true);
  UploadModelWeightsToCuda(model);
  MiniLlamaContext ctx(&model);
  for (auto& w : model.cuda_weights->weights) {
    const std::string name = w.name;
    w.name = "missing";
    EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error) << name;
    w.name = name;
    const auto shape = w.shape;
    w.shape[0] += 1;
    EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error) << name;
    w.shape = shape;
  }
  model.layers[0].wk.type = static_cast<QuantType>(99);
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
  EXPECT_EQ(ModelCudaLinearCalls(model), 0u);
  EXPECT_EQ(ModelCudaKvCacheWriteBytes(model), 0u);
  EXPECT_EQ(ModelCudaAttentionCpuFallbacks(model), 0u);
  EXPECT_TRUE(ctx.cuda_kv_cache.empty());
}
using namespace mini_llama;
using namespace mini_llama::test;
using CudaForwardAttentionTest = CudaGpuTest;

// 测试目的：验证不同层数下所有投影均被分发，并与 CPU 的 GQA、NeoX RoPE
// 和可选 bias 结果一致；运行计数须随真实层数和 token 数变化
TEST_F(CudaForwardAttentionTest, CoversAllLayersGqaAndOptionalBias) {
  for (int layers : {1, 3}) {
    SCOPED_TRACE(layers);
    MiniLlamaModel cpu_model = MakeModel(layers, true);
    cpu_model.config.rope_type = RopeType::kNeoX;
    MiniLlamaModel cuda_model = cpu_model;
    MiniLlamaContext cpu_ctx(&cpu_model), cuda_ctx(&cuda_model);
    const auto batch = MiniBatch::FromTokens({0, 2, 4});
    const auto expected = ForwardBatch(cpu_ctx, cpu_model, batch);
    UploadModelWeightsToCuda(cuda_model);
    EXPECT_EQ(ModelCudaUploadedWeightCount(cuda_model), 3u + 12u * layers);
    EXPECT_EQ(ModelCudaUploadedF32WeightCount(cuda_model), 1u + 7u * layers);
    ResetModelCudaRuntimeStats(cuda_model);
    RequireCloseTensor(ForwardBatch(cuda_ctx, cuda_model, batch), expected, 1e-4f,
                       1e-4f, "GQA logits");
    RequireDeviceStats(cuda_model, 3);
  }
}

// 测试目的：验证 Normal/NeoX RoPE 与不同 GQA 头数的连续 Decode 和 KV 逻辑读取量；
// 重放前缀复用已有缓存，缺失历史的位置应报错，合法后续位置仍与 CPU 一致。
TEST_F(CudaForwardAttentionTest, HistoryReadsRespectGqaAndPrefixReplay) {
  for (RopeType rope : {RopeType::kNormal, RopeType::kNeoX}) {
    SCOPED_TRACE(static_cast<int>(rope));
    for (int kv_heads : {1, 2, 4}) {
      SCOPED_TRACE(kv_heads);
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
        SCOPED_TRACE(p);
        const auto expected = ForwardBatch(c, cpu, batch);
        RequireCloseTensor(ForwardBatch(g, gpu, batch), expected, 2e-4f, 2e-4f,
                           "decode logits");
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
      const auto replay = ForwardBatch(c, cpu, MiniBatch::Single(6, 1));
      RequireCloseTensor(ForwardBatch(g, gpu, MiniBatch::Single(6, 1)), replay,
                         2e-4f, 2e-4f, "replay logits");
      EXPECT_EQ(g.cuda_kv_cache.keys_data(), keys);
      EXPECT_EQ(g.token_history.size(), 2u);
      EXPECT_THROW(ForwardBatch(g, gpu, MiniBatch::Single(3, 3)),
                   std::runtime_error);
      const auto continuation = ForwardBatch(c, cpu, MiniBatch::Single(5, 2));
      RequireCloseTensor(ForwardBatch(g, gpu, MiniBatch::Single(5, 2)), continuation,
                         2e-4f, 2e-4f, "continuation logits");
    }
  }
}
}  // namespace
