// 测试目的：验证设备前向缓冲区布局、KV 历史校验及缓存分配复用。
#include <gtest/gtest.h>

#include <algorithm>
#include <exception>
#include <thread>

#include "mini_llama/forward.h"
#include "names_test.h"

namespace {
using namespace mini_llama;
using namespace mini_llama::test;
using CudaForwardBufferTest = CudaGpuTest;

// Tensor 保持独占显存，生产线程退出后，移动到调用方的对象仍可正常使用。
TEST_F(CudaForwardBufferTest, TensorOwnershipSurvivesProducerThreadExit) {
  CudaTensor escaped;
  std::exception_ptr error;
  std::thread producer([&] {
    try {
      escaped = CudaTensorFromHost(Values({16}));
    } catch (...) {
      error = std::current_exception();
    }
  });
  producer.join();
  ASSERT_FALSE(error);
  RequireCloseTensor(escaped.Download(), Values({16}));
  escaped.Reset();
}

// 使用现有上下文反复重放，算子工作缓冲区与 Matmul 句柄复用不改变 logits 或 KV 语义。
TEST_F(CudaForwardBufferTest, CachedResourcesPreserveRepeatedForwardResults) {
  auto cpu = MakeModel();
  auto gpu = cpu;
  UploadModelWeightsToCuda(gpu);
  MiniLlamaContext reference(&cpu), context(&gpu);
  const auto batch = MiniBatch::FromTokens({0, 1, 2});
  const auto expected = ForwardBatch(reference, cpu, batch);
  for (int iteration = 0; iteration < 3; ++iteration) {
    context.token_history.clear();
    context.pos = 0;
    RequireCloseTensor(ForwardBatch(context, gpu, batch), expected);
    EXPECT_EQ(context.token_history, reference.token_history);
    EXPECT_EQ(context.pos, reference.pos);
  }
}

// Linear 的扁平输出直接写入分头 KV 布局，不添加 reshape 接口或主机中转。
TEST_F(CudaForwardBufferTest, FlatAndHeadShapedDeviceWritesMatch) {
  CudaKvCache cache(1, 2, 2, 4);
  auto flat = CudaTensorFromHost(Values({8}));
  auto heads = CudaTensorFromHost(Values({2, 4}));
  cache.WriteDevice(0, 0, flat, heads);
  cache.WriteDevice(0, 1, heads, flat);
  RequireCloseTensor(cache.ReadKey(0, 0), Values({2, 4}));
  RequireCloseTensor(cache.ReadValue(0, 0), Values({2, 4}));
  RequireCloseTensor(cache.ReadKey(0, 1), Values({2, 4}));
  RequireCloseTensor(cache.ReadValue(0, 1), Values({2, 4}));
  auto wrong_size = CudaTensorFromHost(Values({7}));
  auto wrong_shape = CudaTensorFromHost(Values({4, 2}));
  EXPECT_THROW(cache.WriteDevice(0, 0, flat, wrong_size), std::runtime_error);
  EXPECT_THROW(cache.WriteDevice(0, 0, wrong_shape, flat), std::runtime_error);
  RequireCloseTensor(cache.ReadKey(0, 0), Values({2, 4}));
}

// 测试目的：验证 CUDA 前向拒绝缺失、配置不匹配或跳过位置的 KV 历史；
// 合法 Decode 复用缓存分配，清空缓存后不能继续读取旧历史。
TEST_F(CudaForwardBufferTest, CacheMismatchAndLostOrSkippedHistoryAreRejected) {
  auto model = MakeModel();
  UploadModelWeightsToCuda(model);
  MiniLlamaContext ctx(&model);
  ctx.pos = 1;
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
  ctx.pos = 0;
  ctx.cuda_kv_cache.Reset(1, 4, 2, 4);
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
  ctx = MiniLlamaContext(&model);
  ForwardBatch(ctx, model, MiniBatch::Single(0, 0));
  const auto* keys = ctx.cuda_kv_cache.keys_data();
  ctx.pos = 2;
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
  ctx.pos = 1;
  ForwardToken(ctx, model, 1);
  EXPECT_EQ(ctx.cuda_kv_cache.keys_data(), keys);
  ctx.cuda_kv_cache.Reset();
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
}

// GPU 回退复用原分配，重放后与独立 CPU 上下文的 logits 和 KV 一致。
TEST_F(CudaForwardBufferTest, RetainedPrefixReplaysWithoutReallocatingGpuCache) {
  auto reference_model = MakeModel();
  auto model = reference_model;
  UploadModelWeightsToCuda(model);
  MiniLlamaContext context(&model);
  ForwardBatch(context, model, MiniBatch::FromTokens({0, 1, 2}));
  const auto* keys = context.cuda_kv_cache.keys_data();
  context.token_history.resize(1);
  context.pos = 0;

  MiniLlamaContext reference(&reference_model);
  const auto expected =
      ForwardBatch(reference, reference_model, MiniBatch::FromTokens({0, 2, 1}));
  const auto actual = ForwardBatch(context, model, MiniBatch::FromTokens({2, 1}, 1));
  RequireCloseTensor(actual, expected);
  EXPECT_EQ(context.cuda_kv_cache.keys_data(), keys);
  EXPECT_EQ(context.token_history, reference.token_history);
  const auto& c = model.config;
  for (int layer = 0; layer < c.n_layers; ++layer) {
    for (int pos = 0; pos < 3; ++pos) {
      Tensor key({c.n_kv_heads, c.head_dim});
      Tensor value({c.n_kv_heads, c.head_dim});
      std::copy_n(reference.kv_cache.KeyPtr(layer, pos, 0), key.size(),
                  key.data.begin());
      std::copy_n(reference.kv_cache.valuePtr(layer, pos, 0), value.size(),
                  value.data.begin());
      RequireCloseTensor(context.cuda_kv_cache.ReadKey(layer, pos), key);
      RequireCloseTensor(context.cuda_kv_cache.ReadValue(layer, pos), value);
    }
  }
}

// 测试目的：验证重新上传相同权重后仍可使用兼容的 GPU KV 历史；
// 清除 CUDA 权重后拒绝沿用 GPU 上下文，新建 CPU 上下文可正常前向。
TEST_F(CudaForwardBufferTest,
       ReuploadPreservesCompatibleHistoryCpuSwitchRequiresFreshContext) {
  auto model = MakeModel();
  auto reference = model;
  MiniLlamaContext cpu(&reference);
  const auto prefill = ForwardBatch(cpu, reference, MiniBatch::Single(0, 0));
  const auto decode = ForwardBatch(cpu, reference, MiniBatch::Single(0, 1));
  UploadModelWeightsToCuda(model);
  MiniLlamaContext ctx(&model);
  ResetModelCudaRuntimeStats(model);
  RequireCloseTensor(ForwardBatch(ctx, model, MiniBatch::Single(0, 0)), prefill);
  RequireDeviceStats(model, 1);
  UploadModelWeightsToCuda(model);
  // 上传同一组权重不改变模型语义，已有缓存配置仍可继续使用。
  ResetModelCudaRuntimeStats(model);
  RequireCloseTensor(ForwardBatch(ctx, model, MiniBatch::Single(0, 1)), decode);
  RequireDeviceStats(model, 1);
  ClearModelCudaWeights(model);
  EXPECT_THROW(ForwardToken(ctx, model, 0), std::runtime_error);
  MiniLlamaContext fresh_cpu(&model);
  RequireCloseTensor(ForwardBatch(fresh_cpu, model, MiniBatch::Single(0, 0)),
                     prefill);
}
}  // namespace
