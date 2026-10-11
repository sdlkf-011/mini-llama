#include "mini_llama/cuda_kv_cache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"

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

// 构造非零 F32 小模型，覆盖多层 GQA、FFN 尾维度与可选 QKV 偏置。
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

using CudaKvCacheGpuTest = CudaGpuTest;

// 测试目的：验证默认空状态、容量与指针、维度及总字节溢出，以及 CPU Stub 的分配异常。
TEST(CudaKvCacheTest, EmptyValidationAndStub) {
  CudaKvCache cache;
  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.bytes(), 0u);
  EXPECT_EQ(cache.keys_data(), nullptr);
  EXPECT_EQ(cache.values_data(), nullptr);
  EXPECT_EQ(CudaKvCacheBuilt(), CudaRuntimeBuilt());
  cache.Clear();
  cache.Reset();
  EXPECT_THROW(cache.ReadKey(0, 0), std::out_of_range);
  EXPECT_THROW(cache.Reset(0, 1, 1, 1), std::runtime_error);
  EXPECT_THROW(cache.Reset(1, -1, 1, 1), std::runtime_error);
  EXPECT_THROW(cache.Reset(1, 1, 0, 1), std::runtime_error);
  EXPECT_THROW(cache.Reset(1, 1, 1, -1), std::runtime_error);
  EXPECT_THROW(cache.Reset(1, 1, 1, 1, -1), std::runtime_error);
  EXPECT_THROW(cache.Reset(INT_MAX, INT_MAX, INT_MAX, INT_MAX), std::runtime_error);
  if (!CudaRuntimeBuilt()) {
    EXPECT_THROW(cache.Reset(1, 2, 2, 4), std::runtime_error);
    EXPECT_TRUE(cache.empty());
    EXPECT_FALSE(CudaKvCacheBuilt());
  }
}

// 测试目的：交替执行主机/设备写入，逐槽逐头及整块对照 CPU
// 布局，验证清零、重新分配和释放。
TEST_F(CudaKvCacheGpuTest, LayoutHostDeviceReadAndClearMatchCpu) {
  CudaKvCache cache(2, 3, 2, 4);
  KvCache cpu(2, 3, 2, 4);
  EXPECT_FALSE(cache.empty());
  EXPECT_EQ(cache.bytes(), 2 * 2 * 3 * 8 * sizeof(float));
  EXPECT_EQ(cache.ReadKey(1, 2).data, std::vector<float>(8, 0));
  for (int l = 0; l < 2; ++l) {
    for (int p = 0; p < 3; ++p) {
      Tensor k = Values({2, 4}, 1.3f * l + p);
      Tensor v = Values({2, 4}, 2.7f * l + p);
      cpu.Write(l, p, k, v);
      if (p % 2) {
        auto dk = CudaTensorFromHost(k);
        auto dv = CudaTensorFromHost(v);
        cache.WriteDevice(l, p, dk, dv);
      } else {
        cache.Write(l, p, k, v);
      }
    }
  }
  for (int l = 0; l < 2; ++l) {
    for (int p = 0; p < 3; ++p) {
      Tensor k({2, 4}), v({2, 4});
      std::copy_n(cpu.KeyPtr(l, p, 0), 8, k.data.begin());
      std::copy_n(cpu.valuePtr(l, p, 0), 8, v.data.begin());
      RequireCloseTensor(cache.ReadKey(l, p), k);
      RequireCloseTensor(cache.ReadValue(l, p), v);
      for (int h = 0; h < 2; ++h) {
        Tensor kh({4}), vh({4});
        std::copy_n(cpu.KeyPtr(l, p, h), 4, kh.data.begin());
        std::copy_n(cpu.valuePtr(l, p, h), 4, vh.data.begin());
        RequireCloseTensor(cache.ReadKeyHead(l, p, h), kh);
        RequireCloseTensor(cache.ReadValueHead(l, p, h), vh);
      }
    }
  }
  Tensor all_keys(cpu.keys.shape), all_values(cpu.values.shape);
  CudaSetDevice(cache.device_id());
  CudaMemcpyBytes(all_keys.data.data(), cache.keys_data(), cache.bytes() / 2,
                  CudaMemcpyKind::kDeviceToHost);
  CudaMemcpyBytes(all_values.data.data(), cache.values_data(), cache.bytes() / 2,
                  CudaMemcpyKind::kDeviceToHost);
  RequireCloseTensor(all_keys, cpu.keys);
  RequireCloseTensor(all_values, cpu.values);
  cache.Clear();
  for (int l = 0; l < 2; ++l) {
    for (int p = 0; p < 3; ++p) {
      EXPECT_EQ(cache.ReadKey(l, p).data, std::vector<float>(8, 0));
      EXPECT_EQ(cache.ReadValue(l, p).data, std::vector<float>(8, 0));
    }
  }
  cache.Reset(1, 2, 1, 6);
  EXPECT_EQ(cache.ReadValue(0, 1).shape, (std::vector<int>{1, 6}));
  cache.Reset();
  EXPECT_TRUE(cache.empty());
  EXPECT_EQ(cache.n_layers(), 0);
}

// 测试目的：验证层、位置、头、形状、截断存储和设备边界，确认失败的 Reset
// 保留旧缓存。
TEST_F(CudaKvCacheGpuTest, RejectsIndicesShapesStorageAndDevices) {
  CudaKvCache cache(2, 3, 2, 4);
  Tensor valid({2, 4}, 1), wrong({8}, 1);
  EXPECT_THROW(cache.Write(-1, 0, valid, valid), std::out_of_range);
  EXPECT_THROW(cache.Write(2, 0, valid, valid), std::out_of_range);
  EXPECT_THROW(cache.Write(0, -1, valid, valid), std::out_of_range);
  EXPECT_THROW(cache.Write(0, 3, valid, valid), std::out_of_range);
  EXPECT_THROW(cache.ReadKeyHead(0, 0, -1), std::out_of_range);
  EXPECT_THROW(cache.ReadValueHead(0, 0, 2), std::out_of_range);
  EXPECT_THROW(cache.Write(0, 0, wrong, valid), std::runtime_error);
  wrong = valid;
  wrong.data.pop_back();
  EXPECT_THROW(cache.Write(0, 0, valid, wrong), std::runtime_error);
  auto dk = CudaTensorFromHost(valid);
  CudaTensor wrong_device_shape({4, 2});
  EXPECT_THROW(cache.WriteDevice(0, 0, dk, wrong_device_shape), std::runtime_error);
  EXPECT_THROW(cache.Reset(1, 1, 1, 1, CudaDeviceCount()), std::runtime_error);
  EXPECT_EQ(cache.n_layers(), 2);
  RequireCloseTensor(cache.ReadKey(0, 0), Tensor({2, 4}, 0));
  if (CudaDeviceCount() > 1) {
    auto other = CudaTensorFromHost(valid, 1);
    EXPECT_THROW(cache.WriteDevice(0, 0, dk, other), std::runtime_error);
  }
}

// 测试目的：对照 CPU 多步前向，覆盖偏置及两类 RoPE，验证延迟分配、历史
// KV、统计和快照清除保护。
TEST_F(CudaKvCacheGpuTest, ForwardLazyInitializationAndResidentHistory) {
  for (bool bias : {false, true}) {
    auto cpu = MakeModel(2, bias);
    cpu.config.rope_type = bias ? RopeType::kNeoX : RopeType::kNormal;
    auto gpu = cpu;
    MiniLlamaContext reference(&cpu), context(&gpu);
    EXPECT_TRUE(context.cuda_kv_cache.empty());
    UploadModelWeightsToCuda(gpu);
    EXPECT_TRUE(context.cuda_kv_cache.empty());
    for (int pos = 0; pos < 3; ++pos) {
      reference.pos = context.pos = pos;
      ResetModelCudaRuntimeStats(gpu);
      RequireCloseTensor(ForwardToken(context, gpu, pos),
                         ForwardToken(reference, cpu, pos), 1e-4f);
      ASSERT_FALSE(context.cuda_kv_cache.empty());
      for (int l = 0; l < 2; ++l) {
        Tensor k({2, 4}), v(k.shape);
        std::copy_n(reference.kv_cache.KeyPtr(l, pos, 0), 8, k.data.begin());
        std::copy_n(reference.kv_cache.valuePtr(l, pos, 0), 8, v.data.begin());
        RequireCloseTensor(context.cuda_kv_cache.ReadKey(l, pos), k, 1e-4f);
        RequireCloseTensor(context.cuda_kv_cache.ReadValue(l, pos), v, 1e-4f);
      }
      EXPECT_EQ(ModelCudaAttentionCalls(gpu), 2u);
      EXPECT_EQ(ModelCudaAttentionCpuFallbacks(gpu), 0u);
      EXPECT_EQ(ModelCudaHostToDeviceCopies(gpu), 0u);
      EXPECT_EQ(ModelCudaDeviceToHostCopies(gpu), 1u);
      EXPECT_EQ(gpu.cuda_weights->kv_cache_write_bytes, 2u * 2 * 8 * sizeof(float));
      EXPECT_EQ(gpu.cuda_weights->kv_cache_read_bytes,
                2u * 2 * (pos + 1) * 16 * sizeof(float));
    }
    ClearModelCudaWeights(gpu);
    EXPECT_THROW(ForwardToken(context, gpu, 0), std::runtime_error);
  }
}

// 切换后端不能将缺失的 GPU 历史当作有效缓存；新上下文从零开始。
TEST_F(CudaKvCacheGpuTest, CpuPrefixRequiresNewCudaContext) {
  auto model = MakeModel();
  MiniLlamaContext ctx(&model);
  ForwardToken(ctx, model, 0);
  UploadModelWeightsToCuda(model);
  ctx.pos = 1;
  EXPECT_THROW(ForwardToken(ctx, model, 2), std::runtime_error);
  EXPECT_TRUE(ctx.cuda_kv_cache.empty());
  EXPECT_EQ(ModelCudaHostToDeviceCopies(model), 0u);
  MiniLlamaContext fresh(&model);
  EXPECT_NO_THROW(ForwardToken(fresh, model, 0));
  EXPECT_EQ(ModelCudaHostToDeviceCopies(model), 0u);
}

}  // namespace
