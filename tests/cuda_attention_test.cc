// 测试目的：对照独立 CPU 参考结果验证 CUDA GQA Attention，并检查非法输入和 CPU
// stub。
#include "mini_llama/cuda_attention.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mini_llama/kv_cache.h"
#include "names_test.h"

namespace {
using namespace mini_llama;

// 仅设备探测失败时跳过；探测成功后的分配、启动或数值异常必须使测试失败。
class CudaAttentionGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaRuntimeBuilt()) {
      GTEST_SKIP() << "CUDA not built";
    }
    try {
      if (CudaDeviceCount() == 0) {
        GTEST_SKIP() << "No CUDA device";
      }
      CudaGetDeviceInfo(0);
    } catch (const std::runtime_error& e) {
      GTEST_SKIP() << "CUDA device query unavailable: " << e.what();
    }
  }
};

// 测试目的：验证构建状态与 Runtime 一致，无 CUDA
// 时主机/设备入口均明确抛出未构建异常。
TEST(CudaAttentionTest, BuiltAndStub) {
  EXPECT_EQ(CudaAttentionBuilt(), CudaRuntimeBuilt());
  if (CudaRuntimeBuilt()) {
    GTEST_SKIP() << "CPU stub only";
  }
  Tensor q({2, 4});
  CudaKvCache cache;
  CudaTensor device_q;
  EXPECT_THROW(CudaAttentionDecode(q, cache, 0, 0, 2, 1, 4), std::runtime_error);
  EXPECT_THROW(CudaAttentionDecodeDeviceInput(device_q, cache, 0, 0, 2, 1, 4),
               std::runtime_error);
}

// 测试目的：用独立 CPU
// 参考对照两种入口，覆盖多层、GQA、因果端点及超过线程块宽度的头维度。
TEST_F(CudaAttentionGpuTest, HostAndDeviceGqaMatchIndependentReference) {
  for (int dim : {4, 130}) {
    const int heads = 4, kv_heads = 2;
    CudaKvCache cache(2, 4, kv_heads, dim);
    KvCache cpu(2, 4, kv_heads, dim);
    Tensor q({heads, dim});
    for (size_t i = 0; i < q.size(); ++i) {
      q.data[i] = std::sin(0.17f * i);
    }
    for (int l = 0; l < 2; ++l) {
      for (int t = 0; t < 4; ++t) {
        Tensor k({kv_heads, dim}), v(k.shape);
        for (size_t i = 0; i < k.size(); ++i) {
          k.data[i] = std::cos(0.13f * i + t + l);
          v.data[i] = std::sin(0.21f * i + 2 * t + l);
        }
        cache.Write(l, t, k, v);
        cpu.Write(l, t, k, v);
      }
    }
    auto dq = CudaTensorFromHost(q);
    for (int layer = 0; layer < 2; ++layer) {
      for (int pos : {0, 2, 3}) {
        // 使用独立的双精度分数与归一化计算，避免直接复用被测 CUDA 实现。
        Tensor expected(q.shape);
        for (int h = 0; h < heads; ++h) {
          std::vector<double> scores(pos + 1);
          for (int t = 0; t <= pos; ++t) {
            const float* k = cpu.KeyPtr(layer, t, h / 2);
            for (int d = 0; d < dim; ++d) {
              scores[t] += q.data[h * dim + d] * k[d];
            }
            scores[t] /= std::sqrt(static_cast<double>(dim));
          }
          double maximum = *std::max_element(scores.begin(), scores.end()), sum = 0;
          for (double& score : scores) {
            score = std::exp(score - maximum);
            sum += score;
          }
          for (int d = 0; d < dim; ++d) {
            for (int t = 0; t <= pos; ++t) {
              expected.data[h * dim + d] +=
                  scores[t] / sum * cpu.valuePtr(layer, t, h / 2)[d];
            }
          }
        }
        auto host = CudaAttentionDecode(q, cache, layer, pos, heads, kv_heads, dim);
        auto device = CudaAttentionDecodeDeviceInput(dq, cache, layer, pos, heads,
                                                     kv_heads, dim)
                          .Download();
        EXPECT_EQ(host.shape, q.shape);
        EXPECT_EQ(device.shape, (std::vector<int>{heads * dim}));
        SCOPED_TRACE("layer=" + std::to_string(layer) + " pos=" +
                     std::to_string(pos) + " head_dim=" + std::to_string(dim));
        mini_llama::test::RequireCloseTensor(host, expected, 1e-4f, 0.0f,
                                             "attention host");
        mini_llama::test::RequireCloseTensor(
            device, expected.ReshapeChecked({heads * dim}, "attention reference"),
            1e-4f, 0.0f, "attention device");
      }
    }
  }
}

// 测试目的：验证索引、GQA
// 头数、等元素数错误形状、截断存储和设备不一致在执行前被拒绝。
TEST_F(CudaAttentionGpuTest, RejectsInvalidShapeStorageIndicesAndDevice) {
  CudaKvCache cache(2, 3, 2, 4);
  Tensor q({4, 4});
  auto dq = CudaTensorFromHost(q);
  EXPECT_THROW(CudaAttentionDecode(q, cache, -1, 0, 4, 2, 4), std::out_of_range);
  EXPECT_THROW(CudaAttentionDecode(q, cache, 0, 3, 4, 2, 4), std::out_of_range);
  EXPECT_THROW(CudaAttentionDecode(q, cache, 0, 0, 3, 2, 4), std::runtime_error);
  EXPECT_THROW(CudaAttentionDecode(q, cache, 0, 0, 4, 2, 4, 1), std::runtime_error);
  q.data.pop_back();
  EXPECT_THROW(CudaAttentionDecode(q, cache, 0, 0, 4, 2, 4), std::runtime_error);
  CudaTensor wrong({2, 8});
  EXPECT_THROW(CudaAttentionDecodeDeviceInput(wrong, cache, 0, 0, 4, 2, 4),
               std::runtime_error);
  EXPECT_THROW(CudaAttentionDecodeDeviceInput(dq, cache, 0, 0, 4, 2, 4, 1),
               std::runtime_error);
  EXPECT_THROW(CudaAttentionDecodeDeviceInput(dq, cache, 0, 0, 0, 2, 4),
               std::runtime_error);
}
}  // namespace
