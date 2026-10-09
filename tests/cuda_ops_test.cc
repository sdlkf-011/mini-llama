#include "mini_llama/cuda_ops.h"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "mini_llama/ops.h"

namespace {
using namespace mini_llama;

// 数值对照同时校验形状和有限值；此处小幅输入使用固定绝对容差
void Close(const Tensor& actual, const Tensor& expected) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.data[i]));
    EXPECT_NEAR(actual.data[i], expected.data[i], 2e-5f);
  }
}

// 测试目的：CPU 构建中的全部 Ops 主机/设备入口均抛异常，不访问无效设备指针
// 这是 Stub 契约验证，不代表这些算子已在 GPU 执行
TEST(CudaOpsStub, RejectsAllHostAndDeviceEntrypoints) {
  if (CudaOpsBuilt()) GTEST_SKIP() << "requires CPU-only build";
  Tensor x({16}, 0.2f), q({4, 4}, 0.2f), k({2, 4}, 0.3f);
  CudaTensor device;
  EXPECT_THROW(CudaRmsNorm(x, x, 1e-5f), std::runtime_error);
  EXPECT_THROW(CudaSilu(x), std::runtime_error);
  EXPECT_THROW(CudaElementwiseMul(x, x), std::runtime_error);
  EXPECT_THROW(CudaElementwiseAdd(x, x), std::runtime_error);
  EXPECT_THROW(CudaSoftmax(x), std::runtime_error);
  EXPECT_THROW(CudaEmbeddingLookupDeviceWeight(nullptr, {7, 16}, 0),
               std::runtime_error);
  EXPECT_THROW(CudaRmsNormDeviceInput(device, x, 1e-5f), std::runtime_error);
  EXPECT_THROW(CudaRmsNormDeviceWeight(device, nullptr, {16}, 1e-5f),
               std::runtime_error);
  EXPECT_THROW(CudaSiluDeviceInput(device), std::runtime_error);
  EXPECT_THROW(CudaElementwiseMulDeviceInput(device, device), std::runtime_error);
  EXPECT_THROW(CudaElementwiseAddDeviceInput(device, device), std::runtime_error);
  EXPECT_THROW(CudaElementwiseAddDeviceWeight(device, nullptr, {16}),
               std::runtime_error);
  EXPECT_THROW(CudaRope(q, k, 0, 10000), std::runtime_error);
  EXPECT_THROW(CudaRopeDeviceInput(device, device, 4, 2, 4, 0, 10000),
               std::runtime_error);
}

// 仅在未构建或设备查询不可用时跳过，实际 Kernel/传输失败不能被当成跳过
class CudaOpsGpu : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaOpsBuilt()) GTEST_SKIP() << "CUDA was not built";
    try {
      if (CudaDeviceCount() == 0) GTEST_SKIP() << "no CUDA device";
      CudaGetDeviceInfo(0);
    } catch (const std::exception& e) {
      GTEST_SKIP() << "GPU query unavailable: " << e.what();
    }
  }
};

// 测试目的：对照 RMSNorm、SiLU、加乘的主机/设备入口和主机 Softmax 与 CPU 结果
// 验证 RoPE 主机及设备原地更新，并覆盖 Normal/NeoX 配对和不同的 Q/K 头数
TEST_F(CudaOpsGpu, HostAndDeviceOpsMatchCpu) {
  Tensor x({35}), weight({35}, 1.0f);
  for (size_t i = 0; i < x.size(); ++i) x.data[i] = std::sin(0.3f * i);
  Close(CudaRmsNorm(x, weight, 1e-5f), RmsNorm(x, weight, 1e-5f));
  Close(CudaSilu(x), Silu(x));
  Close(CudaSoftmax(x), Softmax(x));
  Close(CudaElementwiseMul(x, weight), ElementwiseMul(x, weight));
  Tensor doubled = x;
  for (auto& v : doubled.data) v *= 2;
  Close(CudaElementwiseAdd(x, x), doubled);
  auto dx = CudaTensorFromHost(x);
  auto dw = CudaTensorFromHost(weight);
  Close(CudaRmsNormDeviceWeight(dx, dw.data(), {35}, 1e-5f).Download(),
        RmsNorm(x, weight, 1e-5f));
  Close(CudaRmsNormDeviceInput(dx, weight, 1e-5f).Download(),
        RmsNorm(x, weight, 1e-5f));
  Close(CudaSiluDeviceInput(dx).Download(), Silu(x));
  Close(CudaElementwiseMulDeviceInput(dx, dw).Download(), x);
  Close(CudaElementwiseAddDeviceInput(dx, dx).Download(), doubled);
  Close(CudaElementwiseAddDeviceWeight(dx, dx.data(), dx.shape()).Download(),
        doubled);
  for (RopeType type : {RopeType::kNormal, RopeType::kNeoX}) {
    Tensor q({4, 4}, 0.2f), k({2, 4}, 0.3f), eq = q, ek = k;
    Rope(eq, ek, 3, 10000, type);
    auto dq = CudaTensorFromHost(q), dk = CudaTensorFromHost(k);
    CudaRopeDeviceInput(dq, dk, 4, 2, 4, 3, 10000, type);
    Close(dq.Download(), eq);
    Close(dk.Download(), ek);
    CudaRope(q, k, 3, 10000, type);
    Close(q, eq);
    Close(k, ek);
  }
}

}  // namespace
