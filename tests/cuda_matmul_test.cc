#include "mini_llama/cuda_matmul.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "mini_llama/ops.h"

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

// 现有 CPU Linear 只支持向量或单行，逐行复用它生成批量 Golden。
Tensor CpuLinearReference(const Tensor& x, const Tensor& weight) {
  if (x.num_dims() == 1) return Linear(x, weight);
  Tensor result({x.shape[0], weight.shape[0]});
  for (int row = 0; row < x.shape[0]; ++row) {
    Tensor input({x.shape[1]});
    std::copy_n(x.RowPtr(row), input.size(), input.data.begin());
    const Tensor output = Linear(input, weight);
    std::copy(output.data.begin(), output.data.end(), result.RowPtr(row));
  }
  return result;
}

using cuda_matmul = CudaGpuTest;
using cuda_linear = CudaGpuTest;

// 验证构建状态与 CUDA Runtime 一致；CPU 构建下四个计算入口均明确抛出异常
// 此用例不需要 GPU，防止未构建 CUDA 时静默执行或假装计算成功
TEST(CudaMatmulBuildTest, CpuStubRejectsAllComputeEntrypoints) {
  EXPECT_EQ(CudaMatmulBuilt(), CudaRuntimeBuilt());
  if (CudaMatmulBuilt()) GTEST_SKIP() << "CPU stub requires CUDA=OFF";
  Tensor x({3}), w({4, 3});
  EXPECT_THROW(CudaMatmul(w, Tensor({3, 2})), std::runtime_error);
  EXPECT_THROW(CudaLinear(x, w), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceWeight(x, nullptr, w.shape), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceInput(CudaTensor(), nullptr, w.shape),
               std::runtime_error);
}

// 使用 M、K、N 各不相同的非方阵，与 CPU Matmul 比较输出形状和逐元素数值
// 非对称尺寸可暴露 cuBLAS 指针顺序、转置标志或 leading dimension 配置错误
TEST_F(cuda_matmul, NonSquareRowMajorMapping) {
  const Tensor a = Values({2, 3}, 0.4f);
  const Tensor b = Values({3, 5}, 0.9f);
  RequireCloseTensor(CudaMatmul(a, b), Matmul(a, b));
  RequireCloseTensor(CudaMatmul(Values({7, 17}), Values({17, 9}, 1.0f)),
                     Matmul(Values({7, 17}), Values({17, 9}, 1.0f)));
}

// 验证 Matmul 拒绝非二维输入、归约维不匹配、非法设备、存储不足和零维度
// Tensor 的公开 shape/data 被修改后，也必须重新校验，不能直接交给 GEMM
TEST_F(cuda_matmul, RejectsRankDimensionsStorageAndDevice) {
  Tensor a({2, 3}), b({3, 5});
  EXPECT_THROW(CudaMatmul(Tensor({3}), b), std::runtime_error);
  EXPECT_THROW(CudaMatmul(a, Tensor({3})), std::runtime_error);
  EXPECT_THROW(CudaMatmul(a, Tensor({2, 5})), std::runtime_error);
  EXPECT_THROW(CudaMatmul(a, b, -1), std::runtime_error);
  EXPECT_THROW(CudaMatmul(a, b, CudaDeviceCount()), std::runtime_error);
  a.data.pop_back();
  EXPECT_THROW(CudaMatmul(a, b), std::runtime_error);
  a.shape[0] = 0;
  EXPECT_THROW(CudaMatmul(a, b), std::runtime_error);
}

// 验证 CPU 输入与 CPU 权重的向量/批量 Linear，结果与 CPU Golden 一致
// 同时覆盖无 bias 和有 bias，检查偏置沿 batch 维广播而非只加到第一行
TEST_F(cuda_linear, HostVectorAndBatchWithOptionalBias) {
  const Tensor w = Values({5, 3}, 0.8f);
  const Tensor bias = Values({5}, 1.3f);
  for (const auto& shape : {std::vector<int>{3}, std::vector<int>{4, 3}}) {
    const Tensor x = Values(shape, 0.2f);
    Tensor golden = CpuLinearReference(x, w);
    RequireCloseTensor(CudaLinear(x, w), golden);
    for (size_t i = 0; i < golden.size(); ++i) {
      golden.data[i] += bias.data[i % 5];
    }
    RequireCloseTensor(CudaLinear(x, w, &bias), golden);
  }
}

// 权重只在测试初始化时上传，随后复用同一设备指针验证向量、批量与 bias 结果
// 多次调用后检查权重指针及下载内容不变，验证借用权重的生命周期与只读约定
TEST_F(cuda_linear, ResidentWeightSupportsVectorBatchAndBiasWithoutTakingOwnership) {
  const Tensor w = Values({5, 3}, 0.8f);
  const Tensor bias = Values({5}, 1.3f);
  CudaDeviceBuffer w_dev(w.size() * sizeof(float));
  w_dev.Upload(w.data.data(), w_dev.bytes());
  const void* pointer = w_dev.data();
  for (const auto& shape : {std::vector<int>{3}, std::vector<int>{4, 3}}) {
    const Tensor x = Values(shape, 0.2f);
    Tensor golden = CpuLinearReference(x, w);
    RequireCloseTensor(CudaLinearDeviceWeight(x, pointer, w.shape), golden);
    for (size_t i = 0; i < golden.size(); ++i) {
      golden.data[i] += bias.data[i % 5];
    }
    RequireCloseTensor(CudaLinearDeviceWeight(x, pointer, w.shape, &bias), golden);
    EXPECT_EQ(w_dev.data(), pointer);
  }
  Tensor downloaded(w.shape);
  w_dev.Download(downloaded.data.data(), w_dev.bytes());
  RequireCloseTensor(downloaded, w, 0.0f);
}

// 将第一次 GPU Linear 的输出直接作为第二次输入，覆盖向量和批量的设备投影串联
// 检查输出形状、设备编号、两次投影的数值及原输入完整性；仅在测试边界下载结果
TEST_F(cuda_linear, DeviceInputOutputCanChainProjections) {
  const Tensor w1 = Values({5, 3}, 0.8f);
  const Tensor w2 = Values({2, 5}, 1.8f);
  auto dev_w1 = CudaTensorFromHost(w1);
  auto dev_w2 = CudaTensorFromHost(w2);
  for (const auto& shape : {std::vector<int>{3}, std::vector<int>{4, 3}}) {
    const Tensor x = Values(shape, 0.2f);
    auto dev_x = CudaTensorFromHost(x);
    const void* original_input = dev_x.data();
    auto y = CudaLinearDeviceInput(dev_x, dev_w1.data(), w1.shape);
    EXPECT_EQ(y.device_id(), 0);
    EXPECT_EQ(y.shape(), CpuLinearReference(x, w1).shape);
    auto z = CudaLinearDeviceInput(y, dev_w2.data(), w2.shape);
    // 仅在测试边界下载，两个投影之间只传递设备 Tensor。
    RequireCloseTensor(y.Download(), CpuLinearReference(x, w1));
    RequireCloseTensor(z.Download(),
                       CpuLinearReference(CpuLinearReference(x, w1), w2));
    EXPECT_EQ(dev_x.data(), original_input);
    RequireCloseTensor(dev_x.Download(), x, 0.0f);
  }
}

// 覆盖三种 Linear 的非法输入秩、权重形状、特征维、bias、空指针和主机存储
// 单独核对输入设备不匹配的错误来源，并确认异常之后已有设备资源仍可用于计算
TEST_F(cuda_linear, RejectsInvalidShapesBiasPointersAndStorage) {
  Tensor x({3}), w({5, 3});
  auto dev_w = CudaTensorFromHost(w);
  auto dev_x = CudaTensorFromHost(x);
  Tensor bad_bias({1, 5});
  EXPECT_THROW(CudaLinear(x, w, &bad_bias), std::runtime_error);
  bad_bias = Tensor({4});
  EXPECT_THROW(CudaLinearDeviceWeight(x, dev_w.data(), w.shape, &bad_bias),
               std::runtime_error);
  bad_bias = Tensor({5});
  bad_bias.data.pop_back();
  EXPECT_THROW(CudaLinear(x, w, &bad_bias), std::runtime_error);
  EXPECT_THROW(CudaLinear(Tensor({1, 1, 3}), w), std::runtime_error);
  EXPECT_THROW(CudaLinear(x, Tensor({5})), std::runtime_error);
  EXPECT_THROW(CudaLinear(x, Tensor({5, 4})), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceWeight(x, nullptr, w.shape), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceWeight(x, dev_w.data(), {5}), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceWeight(x, dev_w.data(), {0, 3}), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceWeight(x, dev_w.data(), {5, 4}), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceInput(dev_x, nullptr, w.shape), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceInput(dev_x, dev_w.data(), {5, 4}),
               std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceInput(dev_x, dev_w.data(), {5}), std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceInput(CudaTensor(), dev_w.data(), w.shape),
               std::runtime_error);
  EXPECT_THROW(CudaLinearDeviceInput(dev_x, dev_w.data(), w.shape, 1),
               std::runtime_error);
  // 区分设备不匹配与无效设备编号，单 GPU 环境也能验证此检查。
  try {
    CudaLinearDeviceInput(dev_x, dev_w.data(), w.shape, 1);
    FAIL() << "expected device mismatch";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("different CUDA device"),
              std::string::npos);
  }
  w.data.pop_back();
  EXPECT_THROW(CudaLinear(x, w), std::runtime_error);
  x.data.pop_back();
  EXPECT_THROW(CudaLinearDeviceWeight(x, dev_w.data(), {5, 3}), std::runtime_error);
  // 异常后资源仍有效，可继续执行正常计算。
  RequireCloseTensor(CudaLinearDeviceInput(dev_x, dev_w.data(), {5, 3}).Download(),
                     Tensor({5}));
}
}  // namespace

int main(int argc, char** argv) {
  // 支持文档中的 `... matmul` / `... linear`，同时保留标准 GTest 参数。
  std::string filter;
  if (argc > 1 &&
      (std::string(argv[1]) == "matmul" || std::string(argv[1]) == "linear")) {
    filter = "--gtest_filter=cuda_" + std::string(argv[1]) + ".*";
    argv[1] = filter.data();
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
