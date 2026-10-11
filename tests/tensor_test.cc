// 测试目的：验证 CPU/CUDA 张量形状、存储、传输和所有权，以及数值比较器的诊断。
#include "mini_llama/tensor.h"

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "names_test.h"

namespace {

using mini_llama::Tensor;

void Check(bool condition, const std::string& message) {
  EXPECT_TRUE(condition) << message;
}

void CheckNear(float actual, float expected, float tolerance = 1e-5f) {
  EXPECT_NEAR(actual, expected, tolerance);
}

void CheckTensor(const Tensor& actual, const std::vector<int>& expected_shape,
                 const std::vector<float>& expected_data, float tolerance = 1e-5f) {
  Check(actual.shape == expected_shape, "tensor shape mismatch");
  Check(actual.data.size() == expected_data.size(), "tensor size mismatch");
  for (size_t i = 0; i < expected_data.size(); ++i) {
    CheckNear(actual.data[i], expected_data[i], tolerance);
  }
}

void ExpectThrows(const std::function<void()>& operation) {
  EXPECT_THROW(operation(), std::exception);
}

void ExpectThrowsContaining(const std::function<void()>& operation,
                            const std::string& expected_text) {
  EXPECT_THROW(
      {
        try {
          operation();
        } catch (const std::exception& error) {
          EXPECT_NE(std::string(error.what()).find(expected_text),
                    std::string::npos);
          throw;
        }
      },
      std::exception);
}

// 测试不同秩张量的构造、填充值、元素数量和便捷构造函数。
TEST(TensorTest, Construction) {
  Tensor tensor({2, 3, 4}, 7.0f);
  Check(tensor.shape == std::vector<int>({2, 3, 4}), "shape mismatch");
  Check(tensor.size() == 24 && tensor.num_elements() == 24,
        "element count mismatch");
  Check(tensor.num_dims() == 3, "rank mismatch");
  Check(tensor[0] == 7.0f && tensor[23] == 7.0f, "fill mismatch");
  CheckTensor(mini_llama::MakeTensor1D(3, 1.0f), {3}, {1.0f, 1.0f, 1.0f});
  CheckTensor(mini_llama::MakeTensor2D(2, 2, -1.0f), {2, 2},
              {-1.0f, -1.0f, -1.0f, -1.0f});
  Check(mini_llama::MakeTensor3D(1, 2, 3).num_dims() == 3,
        "3D helper rank mismatch");
  Check(mini_llama::MakeTensor4D(1, 1, 2, 2).num_dims() == 4,
        "4D helper rank mismatch");
  Tensor scalar({}, 3.0f);
  Check(scalar.size() == 1 && scalar.FlatIndex({}) == 0 && scalar.At({}) == 3.0f,
        "scalar tensor mismatch");
}

// 测试多维索引展开公式，以及 At/At1-At4 的读写一致性。
TEST(TensorTest, Indexing) {
  Tensor tensor({2, 3, 4}, 0.0f);
  Check(tensor.FlatIndex({1, 2, 3}) == 23, "flat index mismatch");
  tensor.At({1, 2, 3}) = 9.0f;
  Check(tensor.At3(1, 2, 3) == 9.0f, "At3 mismatch");

  Tensor one({3}, 0.0f);
  one.At1(1) = 2.0f;
  Check(one.At1(1) == 2.0f, "At1 mismatch");
  Tensor two({2, 3}, 0.0f);
  two.At2(1, 2) = 3.0f;
  Check(two.At2(1, 2) == 3.0f, "At2 mismatch");
  Tensor four({1, 1, 2, 2}, 0.0f);
  four.At4(0, 0, 1, 1) = 4.0f;
  Check(four.At4(0, 0, 1, 1) == 4.0f, "At4 mismatch");
}

// 测试二维行指针访问，并确认 const 访问不会改变数据。
TEST(TensorTest, RowPtr) {
  Tensor tensor({2, 3}, 0.0f);
  float* row = tensor.RowPtr(1);
  row[0] = 4.0f;
  row[2] = 6.0f;
  CheckTensor(tensor, {2, 3}, {0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 6.0f});
  const Tensor& const_tensor = tensor;
  Check(const_tensor.RowPtr(1)[2] == 6.0f, "const RowPtr mismatch");
}

// 测试非法维度、错误秩和越界索引均被明确拒绝。
TEST(TensorTest, Validation) {
  ExpectThrows([] { Tensor({0, 2}); });
  ExpectThrows([] { Tensor({-1}); });
  Tensor tensor({2, 3}, 0.0f);
  ExpectThrows([&] { tensor.FlatIndex({1}); });
  ExpectThrows([&] { tensor.FlatIndex({2, 0}); });
  ExpectThrows([&] { tensor.At1(0); });
  ExpectThrows([&] { tensor.RowPtr(2); });
  ExpectThrows([&] { tensor.At2(-1, 0); });
}

// 测试 shape 字符串、AssertShape 以及保持元素数量的 ReshapeChecked。
TEST(TensorTest, ShapeOperations) {
  Tensor tensor({2, 3}, 0.0f);
  tensor[5] = 5.0f;
  Check(tensor.ShapeStringShort() == "[2, 3]", "shape string mismatch");
  tensor.AssertShape({2, 3}, "ShapeTest");
  ExpectThrowsContaining([&] { tensor.AssertShape({3, 2}, "ShapeTest"); },
                         "ShapeTest");
  Tensor reshaped = tensor.ReshapeChecked({3, 2}, "ShapeTest");
  CheckTensor(reshaped, {3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f});
  Check(tensor.shape == std::vector<int>({2, 3}), "reshape changed source");
  ExpectThrows([&] { tensor.ReshapeChecked({4, 2}, "ShapeTest"); });
}

using mini_llama::CudaTensor;
using mini_llama::Tensor;

// 空 Tensor 的传输、reset 与设备编号记录无需 CUDA 构建或 GPU。
TEST(CudaTensorTest, EmptyTensorTransfersNeedNoGpu) {
  CudaTensor device({}, 3);
  Tensor host;
  EXPECT_TRUE(device.empty());
  EXPECT_EQ(device.size(), 0u);
  EXPECT_EQ(device.bytes(), 0u);
  EXPECT_EQ(device.device_id(), 3);
  EXPECT_NO_THROW(device.UploadFrom(host));
  EXPECT_NO_THROW(device.DownloadTo(host));
  EXPECT_TRUE(device.Download().data.empty());
  CudaTensor moved(std::move(device));
  EXPECT_EQ(moved.device_id(), 3);
  EXPECT_EQ(device.device_id(), 0);
  EXPECT_TRUE(device.shape().empty());
  EXPECT_EQ(device.size(), 0u);
  EXPECT_NO_THROW(device.UploadFrom(host));
  EXPECT_NO_THROW(device.Reset());
  EXPECT_EQ(device.device_id(), 0);
  host.shape = {1};
  host.data = {1.0f};
  EXPECT_THROW(device.UploadFrom(host), std::runtime_error);
}

// 非法维度、元素数/字节数溢出和负设备编号必须在显存分配前拒绝。
TEST(CudaTensorTest, RejectsInvalidShapesBeforeAllocation) {
  CudaTensor device;
  EXPECT_THROW(device.Reset({0}), std::runtime_error);
  EXPECT_THROW(device.Reset({-1}), std::runtime_error);
  const int large = std::numeric_limits<int>::max();
  EXPECT_THROW(device.Reset({large, large, large}), std::runtime_error);
  EXPECT_THROW(device.Reset({large, large, 2}), std::runtime_error);
  EXPECT_THROW(device.Reset({}, -1), std::runtime_error);
  EXPECT_TRUE(device.empty());
  EXPECT_TRUE(device.shape().empty());
}

// CPU stub 拒绝非空分配，失败后保留原空 Tensor 的元数据与可用性。
TEST(CudaTensorTest, CpuStubRetainsMetadataAfterFailedAllocation) {
  if (mini_llama::CudaRuntimeBuilt()) {
    GTEST_SKIP() << "CPU stub requires MINI_LLAMA_CUDA=OFF";
  }
  CudaTensor device({}, 3);
  EXPECT_THROW(device.Reset({2, 3}), std::runtime_error);
  EXPECT_TRUE(device.shape().empty());
  EXPECT_EQ(device.size(), 0u);
  EXPECT_EQ(device.bytes(), 0u);
  EXPECT_EQ(device.device_id(), 3);
  EXPECT_NO_THROW(device.UploadFrom(Tensor()));
}

class CudaTensorGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!mini_llama::CudaRuntimeBuilt()) {
      GTEST_SKIP() << "CUDA was not built; GPU operations were not executed";
    }
    try {
      if (mini_llama::CudaDeviceCount() == 0) {
        GTEST_SKIP() << "no CUDA devices; GPU operations were not executed";
      }
      mini_llama::CudaGetDeviceInfo(0);
    } catch (const std::runtime_error& error) {
      GTEST_SKIP() << "CUDA device query unavailable: " << error.what();
    }
  }
};

// 真实上传下载、主机存储检查与移动所有权；移动后的源对象仍可安全使用。
TEST_F(CudaTensorGpuTest, RoundTripAndMovesPreserveTensorState) {
  Tensor host({2, 3});
  host.data = {1.0f, -2.0f, 0.0f, 3.5f, 4.0f, -5.0f};
  auto device = mini_llama::CudaTensorFromHost(host);
  EXPECT_EQ(device.Download().data, host.data);
  Tensor malformed = host;
  malformed.data.pop_back();
  EXPECT_THROW(device.UploadFrom(malformed), std::runtime_error);
  EXPECT_THROW(device.DownloadTo(malformed), std::runtime_error);
  void* pointer = device.data();
  CudaTensor moved(std::move(device));
  EXPECT_EQ(moved.data(), pointer);
  EXPECT_EQ(moved.Download().data, host.data);
  EXPECT_TRUE(device.shape().empty());
  EXPECT_EQ(device.size(), 0u);
  EXPECT_EQ(device.bytes(), 0u);
  EXPECT_NO_THROW(device.UploadFrom(Tensor()));
  device.Reset({1});
  device = std::move(moved);
  EXPECT_EQ(device.Download().data, host.data);
  EXPECT_TRUE(moved.shape().empty());
  EXPECT_EQ(moved.size(), 0u);
  device = std::move(device);
  EXPECT_EQ(device.Download().data, host.data);
}

// 分配失败不能丢失旧显存或 shape；原 Tensor 必须仍能下载正确数据。
TEST_F(CudaTensorGpuTest, FailedResetPreservesShapeAndData) {
  Tensor host({2, 3}, 7.0f);
  auto device = mini_llama::CudaTensorFromHost(host);
  void* pointer = device.data();
  EXPECT_THROW(device.Reset({4}, mini_llama::CudaDeviceCount()), std::runtime_error);
  EXPECT_EQ(device.data(), pointer);
  EXPECT_EQ(device.shape(), host.shape);
  EXPECT_EQ(device.size(), host.size());
  EXPECT_EQ(device.Download().data, host.data);
}

using namespace mini_llama;
using mini_llama::test::RequireCloseTensor;

// 在 CPU 上验证比较器的两条接受分支及小值归一化语义。
TEST(CudaComparisonTest, AbsoluteOrNormalizedRelativeTolerance) {
  Tensor expected({3});
  expected.data = {0.001f, -100.0f, 0.0f};
  Tensor actual = expected;
  actual.data = {0.002f, -100.1f, 0.001f};
  RequireCloseTensor(actual, expected, 2e-3f, 2e-3f, "logits");
  RequireCloseTensor(Tensor({1}, 0.01f), Tensor({1}), 0.02f, 0.0f, "absolute");
  RequireCloseTensor(Tensor({1}, 100.1f), Tensor({1}, 100.0f), 0.0f, 2e-3f,
                     "relative");
  EXPECT_NONFATAL_FAILURE(RequireCloseTensor(Tensor({1}, 0.003f), Tensor({1}), 2e-3f,
                                             2e-3f, "near zero"),
                          "abs_error=");
}

// 测试目的：验证比较失败时明确报告张量位置、CPU/CUDA 数值与容差。
TEST(CudaComparisonTest, FailureIdentifiesTensorPositionValuesAndThresholds) {
  Tensor actual({2, 2}), expected({2, 2});
  actual.data[3] = 1;
  EXPECT_NONFATAL_FAILURE(
      RequireCloseTensor(actual, expected, 2e-3f, 2e-3f, "logits"),
      "logits index=3 position=[1,1] CPU expected=0 CUDA actual=1 "
      "abs_error=1 relative_error=1 abs_tol=0.002 rel_tol=0.002");
}

// 测试目的：拒绝非有限参考/设备数值及形状不匹配，避免比较器误判通过。
TEST(CudaComparisonTest, RejectsNonFiniteCpuAndCudaValuesAndWrongShape) {
  EXPECT_FATAL_FAILURE(
      RequireCloseTensor(Tensor({1}, std::numeric_limits<float>::infinity()),
                         Tensor({1}), 2e-3f, 2e-3f, "CUDA infinity"),
      "CUDA infinity");
  EXPECT_FATAL_FAILURE(
      RequireCloseTensor(Tensor({1}),
                         Tensor({1}, std::numeric_limits<float>::quiet_NaN()), 2e-3f,
                         2e-3f, "CPU NaN"),
      "CPU NaN");
  EXPECT_FATAL_FAILURE(RequireCloseTensor(Tensor({2}), Tensor({1, 2})), "tensor");
}
}  // namespace
