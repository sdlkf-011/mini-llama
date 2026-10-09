#include "mini_llama/cuda_tensor.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace {

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

}  // namespace
