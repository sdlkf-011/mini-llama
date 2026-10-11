#include "mini_llama/cuda_quant.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mini_llama/quant.h"

namespace {
using namespace mini_llama;

// 无 CUDA/GPU 时明确跳过；设备确认可用后的 Kernel 或拷贝异常属于测试失败
class QuantGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaQuantBuilt()) {
      GTEST_SKIP() << "CUDA was not built";
    }
    try {
      if (CudaDeviceCount() == 0) {
        GTEST_SKIP() << "no CUDA device";
      }
      CudaGetDeviceInfo(0);
    } catch (const std::exception& e) {
      GTEST_SKIP() << "GPU query unavailable: " << e.what();
    }
  }
};
using cuda_quant_q8_0 = QuantGpuTest;
using cuda_quant_q4 = QuantGpuTest;

// 确定性输入含正负非零值，便于重复验证量化解码和 dot-product 的符号与数值
Tensor Values(const std::vector<int>& shape) {
  Tensor tensor(shape);
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor.data[i] = 0.4f * std::sin(0.17f * static_cast<float>(i));
  }
  return tensor;
}

// 先检查 shape 和有限值，再用绝对/相对组合容差比较同格式 CPU 参考结果
void Close(const Tensor& actual, const Tensor& expected) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual.data[i]));
    EXPECT_NEAR(actual.data[i], expected.data[i],
                2e-5f * std::max(1.0f, std::abs(expected.data[i])))
        << i;
  }
}

// 三种数据位置复用同一份量化块作为 CPU 参考，避免把量化误差算作 CUDA 误差
// 覆盖向量/batch 与主机 bias，并下载原设备块确认常驻权重未被改写
template <typename Block, typename Cpu, typename Host, typename Resident,
          typename Device>
void Verify(const std::vector<Block>& blocks, const std::vector<int>& shape, Cpu cpu,
            Host host, Resident resident, Device device) {
  CudaDeviceBuffer weight(blocks.size() * sizeof(Block));
  weight.Upload(blocks.data(), weight.bytes());
  const void* original = weight.data();
  for (bool batched : {false, true}) {
    Tensor x =
        Values(batched ? std::vector<int>{3, shape[1]} : std::vector<int>{shape[1]});
    Tensor expected = cpu(x, blocks, shape);
    Close(host(x, blocks, shape, nullptr, 0), expected);
    Close(resident(x, weight.data(), blocks.size(), shape, nullptr, 0), expected);
    auto dx = CudaTensorFromHost(x);
    Close(device(dx, weight.data(), blocks.size(), shape, 0).Download(), expected);
    Tensor bias({shape[0]}, 0.2f);
    for (auto& value : expected.data) {
      value += 0.2f;
    }
    Close(host(x, blocks, shape, &bias, 0), expected);
    Close(resident(x, weight.data(), blocks.size(), shape, &bias, 0), expected);
  }
  EXPECT_EQ(weight.data(), original);
  std::vector<Block> returned(blocks.size());
  weight.Download(returned.data(), weight.bytes());
  EXPECT_EQ(std::memcmp(returned.data(), blocks.data(), weight.bytes()), 0);
}

// 通过真实入口拒绝空指针、块数/形状/设备不匹配及截断的输入和 bias 存储
template <typename Block, typename Host, typename Resident, typename Device>
void Reject(const std::vector<Block>& blocks, Host host, Resident resident,
            Device device) {
  const std::vector<int> shape{7, 35};
  Tensor x = Values({35});
  CudaDeviceBuffer weight(blocks.size() * sizeof(Block));
  weight.Upload(blocks.data(), weight.bytes());
  auto dx = CudaTensorFromHost(x);
  EXPECT_THROW(resident(x, nullptr, blocks.size(), shape, nullptr, 0),
               std::runtime_error);
  EXPECT_THROW(device(dx, weight.data(), blocks.size() - 1, shape, 0),
               std::runtime_error);
  EXPECT_THROW(device(dx, weight.data(), blocks.size(), {7, 34}, 0),
               std::runtime_error);
  EXPECT_THROW(device(dx, weight.data(), blocks.size(), shape, 1),
               std::runtime_error);
  auto truncated = x;
  truncated.data.pop_back();
  EXPECT_THROW(host(truncated, blocks, shape, nullptr, 0), std::runtime_error);
  Tensor bias({7});
  bias.data.pop_back();
  EXPECT_THROW(host(x, blocks, shape, &bias, 0), std::runtime_error);
}

// 跨过 grid.y 的边界，并逐行核对三种入口；输入每行不同以检测漏算或行错位。
template <typename Block, typename Cpu, typename Host, typename Resident,
          typename Device>
void VerifyLargeBatch(const std::vector<Block>& blocks, Cpu cpu, Host host,
                      Resident resident, Device device,
                      const std::vector<int>& shape = {1, 32},
                      const std::vector<int>& batch_sizes = {65535, 65536, 65537}) {
  CudaDeviceBuffer weight(blocks.size() * sizeof(Block));
  weight.Upload(blocks.data(), weight.bytes());
  for (int rows : batch_sizes) {
    SCOPED_TRACE(rows);
    Tensor x = Values({rows, shape[1]});
    Tensor expected = cpu(x, blocks, shape);
    EXPECT_NO_THROW(Close(host(x, blocks, shape, nullptr, 0), expected));
    Tensor bias({shape[0]}, 0.2f);
    Tensor biased = expected;
    for (auto& value : biased.data) {
      value += bias.data[0];
    }
    EXPECT_NO_THROW(
        Close(resident(x, weight.data(), blocks.size(), shape, &bias, 0), biased));
    auto dx = CudaTensorFromHost(x);
    EXPECT_NO_THROW(Close(
        device(dx, weight.data(), blocks.size(), shape, 0).Download(), expected));
  }
}

// 测试目的：验证 Q8_0 主机、常驻权重及设备输入入口在跨越 65535 行的大批次中与 CPU
// 一致； 覆盖多输出通道、逐行尾块和 Bias，确保跨步执行没有漏行或错位。
TEST_F(cuda_quant_q8_0, LargeBatchAllEntrypointsMatchCpu) {
  VerifyLargeBatch(QuantizeToQ80(Values({1, 32})), LinearQ80, CudaQ80Linear,
                   CudaQ80LinearDeviceWeight, CudaQ80LinearDeviceInput);
  // 任务数超过 65535 * 128，核对跨步计算后的最后一行和多输出通道。
  VerifyLargeBatch(QuantizeToQ80(Values({129, 1})), LinearQ80, CudaQ80Linear,
                   CudaQ80LinearDeviceWeight, CudaQ80LinearDeviceInput, {129, 1},
                   {65537});
}

// 测试目的：验证 Q4_0 三种 CUDA 入口的大批次输出及 Bias 与 CPU 参考一致；
// 覆盖批次网格边界、多输出通道和逐行尾块，检测漏算或行错位。
TEST_F(cuda_quant_q4, LargeBatchQ40AllEntrypointsMatchCpu) {
  VerifyLargeBatch(QuantizeToQ40(Values({1, 32})), LinearQ40, CudaQ40Linear,
                   CudaQ40LinearDeviceWeight, CudaQ40LinearDeviceInput);
  VerifyLargeBatch(QuantizeToQ40(Values({129, 1})), LinearQ40, CudaQ40Linear,
                   CudaQ40LinearDeviceWeight, CudaQ40LinearDeviceInput, {129, 1},
                   {65537});
}

// 手工 Q4_1 块同时覆盖 d/m 和高低半字节，CPU 使用同一块作为参考。
TEST_F(cuda_quant_q4, LargeBatchQ41AllEntrypointsMatchCpu) {
  BlockQ41 block{};
  block.d = 0x3000;  // FP16 0.125
  block.m = 0xb800;  // FP16 -0.5
  for (int i = 0; i < 16; ++i) {
    block.qs[i] = i | ((15 - i) << 4);
  }
  VerifyLargeBatch(std::vector<BlockQ41>{block}, LinearQ41, CudaQ41Linear,
                   CudaQ41LinearDeviceWeight, CudaQ41LinearDeviceInput);
  VerifyLargeBatch(std::vector<BlockQ41>(129, block), LinearQ41, CudaQ41Linear,
                   CudaQ41LinearDeviceWeight, CudaQ41LinearDeviceInput, {129, 1},
                   {65537});
}

// 测试目的：Q8_0 三种入口在整块与尾块、向量与 batch、可选 bias 下与 CPU 一致
TEST_F(cuda_quant_q8_0, VectorBatchTailBiasAndResidentInputMatchCpu) {
  for (int width : {16, 32, 35, 64, 86}) {
    const std::vector<int> shape{7, width};
    Verify(QuantizeToQ80(Values(shape)), shape, LinearQ80, CudaQ80Linear,
           CudaQ80LinearDeviceWeight, CudaQ80LinearDeviceInput);
  }
}

// 测试目的：Q4_0 的 4 位打包值和逐行尾块正确解码，并满足同样的数据位置/bias 契约
TEST_F(cuda_quant_q4, VectorBatchTailBiasAndResidentInputMatchCpu) {
  for (int width : {16, 32, 35, 64, 86}) {
    const std::vector<int> shape{7, width};
    Verify(QuantizeToQ40(Values(shape)), shape, LinearQ40, CudaQ40Linear,
           CudaQ40LinearDeviceWeight, CudaQ40LinearDeviceInput);
  }
}

// 测试目的：Q8_0 校验在错误元数据或存储不足时抛异常，避免 Kernel 越界读写
TEST_F(cuda_quant_q8_0, RejectsPointersBlocksShapeDeviceAndTruncatedStorage) {
  Reject(QuantizeToQ80(Values({7, 35})), CudaQ80Linear, CudaQ80LinearDeviceWeight,
         CudaQ80LinearDeviceInput);
}

// 测试目的：Q4_0 的常驻块、设备输入和主机 bias 均执行必要的元数据/长度校验
TEST_F(cuda_quant_q4, RejectsPointersBlocksShapeDeviceAndTruncatedStorage) {
  Reject(QuantizeToQ40(Values({7, 35})), CudaQ40Linear, CudaQ40LinearDeviceWeight,
         CudaQ40LinearDeviceInput);
}

// 测试目的：未启用 CUDA 时报告未构建状态，Q8_0 三个计算入口均明确抛异常
TEST(cuda_quant_q8_0_stub, AllEntrypointsRejectCpuOnlyBuild) {
  if (CudaRuntimeBuilt()) {
    GTEST_SKIP() << "requires CPU-only build";
  }
  EXPECT_FALSE(CudaQuantBuilt());
  Tensor x({32});
  CudaTensor device;
  EXPECT_THROW(CudaQ80Linear(x, std::vector<BlockQ80>(1), {1, 32}),
               std::runtime_error);
  EXPECT_THROW(CudaQ80LinearDeviceWeight(x, nullptr, 1, {1, 32}),
               std::runtime_error);
  EXPECT_THROW(CudaQ80LinearDeviceInput(device, nullptr, 1, {1, 32}),
               std::runtime_error);
}

// 测试目的：CPU 构建下 Q4_0/Q4_1 的主机、设备权重和设备输入入口均拒绝执行
// 此用例不验证 Q4_1 的 GPU 数值结果
TEST(cuda_quant_q4_stub, AllEntrypointsRejectCpuOnlyBuild) {
  if (CudaRuntimeBuilt()) {
    GTEST_SKIP() << "requires CPU-only build";
  }
  Tensor x({32});
  CudaTensor device;
  EXPECT_THROW(CudaQ40Linear(x, std::vector<BlockQ40>(1), {1, 32}),
               std::runtime_error);
  EXPECT_THROW(CudaQ40LinearDeviceWeight(x, nullptr, 1, {1, 32}),
               std::runtime_error);
  EXPECT_THROW(CudaQ40LinearDeviceInput(device, nullptr, 1, {1, 32}),
               std::runtime_error);
  EXPECT_THROW(CudaQ41Linear(x, std::vector<BlockQ41>(1), {1, 32}),
               std::runtime_error);
  EXPECT_THROW(CudaQ41LinearDeviceWeight(x, nullptr, 1, {1, 32}),
               std::runtime_error);
  EXPECT_THROW(CudaQ41LinearDeviceInput(device, nullptr, 1, {1, 32}),
               std::runtime_error);
}

}  // namespace
