// 测试目的：验证 CUDA 运行时探测、设备信息、显存拷贝、缓冲区所有权与 CLI 行为。
#include "mini_llama/cuda_runtime.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "main_test.h"
#include "mini_llama/backend.h"

namespace {

using mini_llama::CudaDeviceBuffer;
using mini_llama::CudaDeviceCount;
using mini_llama::CudaDeviceInfo;
using mini_llama::CudaFormatDeviceInfo;
using mini_llama::CudaFreeBytes;
using mini_llama::CudaGetDeviceInfo;
using mini_llama::CudaMallocBytes;
using mini_llama::CudaMemcpyBytes;
using mini_llama::CudaMemcpyKind;
using mini_llama::CudaRuntimeBuilt;
using mini_llama::CudaSetDevice;

// 编译期验证独占所有权、无异常移动与析构，以及 data() 的 const 访问约束
static_assert(!std::is_copy_constructible<CudaDeviceBuffer>::value);
static_assert(!std::is_copy_assignable<CudaDeviceBuffer>::value);
static_assert(std::is_nothrow_move_constructible<CudaDeviceBuffer>::value);
static_assert(std::is_nothrow_move_assignable<CudaDeviceBuffer>::value);
static_assert(std::is_nothrow_destructible<CudaDeviceBuffer>::value);
static_assert(
    std::is_same<decltype(std::declval<CudaDeviceBuffer&>().data()), void*>::value);
static_assert(std::is_same<decltype(std::declval<const CudaDeviceBuffer&>().data()),
                           const void*>::value);

constexpr char kNotBuilt[] =
    "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
    "on a NVIDIA CUDA machine.";

// 共用断言：CPU Stub 必须抛出规定的 runtime_error，不能静默执行 CUDA 操作
template <typename Operation>
void ExpectNotBuilt(Operation operation) {
  try {
    operation();
    FAIL() << "CPU build must reject non-empty CUDA operations";
  } catch (const std::runtime_error& exception) {
    EXPECT_EQ(std::string(exception.what()), kNotBuilt);
  }
}

// 测试目的：验证 Runtime 构建检测与编译宏一致，后端查询正确复用 Runtime
TEST(CudaRuntimeTest, BuildFlagMatchesConfigurationAndBackendWrapper) {
#ifdef MINI_LLAMA_USE_CUDA
  EXPECT_TRUE(CudaRuntimeBuilt());
#else
  EXPECT_FALSE(CudaRuntimeBuilt());
#endif
  EXPECT_EQ(mini_llama::CudaBackendBuilt(), CudaRuntimeBuilt());
}

// 测试目的：验证 CPU 构建中的设备查询、设备选择和非零显存操作统一报错
// CUDA 构建跳过此项，分配失败后的空 Buffer 状态也在此检查
TEST(CudaRuntimeTest, CpuStubRejectsQueriesAndNonEmptyOperations) {
  if (CudaRuntimeBuilt()) {
    GTEST_SKIP() << "CPU stub test requires MINI_LLAMA_CUDA=OFF";
  }
  ExpectNotBuilt([] { CudaDeviceCount(); });
  ExpectNotBuilt([] { CudaGetDeviceInfo(0); });
  ExpectNotBuilt([] { CudaGetDeviceInfo(-1); });
  ExpectNotBuilt([] { CudaSetDevice(0); });
  ExpectNotBuilt([] { CudaSetDevice(-1); });
  ExpectNotBuilt([] { CudaMallocBytes(4); });
  int value = 42;
  ExpectNotBuilt([&] { CudaFreeBytes(&value); });
  for (const auto kind :
       {CudaMemcpyKind::kHostToDevice, CudaMemcpyKind::kDeviceToHost,
        CudaMemcpyKind::kDeviceToDevice}) {
    ExpectNotBuilt([&] { CudaMemcpyBytes(&value, &value, sizeof(value), kind); });
    ExpectNotBuilt([&] { CudaMemcpyBytes(nullptr, nullptr, sizeof(value), kind); });
  }
  ExpectNotBuilt([] { CudaDeviceBuffer buffer(4); });
  CudaDeviceBuffer buffer;
  ExpectNotBuilt([&] { buffer.Reset(4); });
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.bytes(), 0u);
  EXPECT_EQ(buffer.device_id(), 0);
  EXPECT_FALSE(mini_llama::CudaBackendAvailable());
  EXPECT_EQ(mini_llama::CudaDeviceSummary(), kNotBuilt);
}

// 测试目的：验证零长度操作无需 GPU，空 Buffer 可安全释放、移动和自移动
// 同时检查零大小 Buffer 保留所选设备编号，移动后源对象恢复默认状态
TEST(CudaRuntimeTest, ZeroLengthOperationsAndEmptyOwnershipNeedNoGpu) {
  EXPECT_EQ(CudaMallocBytes(0), nullptr);
  EXPECT_NO_THROW(CudaFreeBytes(nullptr));
  for (const auto kind :
       {CudaMemcpyKind::kHostToDevice, CudaMemcpyKind::kDeviceToHost,
        CudaMemcpyKind::kDeviceToDevice}) {
    EXPECT_NO_THROW(CudaMemcpyBytes(nullptr, nullptr, 0, kind));
  }
  CudaDeviceBuffer buffer(0);
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.bytes(), 0u);
  EXPECT_EQ(buffer.device_id(), 0);
  EXPECT_NO_THROW(buffer.Upload(nullptr, 0));
  EXPECT_NO_THROW(buffer.Download(nullptr, 0));
  EXPECT_NO_THROW(buffer.Reset());
  EXPECT_NO_THROW(buffer.Reset(0));
  EXPECT_THROW(buffer.Upload(nullptr, 1), std::runtime_error);
  EXPECT_THROW(buffer.Download(nullptr, 1), std::runtime_error);
  CudaDeviceBuffer moved(std::move(buffer));
  moved = std::move(moved);
  buffer = std::move(moved);
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(moved.data(), nullptr);
  CudaDeviceBuffer selected(0, 3);
  EXPECT_TRUE(selected.empty());
  EXPECT_EQ(selected.device_id(), 3);
  selected.Reset(0, 2);
  EXPECT_EQ(selected.device_id(), 2);
  CudaDeviceBuffer transferred(std::move(selected));
  EXPECT_EQ(transferred.device_id(), 2);
  EXPECT_EQ(selected.device_id(), 0);
}

// 测试目的：验证名称、Compute Capability、显存单位及 CUDA 版本编码的格式化
// 使用给定结构体数据，只测试文本转换，不代表实际 GPU 查询已经执行
TEST(CudaRuntimeTest, FormatsProvidedInfoAndDecodesCudaVersions) {
  CudaDeviceInfo info;
  info.id = 2;
  info.name = "test GPU";
  info.compute_major = 8;
  info.compute_minor = 9;
  info.total_memory_bytes = size_t{3} * 1024 * 1024 * 1024 / 2;
  info.driver_version = 12040;
  info.runtime_version = 13000;
  const std::string text = CudaFormatDeviceInfo(info);
  EXPECT_NE(text.find("GPU 2: test GPU"), std::string::npos);
  EXPECT_NE(text.find("Compute Capability: 8.9"), std::string::npos);
  EXPECT_NE(text.find("memory bytes: 1610612736"), std::string::npos);
  EXPECT_NE(text.find("memory GiB: 1.500"), std::string::npos);
  EXPECT_NE(text.find("Driver Version (CUDA API): 12.4 (12040)"), std::string::npos);
  EXPECT_NE(text.find("CUDA Runtime Version: 13.0 (13000)"), std::string::npos);
  EXPECT_NE(CudaFormatDeviceInfo(CudaDeviceInfo{}).find("Version: unavailable"),
            std::string::npos);
}

// 测试目的：验证负设备编号、非零拷贝的空指针和未知拷贝方向在 API 调用前拒绝
// 此项需要 CUDA 构建，但不要求主机存在可用 GPU
TEST(CudaRuntimeTest, CudaArgumentValidationNeedsNoDevice) {
  if (!CudaRuntimeBuilt()) {
    GTEST_SKIP() << "CUDA argument validation requires MINI_LLAMA_CUDA=ON";
  }
  int value = 0;
  EXPECT_THROW(CudaGetDeviceInfo(-1), std::runtime_error);
  EXPECT_THROW(CudaSetDevice(-1), std::runtime_error);
  EXPECT_THROW(
      CudaMemcpyBytes(nullptr, &value, sizeof(value), CudaMemcpyKind::kHostToDevice),
      std::runtime_error);
  EXPECT_THROW(
      CudaMemcpyBytes(&value, nullptr, sizeof(value), CudaMemcpyKind::kDeviceToHost),
      std::runtime_error);
  EXPECT_THROW(CudaMemcpyBytes(&value, &value, sizeof(value),
                               static_cast<CudaMemcpyKind>(99)),
               std::runtime_error);
}

// 测试目的：验证 --cuda-info 的构建状态、真实设备信息和错误输出及退出码
// CPU 构建应明确报未构建，CUDA 查询失败应保留诊断，成功时逐设备比对输出
TEST(CudaRuntimeTest, CliReportsRuntimeOrExplicitError) {
  const char* argv[] = {"mini-llama", "--cuda-info"};
  std::istringstream input;
  std::ostringstream output, error;
  const int status = mini_llama::RunMain(2, argv, input, output, error);
  if (!CudaRuntimeBuilt()) {
    EXPECT_EQ(status, 1);
    EXPECT_NE(output.str().find("CUDA backend built: no"), std::string::npos);
    EXPECT_EQ(error.str(), std::string("CUDA info failed: ") + kNotBuilt + '\n');
    return;
  }
  EXPECT_NE(output.str().find("CUDA backend built: yes"), std::string::npos);
  // 初始化失败是合法诊断错误；不能伪造设备数或隐去 API 错误
  if (status != 0) {
    EXPECT_NE(error.str().find("CUDA info failed:"), std::string::npos);
    return;
  }
  EXPECT_TRUE(error.str().empty());
  const int count = CudaDeviceCount();
  EXPECT_NE(output.str().find("CUDA GPU count: " + std::to_string(count)),
            std::string::npos);
  for (int device = 0; device < count; ++device) {
    EXPECT_NE(output.str().find(CudaFormatDeviceInfo(CudaGetDeviceInfo(device))),
              std::string::npos);
  }
}

// GPU 测试共用前置检查：未构建、零设备或设备查询失败时明确标记跳过
// 前置查询成功后的显存操作失败仍作为测试失败报告
class CudaRuntimeGpuTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaRuntimeBuilt()) {
      GTEST_SKIP() << "CUDA was not built; GPU operations were not executed";
    }
    try {
      device_count = CudaDeviceCount();
      if (device_count == 0) {
        GTEST_SKIP() << "no CUDA devices; GPU operations were not executed";
      }
      CudaGetDeviceInfo(0);
    } catch (const std::runtime_error& exception) {
      GTEST_SKIP()
          << "CUDA device query unavailable; GPU operations were not executed: "
          << exception.what();
    }
  }
  int device_count = 0;
};

// 测试目的：查询每张真实 GPU 的属性并验证设备选择，拒绝越界编号
// 同时检查后端可用性查询与 Runtime 的设备数量一致
TEST_F(CudaRuntimeGpuTest, QueriesRealDevicesAndRejectsInvalidIds) {
  for (int device = 0; device < device_count; ++device) {
    EXPECT_NO_THROW(CudaSetDevice(device));
    const auto info = CudaGetDeviceInfo(device);
    EXPECT_EQ(info.id, device);
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.compute_major, 0);
    EXPECT_GT(info.total_memory_bytes, 0u);
    EXPECT_GT(info.driver_version, 0);
    EXPECT_GT(info.runtime_version, 0);
    EXPECT_TRUE(mini_llama::CudaBackendAvailable(device));
  }
  EXPECT_NO_THROW(CudaSetDevice(0));
  EXPECT_THROW(CudaSetDevice(device_count), std::runtime_error);
  EXPECT_THROW(CudaSetDevice(std::numeric_limits<int>::max()), std::runtime_error);
  EXPECT_THROW(CudaGetDeviceInfo(device_count), std::runtime_error);
  EXPECT_THROW(CudaGetDeviceInfo(std::numeric_limits<int>::max()),
               std::runtime_error);
  EXPECT_FALSE(mini_llama::CudaBackendAvailable(device_count));
}

// CUDA 校验通过后，模型加载错误仍必须明确失败，不执行 CPU 推理。
TEST_F(CudaRuntimeGpuTest, CudaCliDoesNotFallbackOnMissingModel) {
  const char* argv[] = {"mini-llama", "generate", "--backend",
                        "cuda",       "--model",  "/missing-mini-llama-model"};
  std::istringstream input;
  std::ostringstream output, error;
  EXPECT_EQ(mini_llama::RunMain(6, argv, input, output, error), 1);
  EXPECT_FALSE(error.str().empty());
  EXPECT_TRUE(output.str().empty());
  EXPECT_EQ(error.str().find("status=ok"), std::string::npos);
}

// 测试目的：执行 Host → Device → Device → Host 拷贝并逐元素验证数据一致
// 覆盖三种拷贝方向、const 显存访问以及原始显存分配和释放接口
TEST_F(CudaRuntimeGpuTest, RawHostDeviceAndDeviceDeviceCopiesRoundTrip) {
  const std::array<int, 5> source = {1, -2, 0, 123456, -789};
  std::array<int, 5> result{};
  CudaDeviceBuffer first(sizeof(source)), second(sizeof(source));
  CudaMemcpyBytes(first.data(), source.data(), sizeof(source),
                  CudaMemcpyKind::kHostToDevice);
  const CudaDeviceBuffer& read_only = first;
  CudaMemcpyBytes(second.data(), read_only.data(), sizeof(source),
                  CudaMemcpyKind::kDeviceToDevice);
  CudaMemcpyBytes(result.data(), second.data(), sizeof(result),
                  CudaMemcpyKind::kDeviceToHost);
  EXPECT_EQ(result, source);
  void* raw = CudaMallocBytes(sizeof(int));
  ASSERT_NE(raw, nullptr);
  EXPECT_NO_THROW(CudaFreeBytes(raw));
}

// 测试目的：验证 Buffer 构造、上传下载、容量边界、空指针检查及 Reset 状态
// 重新分配应更新容量，释放后应为空，重复释放应安全返回
TEST_F(CudaRuntimeGpuTest, BufferResetAndCapacityChecks) {
  const std::array<int, 3> source = {7, 8, 9};
  std::array<int, 3> result{};
  CudaDeviceBuffer buffer(sizeof(source));
  ASSERT_NE(buffer.data(), nullptr);
  EXPECT_FALSE(buffer.empty());
  EXPECT_EQ(buffer.bytes(), sizeof(source));
  EXPECT_EQ(buffer.device_id(), 0);
  buffer.Upload(source.data(), sizeof(source));
  buffer.Download(result.data(), sizeof(result));
  EXPECT_EQ(result, source);
  EXPECT_THROW(buffer.Upload(source.data(), sizeof(source) + 1), std::runtime_error);
  EXPECT_THROW(buffer.Download(result.data(), sizeof(result) + 1),
               std::runtime_error);
  EXPECT_THROW(buffer.Upload(nullptr, 1), std::runtime_error);
  EXPECT_THROW(buffer.Download(nullptr, 1), std::runtime_error);
  buffer.Reset(sizeof(int));
  EXPECT_EQ(buffer.bytes(), sizeof(int));
  buffer.Reset();
  EXPECT_TRUE(buffer.empty());
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.bytes(), 0u);
  EXPECT_EQ(buffer.device_id(), 0);
  EXPECT_NO_THROW(buffer.Reset());
}

// 测试目的：验证移动构造和移动赋值转移显存所有权，源对象变为空且可重新使用
// 同时覆盖已有目标资源的替换、自移动及接收空对象后的资源释放
TEST_F(CudaRuntimeGpuTest, MovesTransferOwnershipAndLeaveSafeSources) {
  const int source = 42;
  int result = 0;
  CudaDeviceBuffer first(sizeof(source));
  first.Upload(&source, sizeof(source));
  void* pointer = first.data();
  CudaDeviceBuffer second(std::move(first));
  EXPECT_TRUE(first.empty());
  EXPECT_FALSE(second.empty());
  EXPECT_EQ(first.data(), nullptr);
  EXPECT_EQ(first.bytes(), 0u);
  EXPECT_EQ(first.device_id(), 0);
  EXPECT_EQ(second.data(), pointer);
  CudaDeviceBuffer third(16);
  third = std::move(second);
  EXPECT_TRUE(second.empty());
  EXPECT_FALSE(third.empty());
  EXPECT_EQ(third.data(), pointer);
  EXPECT_EQ(second.data(), nullptr);
  EXPECT_EQ(second.bytes(), 0u);
  EXPECT_EQ(second.device_id(), 0);
  third = std::move(third);
  third.Download(&result, sizeof(result));
  EXPECT_EQ(result, source);
  first.Reset(sizeof(int));   // 移动后的源对象仍可重新使用
  first = std::move(second);  // 接收空对象应释放原资源
  EXPECT_EQ(first.data(), nullptr);
}

// 测试目的：验证非法设备或分配失败时保留旧 Buffer 的指针、容量、设备和数据
// 同时覆盖异常展开中的临时 Buffer 生命周期，以及 CUDA API 错误消息格式
TEST_F(CudaRuntimeGpuTest,
       FailedResetPreservesOldDataAndUnwindingReleasesOwnership) {
  const int source = 99;
  int result = 0;
  CudaDeviceBuffer buffer(sizeof(source));
  buffer.Upload(&source, sizeof(source));
  void* pointer = buffer.data();
  EXPECT_THROW(buffer.Reset(16, device_count), std::runtime_error);
  try {
    buffer.Reset(std::numeric_limits<size_t>::max());
    FAIL() << "an impossible allocation must fail";
  } catch (const std::runtime_error& exception) {
    EXPECT_NE(
        std::string(exception.what()).find("CUDA runtime error in cudaMalloc:"),
        std::string::npos);
  }
  EXPECT_EQ(buffer.data(), pointer);
  EXPECT_EQ(buffer.bytes(), sizeof(source));
  EXPECT_EQ(buffer.device_id(), 0);
  buffer.Download(&result, sizeof(result));
  EXPECT_EQ(result, source);
  EXPECT_THROW(
      {
        CudaDeviceBuffer temporary(16);
        throw std::runtime_error("unwind");
      },
      std::runtime_error);
  CudaDeviceBuffer after_unwind(16);
  EXPECT_NE(after_unwind.data(), nullptr);
}

// 测试目的：交替操作不同设备的 Buffer，验证各自的所属设备和数据保持正确
// 此项至少需要两张 GPU，单 GPU 环境明确跳过
TEST_F(CudaRuntimeGpuTest, BuffersRetainTheirDeviceAcrossOtherDeviceOperations) {
  if (device_count < 2) {
    GTEST_SKIP() << "multi-device ownership requires at least two GPUs";
  }
  const int source = 123;
  int result = 0;
  CudaDeviceBuffer first(sizeof(source), 0), second(sizeof(source), 1);
  EXPECT_EQ(first.device_id(), 0);
  EXPECT_EQ(second.device_id(), 1);
  first.Upload(&source, sizeof(source));
  second.Upload(&source, sizeof(source));
  first.Download(&result, sizeof(result));
  EXPECT_EQ(result, source);
  result = 0;
  second.Download(&result, sizeof(result));
  EXPECT_EQ(result, source);
  first.Reset();
  second.Reset();
}

}  // namespace
