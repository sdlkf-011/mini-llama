#include "mini_llama/cuda_runtime.h"

#include <cstdio>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

// SDK 头文件只在 CUDA 构建中引入，CPU 构建无需 CUDA Toolkit
#ifdef MINI_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace mini_llama {
namespace {

// CPU Stub 共用同一错误，便于 CLI 和库调用方识别未启用 CUDA 的构建
std::runtime_error CudaNotBuiltError() {
  return std::runtime_error(
      "CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
      "on a NVIDIA CUDA machine.");
}

#ifdef MINI_LLAMA_USE_CUDA
// 将 CUDA 返回码转换为异常，保留调用的 API 名称和底层错误原因
void CheckCuda(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA runtime error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

// 在实现层转换拷贝方向，避免公共接口依赖 SDK 枚举；未知值必须拒绝
cudaMemcpyKind ToCudaMemcpyKind(CudaMemcpyKind kind) {
  switch (kind) {
    case CudaMemcpyKind::kHostToDevice:
      return cudaMemcpyHostToDevice;
    case CudaMemcpyKind::kDeviceToHost:
      return cudaMemcpyDeviceToHost;
    case CudaMemcpyKind::kDeviceToDevice:
      return cudaMemcpyDeviceToDevice;
  }
  throw std::runtime_error("unknown CUDA memcpy kind");
}

// 查询实际设备数量后检查编号范围，设备查询失败时直接传播 Runtime 异常
void ValidateDeviceId(int device_id) {
  if (device_id < 0) {
    throw std::runtime_error("CUDA device ID must be non-negative");
  }
  const int count = CudaDeviceCount();
  if (device_id >= count) {
    throw std::runtime_error(
        "CUDA device ID " + std::to_string(device_id) +
        " out of range; available device count: " + std::to_string(count));
  }
}

// 当前设备属于线程状态，不能让 Buffer 操作改变调用方的设备选择
class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) {
    CheckCuda(cudaGetDevice(&previous_), "cudaGetDevice");
    if (previous_ != device_id) {
      CudaSetDevice(device_id);
      changed_ = true;
    }
  }
  ~DeviceGuard() noexcept {
    // 恢复失败仅记录诊断，避免在异常收尾期间再次抛异常
    if (changed_) {
      const auto status = cudaSetDevice(previous_);
      if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA device restore failed: cudaSetDevice: %s\n",
                     cudaGetErrorString(status));
      }
    }
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
  bool changed_ = false;
};
#endif

// CUDA 版本编码为 major * 1000 + minor * 10，如 12040 对应 12.4
// 同时保留原始编码，便于诊断驱动支持的 API 版本与 Runtime 版本
std::string FormatCudaVersion(int version) {
  if (version <= 0) {
    return "unavailable";
  }
  return std::to_string(version / 1000) + "." +
         std::to_string((version % 1000) / 10) + " (" + std::to_string(version) +
         ")";
}

}  // namespace

// 仅反映编译配置，不触发驱动查询或设备初始化
bool CudaRuntimeBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

// 区分无设备、未启用 CUDA 和 API 查询失败，不能把查询错误当作零设备
int CudaDeviceCount() {
#ifdef MINI_LLAMA_USE_CUDA
  int count = 0;
  const auto status = cudaGetDeviceCount(&count);
  // 无设备是正常查询结果；驱动或初始化失败仍保留 API 的具体错误
  if (status == cudaErrorNoDevice) {
    return 0;
  }
  CheckCuda(status, "cudaGetDeviceCount");
  return count;
#else
  throw CudaNotBuiltError();
#endif
}

// 校验设备编号后读取属性，将 SDK 数据转换为公共设备信息结构
CudaDeviceInfo CudaGetDeviceInfo(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
  ValidateDeviceId(device_id);
  cudaDeviceProp properties{};
  CheckCuda(cudaGetDeviceProperties(&properties, device_id),
            "cudaGetDeviceProperties");
  CudaDeviceInfo info;
  info.id = device_id;
  info.name = properties.name;
  info.compute_major = properties.major;
  info.compute_minor = properties.minor;
  info.total_memory_bytes = properties.totalGlobalMem;
  CheckCuda(cudaDriverGetVersion(&info.driver_version), "cudaDriverGetVersion");
  CheckCuda(cudaRuntimeGetVersion(&info.runtime_version), "cudaRuntimeGetVersion");
  return info;
#else
  (void)device_id;
  throw CudaNotBuiltError();
#endif
}

// 显式改变调用线程的当前设备，供原始显存接口和后续推理模块使用
void CudaSetDevice(int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
  ValidateDeviceId(device_id);
  CheckCuda(cudaSetDevice(device_id), "cudaSetDevice");
#else
  (void)device_id;
  throw CudaNotBuiltError();
#endif
}

// 只格式化已有数据，不访问 GPU；显存同时显示字节数和 GiB
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info) {
  std::ostringstream out;
  out << "GPU " << info.id << ": " << info.name
      << "; Compute Capability: " << info.compute_major << '.' << info.compute_minor
      << "; memory bytes: " << info.total_memory_bytes
      << "; memory GiB: " << std::fixed << std::setprecision(3)
      << static_cast<double>(info.total_memory_bytes) / (1024.0 * 1024.0 * 1024.0)
      << "; Driver Version (CUDA API): " << FormatCudaVersion(info.driver_version)
      << "; CUDA Runtime Version: " << FormatCudaVersion(info.runtime_version);
  return out.str();
}

// 在调用线程的当前设备上分配显存，成功后由调用方负责释放
void* CudaMallocBytes(size_t bytes) {
  // 零长度不涉及 CUDA 操作，CPU 构建同样遵守零字节接口约定
  if (bytes == 0) {
    return nullptr;
  }
#ifdef MINI_LLAMA_USE_CUDA
  void* ptr = nullptr;
  CheckCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
#else
  throw CudaNotBuiltError();
#endif
}

// 空指针释放是无操作；非空指针的释放失败必须向调用方报告
void CudaFreeBytes(void* ptr) {
  if (ptr == nullptr) {
    return;
  }
#ifdef MINI_LLAMA_USE_CUDA
  CheckCuda(cudaFree(ptr), "cudaFree");
#else
  throw CudaNotBuiltError();
#endif
}

// 零长度不访问指针；非零拷贝检查指针和方向后调用 CUDA 同步拷贝接口
void CudaMemcpyBytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind) {
  if (bytes == 0) {
    return;
  }
#ifdef MINI_LLAMA_USE_CUDA
  if (dst == nullptr || src == nullptr) {
    throw std::runtime_error(
        "cudaMemcpy requires non-null src and dst for non-empty copies");
  }
  CheckCuda(cudaMemcpy(dst, src, bytes, ToCudaMemcpyKind(kind)), "cudaMemcpy");
#else
  (void)dst;
  (void)src;
  (void)kind;
  throw CudaNotBuiltError();
#endif
}

// 构造复用 Reset 的分配逻辑，失败时尚未取得显存所有权
CudaDeviceBuffer::CudaDeviceBuffer(size_t bytes, int device_id) {
  Reset(bytes, device_id);
}

// 析构使用非抛异常的释放入口，保证异常收尾期间的安全
CudaDeviceBuffer::~CudaDeviceBuffer() { ReleaseNoexcept(); }

// 转移指针、容量及所属设备，并清空源对象，防止重复释放
CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_), device_id_(other.device_id_) {
  other.data_ = nullptr;
  other.bytes_ = 0;
  other.device_id_ = 0;
}

// 自移动保持原状态；普通移动先释放目标资源，再接管源对象所有权
CudaDeviceBuffer& CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  ReleaseNoexcept();
  data_ = other.data_;
  bytes_ = other.bytes_;
  device_id_ = other.device_id_;
  other.data_ = nullptr;
  other.bytes_ = 0;
  other.device_id_ = 0;
  return *this;
}

// 在显存所属设备上释放；只有成功后才清空状态，失败时保留所有权
void CudaDeviceBuffer::Reset() {
  if (data_ == nullptr) {
    return;
  }
#ifdef MINI_LLAMA_USE_CUDA
  DeviceGuard guard(device_id_);
#endif
  CudaFreeBytes(data_);
  data_ = nullptr;
  bytes_ = 0;
  device_id_ = 0;
}

// 重新分配采用先分配、后替换的顺序，避免分配失败导致原数据丢失
void CudaDeviceBuffer::Reset(size_t bytes, int device_id) {
  if (bytes == 0) {
    // 零大小仅释放旧资源并记录所选设备，不创建新的 GPU 分配
    Reset();
    device_id_ = device_id;
    return;
  }
  // 先分配替换对象；分配或释放原对象失败时，替换对象会自动回收资源
  CudaDeviceBuffer replacement;
  if (!CudaRuntimeBuilt()) {
    throw CudaNotBuiltError();
  }
#ifdef MINI_LLAMA_USE_CUDA
  {
    ValidateDeviceId(device_id);
    DeviceGuard guard(device_id);
    replacement.data_ = CudaMallocBytes(bytes);
    replacement.bytes_ = bytes;
    replacement.device_id_ = device_id;
  }
#else
  (void)device_id;
#endif
  // 原资源成功释放后转移替换对象的所有权，替换对象变为空对象
  Reset();
  data_ = std::exchange(replacement.data_, nullptr);
  bytes_ = std::exchange(replacement.bytes_, 0);
  device_id_ = std::exchange(replacement.device_id_, 0);
}

// 上传前检查容量，并临时选择 Buffer 所属设备，结束后恢复原设备
void CudaDeviceBuffer::Upload(const void* src, size_t bytes) {
  if (bytes > bytes_) {
    throw std::runtime_error("CUDA upload exceeds device buffer size");
  }
  if (bytes == 0) {
    return;
  }
#ifdef MINI_LLAMA_USE_CUDA
  DeviceGuard guard(device_id_);
#endif
  CudaMemcpyBytes(data_, src, bytes, CudaMemcpyKind::kHostToDevice);
}

// 下载不改变 Buffer 所有权，容量检查和设备切换规则与上传一致
void CudaDeviceBuffer::Download(void* dst, size_t bytes) const {
  if (bytes > bytes_) {
    throw std::runtime_error("CUDA download exceeds device buffer size");
  }
  if (bytes == 0) {
    return;
  }
#ifdef MINI_LLAMA_USE_CUDA
  DeviceGuard guard(device_id_);
#endif
  CudaMemcpyBytes(dst, data_, bytes, CudaMemcpyKind::kDeviceToHost);
}

// 用于析构和 noexcept 移动赋值；释放失败记录诊断，不让异常向外传播
void CudaDeviceBuffer::ReleaseNoexcept() noexcept {
  try {
    Reset();
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "CUDA buffer release failed: %s\n", exception.what());
  } catch (...) {
    std::fprintf(stderr, "CUDA buffer release failed: unknown exception\n");
  }
}

}  // namespace mini_llama
