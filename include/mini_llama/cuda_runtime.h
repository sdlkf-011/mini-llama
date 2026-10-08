#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_

#include <cstddef>
#include <string>

namespace mini_llama {

// 设备属性不暴露 CUDA SDK 类型；版本采用 CUDA 的整数编码
struct CudaDeviceInfo {
  int id = 0;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  size_t total_memory_bytes = 0;
  int driver_version = 0;  // 驱动支持的 CUDA API 版本，不是 NVIDIA 驱动发行号
  int runtime_version = 0;  // CUDA Runtime 版本，如 12040 表示 12.4
};

enum class CudaMemcpyKind {
  kHostToDevice,
  kDeviceToHost,
  kDeviceToDevice,
};

// 仅检测编译配置，不查询驱动或 GPU
bool CudaRuntimeBuilt();

// 无设备时返回 0；未构建 CUDA 或 API 查询失败时抛出 runtime_error
int CudaDeviceCount();

// 选择调用线程当前使用的 CUDA 设备；编号非法、未构建或 API 失败时抛异常
void CudaSetDevice(int device_id);
CudaDeviceInfo CudaGetDeviceInfo(int device_id);

// 纯格式化接口，在 CPU 构建中也可使用
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);

// 原始接口操作调用线程的当前设备，分配结果须交给 CudaFreeBytes 释放
// 零字节分配、拷贝及空指针释放不执行 CUDA 操作，在两种构建中均安全
void* CudaMallocBytes(size_t bytes);
void CudaFreeBytes(void* ptr);
void CudaMemcpyBytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind);

// 独占显存所有权，操作时选择所属设备并恢复原设备；析构不抛异常
class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;
  explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
  ~CudaDeviceBuffer();

  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

  CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;
  CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

  // 显式释放失败时抛异常并保留所有权；重新分配失败时保留原 Buffer
  void Reset();
  void Reset(size_t bytes, int device_id = 0);
  void Upload(const void* src, size_t bytes);
  void Download(void* dst, size_t bytes) const;

  void* data() { return data_; }

  const void* data() const { return data_; }

  size_t bytes() const { return bytes_; }

  int device_id() const { return device_id_; }

  bool empty() const { return data_ == nullptr; }

 private:
  void ReleaseNoexcept() noexcept;

  void* data_ = nullptr;
  size_t bytes_ = 0;
  int device_id_ = 0;
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
