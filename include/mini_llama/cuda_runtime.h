#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_

#include <cstddef>
#include <string>

namespace mini_llama {

// 设备属性不暴露 CUDA SDK 类型；版本采用 CUDA 的整数编码
struct CudaDeviceInfo {
  int id = 0;                     // 当前进程可见的设备编号
  std::string name;               // Runtime 返回的设备名称
  int compute_major = 0;          // Compute Capability 主版本
  int compute_minor = 0;          // Compute Capability 次版本
  size_t total_memory_bytes = 0;  // 设备总显存字节数
  int driver_version = 0;  // 驱动支持的 CUDA API 版本，不是 NVIDIA 驱动发行号
  int runtime_version = 0;  // CUDA Runtime 版本，如 12040 表示 12.4
};

// 显式指定主机/设备之间的传输方向，公共接口不暴露 SDK 枚举
enum class CudaMemcpyKind {
  kHostToDevice,    // 主机内存 -> 设备显存
  kDeviceToHost,    // 设备显存 -> 主机内存
  kDeviceToDevice,  // 设备显存 -> 设备显存
};

// 仅检测编译配置，不查询驱动或 GPU
bool CudaRuntimeBuilt();

// 查询可见设备数量，无设备返回 0；未构建 CUDA 或 API 查询失败时抛异常
int CudaDeviceCount();

// 选择调用线程当前使用的 CUDA 设备；编号非法、未构建或 API 失败时抛异常
void CudaSetDevice(int device_id);
// 检查可见设备编号后查询属性；CPU 构建、越界或 API 失败时抛异常
CudaDeviceInfo CudaGetDeviceInfo(int device_id);

// 纯格式化接口，在 CPU 构建中也可使用
std::string CudaFormatDeviceInfo(const CudaDeviceInfo& info);

// 原始接口操作调用线程的当前设备，分配结果须交给 CudaFreeBytes 释放
// 零字节分配、拷贝及空指针释放在 CPU/CUDA 构建中均不执行 CUDA 操作
// 在当前设备分配 bytes 字节；零字节返回 nullptr
void* CudaMallocBytes(size_t bytes);

// 释放原始显存；空指针直接返回，非空指针的释放失败会抛异常
void CudaFreeBytes(void* ptr);

// 同步复制指定字节数；非零拷贝拒绝空指针和未知方向
void CudaMemcpyBytes(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind);

// 独占显存所有权，操作时临时选择所属设备并恢复原设备；析构不抛异常
class CudaDeviceBuffer {
 public:
  CudaDeviceBuffer() = default;
  explicit CudaDeviceBuffer(size_t bytes, int device_id = 0);
  ~CudaDeviceBuffer();

  CudaDeviceBuffer(const CudaDeviceBuffer&) = delete;
  CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

  // 转移指针、容量和设备编号，源 Buffer 恢复为空的默认状态
  CudaDeviceBuffer(CudaDeviceBuffer&& other) noexcept;

  // 先尝试释放目标资源再接管源资源；自移动不改变状态
  CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other) noexcept;

  // 显式释放失败时抛异常并保留所有权
  // 非空 Buffer 释放成功后清空状态；空 Buffer 直接返回
  void Reset();

  // 先分配再替换；分配或旧资源释放失败时保留原 Buffer
  void Reset(size_t bytes, int device_id = 0);

  // 从主机上传至 Buffer 起始位置；检查容量，零长度不访问设备
  void Upload(const void* src, size_t bytes);

  // 从 Buffer 起始位置下载到主机；检查容量，零长度不访问设备
  void Download(void* dst, size_t bytes) const;

  // 借用可写设备指针；调用方不得自行释放，指针不能由 CPU 直接解引用
  void* data() { return data_; }

  // 借用只读设备指针，生命周期受当前 Buffer 所有权约束
  const void* data() const { return data_; }

  // 返回当前记录的显存容量，单位为字节
  size_t bytes() const { return bytes_; }

  // 返回 Buffer 记录的所属设备编号，不查询 Runtime
  int device_id() const { return device_id_; }

  // 按指针判断是否持有显存分配
  bool empty() const { return data_ == nullptr; }

 private:
  void ReleaseNoexcept() noexcept;

  void* data_ = nullptr;  // 当前独占的原始设备指针
  size_t bytes_ = 0;      // 当前分配容量
  int device_id_ = 0;     // 释放与拷贝时选择的所属设备
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_RUNTIME_H_
