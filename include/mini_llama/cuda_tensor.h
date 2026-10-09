#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 连续布局的 F32 激活值；量化权重继续保留原有块格式
class CudaTensor {
 public:
  CudaTensor() = default;
  explicit CudaTensor(const std::vector<int>& shape, int device_id = 0);

  CudaTensor(const CudaTensor&) = delete;
  CudaTensor& operator=(const CudaTensor&) = delete;

  // 转移元数据与显存，源对象恢复为可安全使用的空 Tensor
  CudaTensor(CudaTensor&& other) noexcept;
  CudaTensor& operator=(CudaTensor&& other) noexcept;

  // 先释放显存，再清空 shape、元素数与设备编号；显式释放失败时保留元数据
  void Reset();

  // 校验维度、元素数和字节数溢出后重新分配，成功后更新元数据
  // 分配失败时保留原显存、shape、元素数与设备编号
  void Reset(const std::vector<int>& shape, int device_id = 0);

  // 上传已有主机 Tensor；shape、元素数与设备字节数必须一致，不自动 resize
  void UploadFrom(const Tensor& src);

  // 构造同 shape 的主机 Tensor 并同步下载全部元素
  Tensor Download() const;

  // 下载到已分配的主机 Tensor；不改变目标 shape，传输前验证存储大小
  void DownloadTo(Tensor& dst) const;

  // 返回借用的设备指针，调用方不取得释放权限
  void* data();
  // 返回借用的只读设备指针。
  const void* data() const;

  // 返回逻辑形状的只读引用，引用有效期受本对象生命周期与 Reset 约束
  const std::vector<int>& shape() const { return shape_; }

  // 返回逻辑维度数量；默认空 Tensor 的秩为零
  int num_dims() const { return static_cast<int>(shape_.size()); }

  // 返回记录的逻辑元素数，单位为 F32 元素
  size_t size() const { return numel_; }

  // 返回底层 Buffer 当前的实际字节容量
  size_t bytes() const { return buffer_.bytes(); }

  // 返回分配时记录的设备编号，不查询当前线程设备
  int device_id() const { return device_id_; }

  // 按底层 Buffer 是否持有显存判断为空
  bool empty() const { return buffer_.empty(); }

  // 仅格式化逻辑形状，不下载数据或访问 GPU
  std::string ShapeStringShort() const;

 private:
  std::vector<int> shape_;   // 当前逻辑形状
  size_t numel_ = 0;         // 已校验的逻辑元素数
  int device_id_ = 0;        // Tensor 记录的设备编号
  CudaDeviceBuffer buffer_;  // 实际显存与独占所有权
};

// 在指定设备按主机 Tensor 的 shape 分配并上传，返回独占资源的设备 Tensor
CudaTensor CudaTensorFromHost(const Tensor& src, int device_id = 0);

}  // namespace mini_llama
#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_TENSOR_H_
