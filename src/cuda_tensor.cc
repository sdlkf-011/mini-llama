#include "mini_llama/cuda_tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_llama {

namespace {

// CUDA Tensor 内部校验形状，不依赖 Tensor 模块的私有辅助函数
size_t CheckedNumel(const std::vector<int>& shape) {
  size_t numel = 1;
  for (int dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("CudaTensor: dimensions must be positive");
    }
    if (numel > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error("CudaTensor: shape element count overflow");
    }
    numel *= static_cast<size_t>(dim);
  }
  return numel;
}

// 传输前核对逻辑 shape、主机元素数与设备字节容量，避免按错误长度复制
void CheckHost(const CudaTensor& device, const Tensor& host, const char* caller) {
  if (device.shape() != host.shape) {
    throw std::runtime_error(std::string(caller) +
                             ": shape mismatch GPU=" + device.ShapeStringShort() +
                             " CPU=" + host.ShapeStringShort());
  }
  if (host.size() != device.size()) {
    throw std::runtime_error(std::string(caller) + ": CPU storage size mismatch");
  }
  if (device.size() > std::numeric_limits<size_t>::max() / sizeof(float) ||
      device.bytes() != device.size() * sizeof(float)) {
    throw std::runtime_error(std::string(caller) + ": GPU storage size mismatch");
  }
}

}  // namespace

// 构造复用 Reset 的形状验证、显存分配与元数据发布流程
CudaTensor::CudaTensor(const std::vector<int>& shape, int device_id) {
  Reset(shape, device_id);
}

// 元数据与 Buffer 一起转移，避免源 Tensor 残留元素数或设备编号。
CudaTensor::CudaTensor(CudaTensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      numel_(std::exchange(other.numel_, 0)),
      device_id_(std::exchange(other.device_id_, 0)),
      buffer_(std::move(other.buffer_)) {
  other.shape_.clear();
}

// 自移动保持状态；普通移动释放目标显存后转移全部状态。
CudaTensor& CudaTensor::operator=(CudaTensor&& other) noexcept {
  if (this != &other) {
    buffer_ = std::move(other.buffer_);
    shape_ = std::move(other.shape_);
    numel_ = std::exchange(other.numel_, 0);
    device_id_ = std::exchange(other.device_id_, 0);
    other.shape_.clear();
  }
  return *this;
}

// 显式释放成功后重置元数据；释放抛异常时下面的清空语句不会执行
void CudaTensor::Reset() {
  buffer_.Reset();  // 显式释放失败时保留原来的元数据
  shape_.clear();
  numel_ = 0;
  device_id_ = 0;
}

// 先验证并复制新形状，再调用 Buffer 分配；只有分配成功后才发布新元数据
// Buffer 先分配再替换，分配失败时保留原数据与元数据
void CudaTensor::Reset(const std::vector<int>& shape, int device_id) {
  const size_t numel = shape.empty() ? 0 : CheckedNumel(shape);
  if (numel > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error("CudaTensor: byte count overflow");
  }
  if (device_id < 0) {
    throw std::runtime_error("CudaTensor: device ID must be non-negative");
  }
  auto replacement_shape = shape;  // 先复制 Shape，复制失败时不修改原显存
  buffer_.Reset(numel * sizeof(float), device_id);
  shape_.swap(replacement_shape);
  numel_ = numel;
  device_id_ = device_id;
}

// 保留设备 shape，只上传通过 CheckHost 校验的连续 F32 数据
void CudaTensor::UploadFrom(const Tensor& src) {
  CheckHost(*this, src, "CudaTensor::UploadFrom");
  buffer_.Upload(src.data.data(), bytes());
}

// 按记录的 shape 与元素数准备主机存储，再复用 DownloadTo 完成传输
Tensor CudaTensor::Download() const {
  Tensor result;
  result.shape = shape_;
  result.data.resize(numel_);
  DownloadTo(result);
  return result;
}

// 向已有主机存储下载；大小不匹配时报错，不隐式改变目标张量
void CudaTensor::DownloadTo(Tensor& dst) const {
  CheckHost(*this, dst, "CudaTensor::DownloadTo");
  buffer_.Download(dst.data.data(), bytes());
}

// 借用 Buffer 的可写设备指针，不转移显存所有权
void* CudaTensor::data() { return buffer_.data(); }

// 借用 Buffer 的只读设备指针，不触发传输
const void* CudaTensor::data() const { return buffer_.data(); }

// 复用主机 Tensor 的形状格式化，仅填充元数据，不分配或下载设备数据
std::string CudaTensor::ShapeStringShort() const {
  Tensor metadata;
  metadata.shape = shape_;
  return metadata.ShapeStringShort();
}

// 先构造匹配的设备 Tensor 再上传；返回值持有该分配的独占所有权
CudaTensor CudaTensorFromHost(const Tensor& src, int device_id) {
  CudaTensor result(src.shape, device_id);
  result.UploadFrom(src);
  return result;
}

}  // namespace mini_llama
