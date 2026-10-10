#include "mini_llama/cuda_kv_cache.h"

#include <limits>
#include <stdexcept>
#include <utility>

#ifdef MINI_LLAMA_USE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace mini_llama {
namespace {
// 使用 size_t 除法检查乘法溢出，避免先乘后判断
size_t Multiply(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b) {
    throw std::runtime_error("CudaKvCache: byte count overflow");
  }
  return a * b;
}
}  // namespace

// 两块显存释放完成后才清空尺寸，释放异常按 Buffer 规范传播
void CudaKvCache::Reset() {
  keys_.Reset();
  values_.Reset();
  n_layers_ = max_seq_len_ = n_kv_heads_ = head_dim_ = device_id_ = 0;
}

// 先在临时对象中完成双 Buffer 分配与清零，再发布配置和所有权
void CudaKvCache::Reset(int layers, int positions, int heads, int dim, int device) {
  if (layers <= 0 || positions <= 0 || heads <= 0 || dim <= 0 || device < 0) {
    throw std::runtime_error("CudaKvCache: invalid dimensions or device ID");
  }
  const size_t slot = Multiply(Multiply(heads, dim), sizeof(float));
  const size_t bytes = Multiply(Multiply(layers, positions), slot);
  Multiply(bytes, 2);  // 同时校验 Key/Value 总显存字节数
  CudaKvCache replacement;
  replacement.keys_.Reset(bytes, device);
  replacement.values_.Reset(bytes, device);
  replacement.n_layers_ = layers;
  replacement.max_seq_len_ = positions;
  replacement.n_kv_heads_ = heads;
  replacement.head_dim_ = dim;
  replacement.device_id_ = device;
  replacement.Clear();
  *this = std::move(replacement);
}

bool CudaKvCacheBuilt() { return CudaRuntimeBuilt(); }

CudaKvCache::CudaKvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
                         int device_id) {
  Reset(n_layers, max_seq_len, n_kv_heads, head_dim, device_id);
}

bool CudaKvCache::empty() const { return keys_.empty() || values_.empty(); }
size_t CudaKvCache::bytes() const { return keys_.bytes() + values_.bytes(); }
const void* CudaKvCache::keys_data() const { return keys_.data(); }
const void* CudaKvCache::values_data() const { return values_.data(); }

void CudaKvCache::ValidateIndices(int layer, int pos) const {
  if (empty() || layer < 0 || layer >= n_layers_ || pos < 0 || pos >= max_seq_len_) {
    throw std::out_of_range("CudaKvCache: layer or position out of range");
  }
}

void CudaKvCache::ValidateHeadIndex(int kv_head) const {
  if (empty() || kv_head < 0 || kv_head >= n_kv_heads_) {
    throw std::out_of_range("CudaKvCache: head out of range");
  }
}

// Reset 已校验整个缓存容量；索引校验后按相同布局计算字节偏移
size_t CudaKvCache::SlotOffsetBytes(int layer, int pos) const {
  ValidateIndices(layer, pos);
  return (static_cast<size_t>(layer) * max_seq_len_ + pos) *
         static_cast<size_t>(n_kv_heads_) * head_dim_ * sizeof(float);
}

size_t CudaKvCache::HeadOffsetBytes(int layer, int pos, int kv_head) const {
  ValidateHeadIndex(kv_head);
  return SlotOffsetBytes(layer, pos) +
         static_cast<size_t>(kv_head) * head_dim_ * sizeof(float);
}

void CudaKvCache::ValidateWriteTensors(const Tensor& k, const Tensor& v) const {
  const std::vector<int> shape{n_kv_heads_, head_dim_};
  const size_t elements = static_cast<size_t>(n_kv_heads_) * head_dim_;
  if (k.shape != shape || v.shape != shape || k.size() != elements ||
      v.size() != elements) {
    throw std::runtime_error("CudaKvCache: host tensor shape/storage mismatch");
  }
}

void CudaKvCache::Clear() {
  if (empty()) {
    return;
  }
  CudaSetDevice(device_id_);
#ifdef MINI_LLAMA_USE_CUDA
  // 清零属于缓存内部实现，不扩展 Runtime 公共接口或上传主机零值
  for (auto* buffer : {&keys_, &values_}) {
    const auto status = cudaMemset(buffer->data(), 0, buffer->bytes());
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("CudaKvCache::Clear: cudaMemset: ") +
                               cudaGetErrorString(status));
    }
  }
#else
  throw std::runtime_error("CUDA KV cache was not built");
#endif
}

void CudaKvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
  const size_t offset = SlotOffsetBytes(layer, pos);
  ValidateWriteTensors(k, v);
  const size_t slot_bytes =
      static_cast<size_t>(n_kv_heads_) * head_dim_ * sizeof(float);
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(static_cast<char*>(keys_.data()) + offset, k.data.data(),
                  slot_bytes, CudaMemcpyKind::kHostToDevice);
  CudaMemcpyBytes(static_cast<char*>(values_.data()) + offset, v.data.data(),
                  slot_bytes, CudaMemcpyKind::kHostToDevice);
}

// 完整校验两个输入后再写入；扁平 Linear 输出与分头向量布局相同
void CudaKvCache::WriteDevice(int layer, int pos, const CudaTensor& k,
                              const CudaTensor& v) {
  const size_t offset = SlotOffsetBytes(layer, pos);
  const size_t slot_bytes =
      static_cast<size_t>(n_kv_heads_) * head_dim_ * sizeof(float);
  const std::vector<int> shape{n_kv_heads_, head_dim_};
  const size_t slot_elements = slot_bytes / sizeof(float);
  const std::vector<int> flat_shape{
      slot_elements <= static_cast<size_t>(std::numeric_limits<int>::max())
          ? static_cast<int>(slot_elements)
          : 0};
  const auto valid_shape = [&](const CudaTensor& tensor) {
    return tensor.shape() == shape || tensor.shape() == flat_shape;
  };
  if (!valid_shape(k) || !valid_shape(v) || k.bytes() != slot_bytes ||
      v.bytes() != slot_bytes || k.size() != slot_bytes / sizeof(float) ||
      v.size() != slot_bytes / sizeof(float) || k.empty() || v.empty() ||
      k.device_id() != device_id_ || v.device_id() != device_id_) {
    throw std::runtime_error(
        "CudaKvCache: device tensor shape/storage/device mismatch");
  }
  CudaSetDevice(device_id_);
  CudaMemcpyBytes(static_cast<char*>(keys_.data()) + offset, k.data(), slot_bytes,
                  CudaMemcpyKind::kDeviceToDevice);
  CudaMemcpyBytes(static_cast<char*>(values_.data()) + offset, v.data(), slot_bytes,
                  CudaMemcpyKind::kDeviceToDevice);
}

namespace {
// 同步下载已校验范围的数据，只供测试和调试调用
Tensor ReadSlot(const CudaDeviceBuffer& buffer, size_t offset,
                const std::vector<int>& shape, int device_id) {
  Tensor result(shape);
  CudaSetDevice(device_id);
  CudaMemcpyBytes(result.data.data(),
                  static_cast<const char*>(buffer.data()) + offset,
                  result.size() * sizeof(float), CudaMemcpyKind::kDeviceToHost);
  return result;
}
}  // namespace

Tensor CudaKvCache::ReadKey(int layer, int pos) const {
  return ReadSlot(keys_, SlotOffsetBytes(layer, pos), {n_kv_heads_, head_dim_},
                  device_id_);
}

Tensor CudaKvCache::ReadValue(int layer, int pos) const {
  return ReadSlot(values_, SlotOffsetBytes(layer, pos), {n_kv_heads_, head_dim_},
                  device_id_);
}

Tensor CudaKvCache::ReadKeyHead(int layer, int pos, int kv_head) const {
  return ReadSlot(keys_, HeadOffsetBytes(layer, pos, kv_head), {head_dim_},
                  device_id_);
}

Tensor CudaKvCache::ReadValueHead(int layer, int pos, int kv_head) const {
  return ReadSlot(values_, HeadOffsetBytes(layer, pos, kv_head), {head_dim_},
                  device_id_);
}
}  // namespace mini_llama
