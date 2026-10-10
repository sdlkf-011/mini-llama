#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_KV_CACHE_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_KV_CACHE_H_

#include <cstddef>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 仅报告 CUDA 编译状态，不查询驱动或设备
bool CudaKvCacheBuilt();

// 独占 Key/Value 显存；逻辑布局与 CPU KvCache 一致：
// [n_layers, max_seq_len, n_kv_heads, head_dim]，元素为连续行主序 F32
class CudaKvCache {
 public:
  CudaKvCache() = default;
  // 构造时委托 Reset 校验、分配并清零缓存
  CudaKvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
              int device_id = 0);

  CudaKvCache(const CudaKvCache&) = delete;
  CudaKvCache& operator=(const CudaKvCache&) = delete;
  CudaKvCache(CudaKvCache&&) noexcept = default;
  CudaKvCache& operator=(CudaKvCache&&) noexcept = default;

  // 显式释放两块显存；全部释放成功后清空配置元数据
  void Reset();

  // 维度必须为正，设备编号非负；校验总字节数溢出
  // 新缓存分配及清零成功后再替换，准备失败时保留旧缓存
  void Reset(int n_layers, int max_seq_len, int n_kv_heads, int head_dim,
             int device_id = 0);

  // 保留配置与显存分配，将全部 Key/Value 清零；空缓存直接返回
  void Clear();

  // K/V 必须为 [n_kv_heads, head_dim]，存储元素数须与形状一致
  // 校验层和位置后执行 Host-to-Device 拷贝
  void Write(int layer, int pos, const Tensor& k, const Tensor& v);

  // 设备写入要求相同的二维形状、元素数、字节数和 device_id
  // 仅执行 Device-to-Device 拷贝，不下载 K/V
  void WriteDevice(int layer, int pos, const CudaTensor& k, const CudaTensor& v);

  // 测试与调试用同步回读，返回 [n_kv_heads, head_dim]；不用于热路径
  Tensor ReadKey(int layer, int pos) const;
  Tensor ReadValue(int layer, int pos) const;

  // 回读指定 KV 头，返回 [head_dim]；额外校验头索引
  Tensor ReadKeyHead(int layer, int pos, int kv_head) const;
  Tensor ReadValueHead(int layer, int pos, int kv_head) const;

  // 借用整个缓存的只读设备指针，空缓存返回 nullptr
  // 指针不可在 CPU 解引用，调用方不得释放；有效期受 Reset 和所有权约束
  const void* keys_data() const;
  const void* values_data() const;

  // 任一 Key/Value Buffer 未分配时视为空缓存
  bool empty() const;

  // 返回 Key 与 Value 两块 Buffer 的实际字节容量总和
  size_t bytes() const;

  int n_layers() const { return n_layers_; }

  int max_seq_len() const { return max_seq_len_; }

  int n_kv_heads() const { return n_kv_heads_; }

  int head_dim() const { return head_dim_; }

  int device_id() const { return device_id_; }

 private:
  void ValidateIndices(int layer, int pos) const;
  void ValidateHeadIndex(int kv_head) const;
  void ValidateWriteTensors(const Tensor& k, const Tensor& v) const;

  // 槽位偏移为 ((layer * max_seq_len) + pos) * n_kv_heads * head_dim *
  // sizeof(float)
  size_t SlotOffsetBytes(int layer, int pos) const;

  // 在槽位偏移上增加 kv_head * head_dim * sizeof(float)
  size_t HeadOffsetBytes(int layer, int pos, int kv_head) const;

  int n_layers_ = 0;
  int max_seq_len_ = 0;
  int n_kv_heads_ = 0;
  int head_dim_ = 0;
  int device_id_ = 0;
  CudaDeviceBuffer keys_;
  CudaDeviceBuffer values_;
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_KV_CACHE_H_
