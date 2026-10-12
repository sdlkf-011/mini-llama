#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_KV_CACHE_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_KV_CACHE_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

// KV 缓存为所有层和位置存储键和值
// keys:   [层数, 最大序列长度, KV 头数, 头维度]
// values: [层数, 最大序列长度, KV 头数, 头维度]
struct KvCache {
  Tensor keys;
  Tensor values;

  KvCache() = default;
  KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);

  // 为指定层和位置写入 k 和 v
  // k: [KV 头数, 头维度]
  // v: [KV 头数, 头维度]
  void Write(int layer, int pos, const Tensor& k, const Tensor& v);

  // 借用指定层、位置、KV 头的键或值，连续可读范围为 head_dim 个元素
  // 遍历历史时由调用方逐位置查询；返回指针不拥有数据，缓存存储变化后不可继续使用
  // 拒绝无效缓存形状、存储大小和越界索引
  const float* KeyPtr(int layer, int pos, int kv_head) const;
  const float* valuePtr(int layer, int pos, int kv_head) const;
};

}  // namespace mini_llama

#endif  //  MINI_LLAMA_INCLUDE_MINI_LLAMA_KV_CACHE_H_
