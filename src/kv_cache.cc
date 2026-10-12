#include "mini_llama/kv_cache.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "mini_llama/tensor.h"

namespace mini_llama {

// 检查缓存为合法四维张量，且实际存储与形状一致，避免公开字段被修改后越界
static void Require4DCacheShape(const Tensor& t, const char* name) {
  if (t.num_dims() != 4) {
    throw std::runtime_error(std::string(name) + " must be a 4D tensor");
  }
  size_t elements = 1;
  for (int dim : t.shape) {
    if (dim <= 0 ||
        elements > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error(std::string(name) + " has invalid dimensions");
    }
    elements *= static_cast<size_t>(dim);
  }
  if (t.data.size() != elements) {
    throw std::runtime_error(std::string(name) + " storage does not match shape");
  }
}

KvCache::KvCache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
  keys = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
  values = MakeTensor4D(n_layers, max_seq_len, n_kv_heads, head_dim, 0.0f);
}

void KvCache::Write(int layer, int pos, const Tensor& k, const Tensor& v) {
  // k: [n_kv_heads, head_dim]
  // v: [n_kv_heads, head_dim]
  Require4DCacheShape(keys, "keys");
  Require4DCacheShape(values, "values");
  if (keys.shape != values.shape) {
    throw std::runtime_error("KvCache::Write key/value cache shape mismatch");
  }

  if (k.num_dims() != 2 || v.num_dims() != 2 || k.shape != v.shape) {
    throw std::out_of_range(
        "KvCache::Write expected matching [n_kv_heads, head_dim] tensors");
  }

  int n_kv_heads = k.shape[0];
  int head_dim = k.shape[1];
  if (layer < 0 || layer >= keys.shape[0]) {
    throw std::out_of_range("KvCache::Write layer out of range");
  }
  if (pos < 0 || pos >= keys.shape[1]) {
    throw std::out_of_range("KvCache::Write position out of range");
  }
  if (n_kv_heads != keys.shape[2] || head_dim != keys.shape[3]) {
    throw std::out_of_range(
        "KvCache::Write tensor shape does not match cache shape");
  }
  const size_t slot_elements = static_cast<size_t>(n_kv_heads) * head_dim;
  if (k.data.size() != slot_elements || v.data.size() != slot_elements) {
    throw std::runtime_error("KvCache::Write input storage does not match shape");
  }

  // keys shape: [n_layers, max_seq_len, n_kv_heads, head_dim]
  // 我们需要写在 [layer, pos, :, :]
  size_t layer_stride =
      static_cast<size_t>(keys.shape[1]) * keys.shape[2] * keys.shape[3];
  size_t pos_stride = static_cast<size_t>(keys.shape[2]) * keys.shape[3];
  size_t head_stride = static_cast<size_t>(head_dim);

  size_t base = layer * layer_stride + pos * pos_stride;

  for (int h = 0; h < n_kv_heads; ++h) {
    size_t offset = base + h * head_stride;
    const size_t input_offset = static_cast<size_t>(h) * head_stride;
    std::memcpy(&keys.data[offset], &k.data[input_offset], head_dim * sizeof(float));
    std::memcpy(&values.data[offset], &v.data[input_offset],
                head_dim * sizeof(float));
  }
}

const float* KvCache::KeyPtr(int layer, int pos, int kv_head) const {
  Require4DCacheShape(keys, "keys");
  if (layer < 0 || layer >= keys.shape[0] || pos < 0 || pos >= keys.shape[1] ||
      kv_head < 0 || kv_head >= keys.shape[2]) {
    throw std::out_of_range("KvCache::KeyPtr FlatIndex out of range");
  }

  size_t layer_stride =
      static_cast<size_t>(keys.shape[1]) * keys.shape[2] * keys.shape[3];
  size_t pos_stride = static_cast<size_t>(keys.shape[2]) * keys.shape[3];
  size_t head_stride = static_cast<size_t>(keys.shape[3]);

  return &keys.data[layer * layer_stride + pos * pos_stride + kv_head * head_stride];
}

const float* KvCache::valuePtr(int layer, int pos, int kv_head) const {
  Require4DCacheShape(values, "values");
  if (layer < 0 || layer >= values.shape[0] || pos < 0 || pos >= values.shape[1] ||
      kv_head < 0 || kv_head >= values.shape[2]) {
    throw std::out_of_range("KvCache::valuePtr FlatIndex out of range");
  }

  size_t layer_stride =
      static_cast<size_t>(values.shape[1]) * values.shape[2] * values.shape[3];
  size_t pos_stride = static_cast<size_t>(values.shape[2]) * values.shape[3];
  size_t head_stride = static_cast<size_t>(values.shape[3]);

  return &values
              .data[layer * layer_stride + pos * pos_stride + kv_head * head_stride];
}

}  // namespace mini_llama
