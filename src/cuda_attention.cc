#include "mini_llama/cuda_attention.h"

#include <stdexcept>

namespace mini_llama {

namespace {

// 统一 CPU Stub 异常，明确区分未构建 CUDA 与真实设备执行失败
std::runtime_error CudaAttentionNotBuiltError() {
  return std::runtime_error(
      "CUDA attention was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on "
      "a NVIDIA CUDA machine.");
}

}  // namespace

// 编译能力查询不初始化 CUDA Runtime
bool CudaAttentionBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA
// 无 CUDA 构建时两个计算入口均抛异常，不访问设备指针或静默回退
Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache, int layer,
                           int pos, int n_heads, int n_kv_heads, int head_dim,
                           int device_id) {
  (void)q;
  (void)kv_cache;
  (void)layer;
  (void)pos;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)device_id;
  throw CudaAttentionNotBuiltError();
}

CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache, int layer,
                                          int pos, int n_heads, int n_kv_heads,
                                          int head_dim, int device_id) {
  (void)q;
  (void)kv_cache;
  (void)layer;
  (void)pos;
  (void)n_heads;
  (void)n_kv_heads;
  (void)head_dim;
  (void)device_id;
  throw CudaAttentionNotBuiltError();
}
#endif

}  // namespace mini_llama
