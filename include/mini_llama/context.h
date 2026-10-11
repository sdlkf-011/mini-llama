#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CONTEXT_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CONTEXT_H_

#include <vector>

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"

namespace mini_llama {

// 推理上下文：保存 KV 缓存、当前位置、token 历史和统计信息
struct MiniLlamaContext {
  const MiniLlamaModel* model = nullptr;
  KvCache kv_cache;
  // 完整 CUDA 前向首次执行时分配显存，后续 Token 保留历史
  CudaKvCache cuda_kv_cache;
  int pos = 0;

  // 所有已经送入此上下文的 token（包括预填充阶段）
  std::vector<int> token_history;

  // 统计信息
  int n_prefill_tokens = 0;
  int n_decode_tokens = 0;

  MiniLlamaContext() = default;
  MiniLlamaContext(const MiniLlamaContext&) = delete;
  MiniLlamaContext& operator=(const MiniLlamaContext&) = delete;
  MiniLlamaContext(MiniLlamaContext&&) noexcept = default;
  MiniLlamaContext& operator=(MiniLlamaContext&&) noexcept = default;

  explicit MiniLlamaContext(const MiniLlamaModel* model);
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CONTEXT_H_
