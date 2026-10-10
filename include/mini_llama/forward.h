#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_FORWARD_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 单个 token 的前向传播
// 返回 logits：[vocab_size]
// 命中上传快照的 F32 Linear 使用 CUDA；其余算子和未命中/量化 Linear 使用 CPU
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token);

// 一批 token 的前向传播
// 内部按顺序处理这些 token，并返回最后一个 token 的 logits
// 这样统一了预填充（多 token）和解码（单 token）路径
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_FORWARD_H_
