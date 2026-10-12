#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_FORWARD_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_FORWARD_H_

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 单个 token 的前向传播
// 返回 logits：[vocab_size]
// 存在 CUDA 权重快照时执行完整设备前向，仅下载最终 logits；否则使用 CPU
// ctx 必须由当前 model 构造；切换模型时新建上下文并重放输入
Tensor ForwardToken(MiniLlamaContext& ctx, const MiniLlamaModel& model, int token);

// 一批 token 的前向传播
// 内部按顺序处理这些 token，并返回最后一个 token 的 logits
// 每个 token 调用现有 ForwardToken，完成前向与 KV 写入
// 这样统一了预填充（多 token）和解码（单 token）路径
// 执行前检查整批 token/position 范围；这些输入越界时不改变上下文
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_FORWARD_H_
