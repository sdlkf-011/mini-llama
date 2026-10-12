#include "mini_llama/batch.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {
MiniBatch MiniBatch::Single(int token, int pos) {
  MiniBatch batch;
  batch.tokens.push_back(token);
  batch.positions.push_back(pos);

  return batch;
}

MiniBatch MiniBatch::FromTokens(const std::vector<int>& toks, int start_pos) {
  // token 数和连续位置都必须可由 int 表示，先检查再复制输入与计算位置
  const int max_position = std::numeric_limits<int>::max();
  if (toks.size() > static_cast<size_t>(max_position) ||
      (!toks.empty() &&
       start_pos > max_position - static_cast<int>(toks.size() - 1))) {
    throw std::out_of_range("MiniBatch: token count or position exceeds int range");
  }
  MiniBatch batch;
  batch.tokens = toks;
  batch.positions.reserve(toks.size());

  for (size_t i = 0; i < toks.size(); ++i) {
    batch.positions.push_back(start_pos + static_cast<int>(i));
  }

  return batch;
}

// 按顺序处理批次中的所有 token
// 返回批次中*最后一个* token 的 logits
// 这样统一了预填充（多 token）和解码（单 token）路径
Tensor ForwardBatch(MiniLlamaContext& ctx, const MiniLlamaModel& model,
                    const MiniBatch& batch) {
  if (batch.num_tokens() == 0) {
    throw std::runtime_error("ForwardBatch: empty batch");
  }

  if (batch.tokens.size() != batch.positions.size()) {
    throw std::runtime_error("ForwardBatch: tokens and positions size mismatch");
  }
  if (ctx.model != &model) {
    throw std::runtime_error("ForwardBatch: context belongs to a different model");
  }

  // 先校验整个输入，避免末尾 token 或 position 越界时已写入部分 KV 历史
  // 模型、权重和缓存的完整校验仍由 ForwardToken 负责
  for (size_t i = 0; i < batch.tokens.size(); ++i) {
    if (batch.tokens[i] < 0 || batch.tokens[i] >= model.config.vocab_size) {
      throw std::out_of_range("ForwardBatch token id out of range");
    }
    if (batch.positions[i] < 0 || batch.positions[i] >= model.config.max_seq_len) {
      throw std::out_of_range("ForwardBatch position out of range");
    }
  }

  Tensor logits;
  for (int i = 0; i < batch.num_tokens(); ++i) {
    ctx.pos = batch.positions[i];
    logits = ForwardToken(ctx, model, batch.tokens[i]);
    ctx.token_history.push_back(batch.tokens[i]);
  }

  return logits;
}

}  // namespace mini_llama
