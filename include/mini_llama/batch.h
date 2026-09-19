#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_BATCH_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_BATCH_H_

#include <vector>

namespace mini_llama {

// MiniBatch 对应 llama_batch：一组带位置的 token 集合
// 对于这个教学实现，每个 token 映射到一个位置
struct MiniBatch {
  std::vector<int> tokens;     // token id
  std::vector<int> positions;  // 每个 token 对应的位置

  int num_tokens() const { return static_cast<int>(tokens.size()); }

  // 构建单 token 批次（解码步骤）。
  static MiniBatch Single(int token, int pos);

  // 从序列构建多 token 批次（预填充步骤）。
  // 位置从 start_pos 开始，每次递增 1。
  static MiniBatch FromTokens(const std::vector<int>& toks, int start_pos = 0);
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_BATCH_H_
