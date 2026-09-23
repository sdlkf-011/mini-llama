#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CHAT_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CHAT_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/radix_tree.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

struct ChatMessage {
  // 消息角色使用 "system"、"user" 或 "assistant"
  std::string role;
  std::string content;
};

// 管理交互式聊天会话的消息、前缀和累计统计信息
class ChatSession {
 public:
  std::vector<ChatMessage> messages;
  std::vector<int> token_history;
  RadixTree prefix_cache;
  SamplingParams sampling_params;

  // 统计信息用于 /stats 命令展示整个会话的累计开销
  int total_prompt_tokens = 0;
  int total_generated_tokens = 0;
  double total_time_ms = 0.0;

  ChatSession() = default;

  // 重置消息、缓存前缀和统计信息
  void Clear();

  // 向历史记录中添加一条消息
  void AddMessage(const std::string& role, const std::string& content);

  // 在一轮对话后累加统计信息
  void RecordTurn(int prompt_tokens, int generated_tokens, double time_ms);

  // 替换最近已评估上下文的当前 token 历史记录
  void SetTokenHistory(const std::vector<int>& tokens);

  // 将一个生成且已评估的 token 追加到所跟踪的上下文中
  void AppendToken(int token);

  // 记录一个其前缀已被评估过的 token 序列
  void RecordPrefix(const std::vector<int>& tokens);

  // 返回某个 token 序列的最长已缓存前缀
  size_t LongestCachedPrefix(const std::vector<int>& tokens) const;
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CHAT_H_
