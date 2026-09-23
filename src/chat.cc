#include "mini_llama/chat.h"

#include <cstddef>
#include <string>
#include <vector>

namespace mini_llama {

void ChatSession::Clear() {
  // /clear 需要同时丢弃消息、可复用前缀和累计统计，避免状态交叉污染。
  messages.clear();
  token_history.clear();
  prefix_cache.Clear();
  total_prompt_tokens = 0;
  total_generated_tokens = 0;
  total_time_ms = 0;
}

void ChatSession::AddMessage(const std::string& role, const std::string& content) {
  messages.push_back({role, content});
}

void ChatSession::RecordTurn(int prompt_tokens, int generated_tokens,
                             double time_ms) {
  // 统计以完整轮次为单位累加，供终端随时查询。
  total_prompt_tokens += prompt_tokens;
  total_generated_tokens += generated_tokens;
  total_time_ms += time_ms;
}

void ChatSession::SetTokenHistory(const std::vector<int>& tokens) {
  token_history = tokens;
}

void ChatSession::AppendToken(int token) { token_history.push_back(token); }

void ChatSession::RecordPrefix(const std::vector<int>& tokens) {
  // 前缀树只记录已完成前向计算的序列，不能登记尚未评估的 token。
  prefix_cache.Insert(tokens);
}

size_t ChatSession::LongestCachedPrefix(const std::vector<int>& tokens) const {
  return prefix_cache.LongestPrefix(tokens);
}

}  // namespace mini_llama
