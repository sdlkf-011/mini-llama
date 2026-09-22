#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_

#include <string>
#include <vector>

#include "mini_llama/chat.h"

namespace mini_llama {

// 根据聊天消息历史构建文本提示词，并执行本阶段支持的 Jinja2 子集
// 模板来源与执行逻辑解耦，后续模块只需通过 SetChatTemplate 注入字符串
class PromptBuilder {
 public:
  PromptBuilder() = default;

  // 设置聊天模板；空模板使用纯文本模式，qwen2 使用内置回退格式
  void SetChatTemplate(const std::string& template_str);

  // 根据消息构建提示词
  std::string Build(const std::vector<ChatMessage>& messages) const;

 private:
  std::string chat_template_;

  std::string BuildPlain(const std::vector<ChatMessage>& messages) const;
  std::string BuildQwen2(const std::vector<ChatMessage>& messages) const;
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_PROMPT_BUILDER_H_
