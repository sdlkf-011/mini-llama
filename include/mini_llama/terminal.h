#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_TERMINAL_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_TERMINAL_H_

#include <iosfwd>
#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

// 处理聊天模式下的交互式终端 I/O
class Terminal {
 public:
  // 默认连接标准输入输出，也可注入流以便单元测试
  Terminal();
  Terminal(std::istream& input, std::ostream& output);

  // 流输入立即打印提示符；交互终端在 ReadLine 中由行编辑器显示和重绘
  void PrintUserPrompt() const;

  // 从输入流读取一行；遇到 EOF 时返回空字符串
  std::string ReadLine() const;

  // 从输入流读取一行；返回 false 表示遇到 EOF
  bool ReadLine(std::string* line) const;

  // 打印助手前缀
  void PrintAssistantPrefix() const;

  // 打印单个解码后的 token，不换行
  void PrintTokenText(const std::string& text) const;

  // 刷新输出流，保证 token 能够流式显示
  void Flush() const;

  // 打印一个换行
  void NewLine() const;

  // 打印 /help 文本
  void PrintHelp() const;

  // 打印会话统计信息
  void PrintStats(const ChatSession& session) const;

  // 打印当前采样参数
  void PrintParams(const SamplingParams& params) const;

  // 打印一条通用消息
  void PrintMessage(const std::string& msg) const;

 private:
  bool UsesLineEditor() const;
  std::istream* input_;
  std::ostream* output_;
  mutable bool prompt_pending_ = false;
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_TERMINAL_H_
