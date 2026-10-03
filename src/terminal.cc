#include "mini_llama/terminal.h"

#include <iomanip>
#include <iostream>
#include <istream>
#include <ostream>
#include <string>

#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

Terminal::Terminal() : Terminal(std::cin, std::cout) {}

Terminal::Terminal(std::istream& input, std::ostream& output)
    : input_(&input), output_(&output) {}

void Terminal::PrintUserPrompt() const {
  *output_ << "mini-llama > ";
  output_->flush();
}

std::string Terminal::ReadLine() const {
  std::string line;
  ReadLine(&line);
  return line;
}

bool Terminal::ReadLine(std::string* line) const {
  return line != nullptr && static_cast<bool>(std::getline(*input_, *line));
}

void Terminal::PrintAssistantPrefix() const { *output_ << "assistant > "; }

void Terminal::PrintTokenText(const std::string& text) const { *output_ << text; }

void Terminal::Flush() const { output_->flush(); }

void Terminal::NewLine() const { *output_ << '\n'; }

void Terminal::PrintHelp() const {
  // 所有聊天命令的帮助集中在终端模块，调用方无需追加单独的说明
  *output_ << "Commands:\n"
           << "  /help   Show this help\n"
           << "  /clear  Clear conversation history\n"
           << "  /stats  Show session statistics\n"
           << "  /params Show current sampling parameters\n"
           << "  /exit   Exit chat\n";
}

void Terminal::PrintStats(const ChatSession& session) const {
  // 临时调整浮点格式，避免影响后续终端输出
  const std::ios::fmtflags old_flags = output_->flags();
  const std::streamsize old_precision = output_->precision();
  const double tokens_per_second =
      session.total_time_ms > 0.0
          ? session.total_generated_tokens * 1000.0 / session.total_time_ms
          : 0.0;
  *output_ << "Messages: " << session.messages.size() << '\n'
           << "Context tokens: " << session.token_history.size() << '\n'
           << "Prompt tokens: " << session.total_prompt_tokens << '\n'
           << "Generated tokens: " << session.total_generated_tokens << '\n'
           << std::fixed << std::setprecision(2)
           << "Elapsed: " << session.total_time_ms << " ms\n"
           << "Speed: " << tokens_per_second << " tokens/s\n";
  output_->flags(old_flags);
  output_->precision(old_precision);
}

void Terminal::PrintParams(const SamplingParams& params) const {
  *output_ << "temperature=" << params.temperature << ", top_k=" << params.top_k
           << ", seed=" << params.seed << '\n';
}

void Terminal::PrintMessage(const std::string& msg) const {
  *output_ << msg;
  if (msg.empty() || msg.back() != '\n') {
    *output_ << '\n';
  }
}

}  // namespace mini_llama
