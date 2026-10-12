#include "mini_llama/terminal.h"

#include <iomanip>
#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>

#ifdef MINI_LLAMA_USE_LIBEDIT
#include <histedit.h>
#include <langinfo.h>
#include <locale.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#endif

#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

namespace {
constexpr char kUserPrompt[] = "mini-llama > ";

#ifdef MINI_LLAMA_USE_LIBEDIT
// libedit 使用调用线程的 LC_CTYPE；避免改变其他线程的数字或字符格式。
class EditorLocale {
 public:
  EditorLocale() {
    locale_ = newlocale(LC_CTYPE_MASK, "", nullptr);
    if (locale_ == nullptr ||
        std::string(nl_langinfo_l(CODESET, locale_)) != "UTF-8") {
      if (locale_ != nullptr) freelocale(locale_);
      locale_ = newlocale(LC_CTYPE_MASK, "C.UTF-8", nullptr);
      if (locale_ == nullptr) {
        locale_ = newlocale(LC_CTYPE_MASK, "en_US.UTF-8", nullptr);
      }
    }
    if (locale_ == nullptr) {
      throw std::runtime_error("Terminal: a UTF-8 locale is required");
    }
    previous_ = uselocale(locale_);
    if (previous_ == nullptr) {
      freelocale(locale_);
      throw std::runtime_error("Terminal: cannot activate UTF-8 locale");
    }
  }

  ~EditorLocale() {
    uselocale(previous_);
    freelocale(locale_);
  }

  EditorLocale(const EditorLocale&) = delete;
  EditorLocale& operator=(const EditorLocale&) = delete;

 private:
  locale_t locale_ = nullptr;
  locale_t previous_ = nullptr;
};

const char* EditorPrompt(EditLine* editor) {
  void* data = nullptr;
  el_get(editor, EL_CLIENTDATA, &data);
  return static_cast<const char*>(data);
}
#endif
}  // namespace

Terminal::Terminal() : Terminal(std::cin, std::cout) {}

Terminal::Terminal(std::istream& input, std::ostream& output)
    : input_(&input), output_(&output) {}

bool Terminal::UsesLineEditor() const {
#ifdef MINI_LLAMA_USE_LIBEDIT
  const char* term = std::getenv("TERM");
  return input_ == &std::cin && output_ == &std::cout && isatty(STDIN_FILENO) &&
         isatty(STDOUT_FILENO) && term != nullptr && term[0] != '\0' &&
         std::string(term) != "dumb";
#else
  return false;
#endif
}

void Terminal::PrintUserPrompt() const {
  if (UsesLineEditor()) {
    prompt_pending_ = true;
    return;
  }
  *output_ << kUserPrompt;
  output_->flush();
}

std::string Terminal::ReadLine() const {
  std::string line;
  ReadLine(&line);
  return line;
}

bool Terminal::ReadLine(std::string* line) const {
  if (line == nullptr) return false;
#ifdef MINI_LLAMA_USE_LIBEDIT
  if (UsesLineEditor()) {
    EditorLocale locale;
    std::unique_ptr<EditLine, decltype(&el_end)> editor(
        el_init("mini-llama", stdin, stdout, stderr), &el_end);
    if (!editor) throw std::runtime_error("Terminal: cannot create line editor");
    const char* prompt = prompt_pending_ ? kUserPrompt : "";
    prompt_pending_ = false;
    if (el_set(editor.get(), EL_CLIENTDATA, prompt) != 0 ||
        el_set(editor.get(), EL_PROMPT, &EditorPrompt) != 0 ||
        el_set(editor.get(), EL_EDITOR, "emacs") != 0 ||
        el_set(editor.get(), EL_SIGNAL, 1) != 0) {
      throw std::runtime_error("Terminal: cannot configure line editor");
    }
    output_->flush();
    int count = 0;
    const char* text = el_gets(editor.get(), &count);
    if (text == nullptr || count <= 0) {
      line->clear();
      return false;
    }
    line->assign(text, static_cast<size_t>(count));
    if (!line->empty() && line->back() == '\n') line->pop_back();
    return true;
  }
#endif
  return static_cast<bool>(std::getline(*input_, *line));
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
