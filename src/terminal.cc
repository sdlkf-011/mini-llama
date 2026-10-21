#include "mini_llama/terminal.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cwchar>
#include <iomanip>
#include <iostream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <langinfo.h>
#include <locale.h>
#include <unistd.h>

#include <cstdlib>

#ifndef MINI_LLAMA_USE_LIBEDIT
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#endif
#endif

#ifdef MINI_LLAMA_USE_LIBEDIT
#include <histedit.h>

#include <cstdio>
#endif

#include "mini_llama/chat.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

namespace {
constexpr char kUserPrompt[] = "mini-llama > ";

#if defined(__unix__) || defined(__APPLE__)
// 行编辑使用调用线程的 LC_CTYPE；避免改变其他线程的数字或字符格式。
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

#ifndef MINI_LLAMA_USE_LIBEDIT

volatile sig_atomic_t input_signal = 0;
constexpr int kSignals[] = {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGTSTP};

void OnInputSignal(int signal) { input_signal = signal; }

struct InputSignal {
  int number;
};

// 只在一轮交互输入期间接管终端，正常返回、异常和上述信号退出均恢复原设置。
class InputMode {
 public:
  InputMode() {
    if (tcgetattr(STDIN_FILENO, &original_) != 0) {
      throw std::runtime_error("Terminal: cannot read terminal settings");
    }
    input_signal = 0;
    struct sigaction action {};
    action.sa_handler = OnInputSignal;
    sigemptyset(&action.sa_mask);
    for (int signal : kSignals) {
      if (sigaction(signal, &action, &previous_[installed_]) != 0) {
        Restore();
        throw std::runtime_error("Terminal: cannot install signal handler");
      }
      ++installed_;
    }
    termios settings = original_;
    settings.c_iflag &= ~(IXON | ICRNL | INLCR | IGNCR | ISTRIP);
    settings.c_lflag &= ~(ICANON | ECHO | ECHONL | IEXTEN);
    settings.c_cc[VMIN] = 1;
    settings.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &settings) != 0) {
      Restore();
      throw std::runtime_error("Terminal: cannot configure terminal input");
    }
    active_ = true;
  }

  ~InputMode() { Restore(); }
  InputMode(const InputMode&) = delete;
  InputMode& operator=(const InputMode&) = delete;

 private:
  void Restore() {
    if (active_) {
      while (tcsetattr(STDIN_FILENO, TCSANOW, &original_) != 0 && errno == EINTR) {
      }
      active_ = false;
    }
    while (installed_ > 0) {
      --installed_;
      sigaction(kSignals[installed_], &previous_[installed_], nullptr);
    }
  }

  termios original_{};
  struct sigaction previous_[sizeof(kSignals) / sizeof(kSignals[0])]{};
  size_t installed_ = 0;
  bool active_ = false;
};

// -1 表示 EOF，-2 表示转义序列等待超时。轮询也避免信号恰好在 read 前
// 到达时无限等待；信号处理函数只写 sig_atomic_t，不执行终端或流操作。
int ReadByte(int timeout_ms = -1) {
  for (;;) {
    if (input_signal != 0) throw InputSignal{input_signal};
    pollfd descriptor{STDIN_FILENO, POLLIN, 0};
    const int ready = poll(&descriptor, 1, timeout_ms < 0 ? 100 : timeout_ms);
    if (input_signal != 0) throw InputSignal{input_signal};
    if (ready < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("Terminal: cannot wait for input");
    }
    if (ready == 0) {
      if (timeout_ms >= 0) return -2;
      continue;
    }
    unsigned char byte;
    const ssize_t count = read(STDIN_FILENO, &byte, 1);
    if (count == 1) return byte;
    if (count == 0) return -1;
    if (errno != EINTR) throw std::runtime_error("Terminal: cannot read input");
  }
}

std::string Encode(wchar_t character) {
  char bytes[MB_LEN_MAX];
  std::mbstate_t state{};
  const size_t count = std::wcrtomb(bytes, character, &state);
  if (count == static_cast<size_t>(-1)) return "?";
  return std::string(bytes, count);
}

size_t Previous(const std::wstring& text, size_t cursor) {
  if (cursor == 0) return 0;
  --cursor;
  while (cursor > 0 && wcwidth(text[cursor]) == 0) --cursor;
  return cursor;
}

size_t Next(const std::wstring& text, size_t cursor) {
  if (cursor == text.size()) return cursor;
  ++cursor;
  while (cursor < text.size() && wcwidth(text[cursor]) == 0) ++cursor;
  return cursor;
}

// 不依赖字节数推断光标位置；宽字符按显示列布局，并显式处理行末换行。
class InputDisplay {
 public:
  explicit InputDisplay(std::ostream& output) : output_(output) {}

  void Draw(const char* prompt, const std::wstring& text, size_t cursor) {
    winsize window{};
    const int columns =
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &window) == 0 && window.ws_col > 0
            ? window.ws_col
            : 80;
    output_ << '\r';
    if (cursor_row_ > 0) output_ << "\x1b[" << cursor_row_ << 'A';
    output_ << "\x1b[J";
    int row = 0, column = 0;
    auto put = [&](wchar_t character) {
      const int width = std::max(0, wcwidth(character));
      if (column + width > columns) {
        output_ << "\r\n";
        ++row;
        column = 0;
      }
      output_ << Encode(character);
      column += width;
      if (column >= columns) {
        output_ << "\r\n";
        ++row;
        column = 0;
      }
    };
    for (const char* p = prompt; *p != '\0'; ++p) put(*p);
    int target_row = row, target_column = column;
    for (size_t i = 0; i < text.size(); ++i) {
      put(text[i]);
      if (i + 1 == cursor) {
        target_row = row;
        target_column = column;
      }
    }
    output_ << '\r';
    if (row > target_row) output_ << "\x1b[" << row - target_row << 'A';
    if (target_column > 0) output_ << "\x1b[" << target_column << 'C';
    output_.flush();
    cursor_row_ = target_row;
  }

 private:
  std::ostream& output_;
  int cursor_row_ = 0;
};

void EscapeKey(const std::wstring& text, size_t* cursor, size_t* erase) {
  const int prefix = ReadByte(50);
  if (prefix != '[' && prefix != 'O') return;
  std::string key;
  for (int i = 0; i < 16; ++i) {
    const int byte = ReadByte(50);
    if (byte < 0) return;
    key += static_cast<char>(byte);
    if ((byte >= 'A' && byte <= 'Z') || byte == '~') break;
  }
  if (key == "D") *cursor = Previous(text, *cursor);
  if (key == "C") *cursor = Next(text, *cursor);
  if (key == "H" || key == "1~" || key == "7~") *cursor = 0;
  if (key == "F" || key == "4~" || key == "8~") *cursor = text.size();
  if (key == "3~") *erase = Next(text, *cursor) - *cursor;
}

bool ReadTerminalInput(const char* prompt, std::string* line, std::ostream& output) {
  line->clear();
  std::wstring text;
  size_t cursor = 0;
  for (;;) {
    try {
      InputMode mode;
      InputDisplay display(output);
      std::mbstate_t decoder{};
      display.Draw(prompt, text, cursor);
      for (;;) {
        const int byte = ReadByte();
        if (byte < 0 || byte == '\r' || byte == '\n' ||
            (byte == 4 && text.empty())) {
          display.Draw(prompt, text, text.size());
          output << "\r\n";
          output.flush();
          if ((byte < 0 || byte == 4) && text.empty()) return false;
          for (wchar_t character : text) *line += Encode(character);
          return true;
        }
        size_t erase = 0;
        if (byte < 32 || byte == 127) {
          decoder = {};
          switch (byte) {
            case 1:
              cursor = 0;
              break;  // Ctrl-A
            case 2:
              cursor = Previous(text, cursor);
              break;
            case 3:
              throw InputSignal{SIGINT};
            case 4:
              erase = Next(text, cursor) - cursor;
              break;
            case 5:
              cursor = text.size();
              break;  // Ctrl-E
            case 6:
              cursor = Next(text, cursor);
              break;
            case 8:
            case 127: {
              const size_t previous = Previous(text, cursor);
              erase = cursor - previous;
              cursor = previous;
              break;
            }
            case 11:
              erase = text.size() - cursor;
              break;  // Ctrl-K
            case 21:
              erase = cursor;
              cursor = 0;
              break;  // Ctrl-U
            case 26:
              throw InputSignal{SIGTSTP};
            case 27:
              EscapeKey(text, &cursor, &erase);
              break;
          }
          text.erase(cursor, erase);
        } else {
          const char input = static_cast<char>(byte);
          wchar_t character;
          const size_t count = std::mbrtowc(&character, &input, 1, &decoder);
          if (count == static_cast<size_t>(-2)) continue;
          if (count == static_cast<size_t>(-1)) {
            decoder = {};
            continue;
          }
          if (wcwidth(character) < 0) continue;
          text.insert(cursor++, 1, character);
        }
        display.Draw(prompt, text, cursor);
      }
    } catch (const InputSignal& signal) {
      // InputMode 已恢复 termios 和原处理器，再交还信号的默认/原有行为。
      raise(signal.number);
      if (signal.number != SIGTSTP) return false;
      output << "\r\n";
    }
  }
}

#endif

#endif

#ifdef MINI_LLAMA_USE_LIBEDIT
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
#if defined(__unix__) || defined(__APPLE__)
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
#if defined(__unix__) || defined(__APPLE__)
  if (UsesLineEditor()) {
    EditorLocale locale;
    const char* prompt = prompt_pending_ ? kUserPrompt : "";
    prompt_pending_ = false;
#ifdef MINI_LLAMA_USE_LIBEDIT
    std::unique_ptr<EditLine, decltype(&el_end)> editor(
        el_init("mini-llama", stdin, stdout, stderr), &el_end);
    if (!editor) throw std::runtime_error("Terminal: cannot create line editor");
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
#else
    return ReadTerminalInput(prompt, line, *output_);
#endif
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
