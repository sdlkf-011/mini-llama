// 测试目的：集中声明 main.cc 的可测试入口，并捕获 CLI 的退出码、标准输出和错误流。
#ifndef MINI_LLAMA_TESTS_MAIN_TEST_H_
#define MINI_LLAMA_TESTS_MAIN_TEST_H_

#include <sstream>
#include <string>
#include <vector>

#include "mini_llama/model.h"
#include "mini_llama/tokenizer.h"

namespace mini_llama {
int RunMain(int argc, const char* const argv[], std::istream& input,
            std::ostream& output, std::ostream& error);
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::istream& input, std::ostream& output,
        std::ostream& error);
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);
}  // namespace mini_llama

namespace mini_llama::test {

struct Result {
  int status;
  std::string output;
  std::string error;
};

inline Result Invoke(const std::vector<std::string>& args,
                     const std::string& text = "") {
  std::vector<const char*> argv;
  for (const auto& arg : args) {
    argv.push_back(arg.c_str());
  }
  std::istringstream input(text);
  std::ostringstream output, error;
  const int status = mini_llama::RunMain(static_cast<int>(argv.size()), argv.data(),
                                         input, output, error);
  return {status, output.str(), error.str()};
}

}  // namespace mini_llama::test

#endif  // MINI_LLAMA_TESTS_MAIN_TEST_H_
