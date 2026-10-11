// 测试目的：构造两套词表、BPE 文件和 chat template
// metadata，分别支撑单元与集成测试。
#ifndef MINI_LLAMA_TESTS_GGUF_TOKENIZER_FIXTURE_H_
#define MINI_LLAMA_TESTS_GGUF_TOKENIZER_FIXTURE_H_

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace mini_llama::test::gguf_tokenizer {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("mini_llama_gguf_tokenizer_" + std::to_string(unique));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }
  std::filesystem::path File(const std::string& name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

template <typename Value>
inline void AppendScalar(std::vector<uint8_t>* output, Value value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  output->insert(output->end(), bytes, bytes + sizeof(Value));
}

inline void AppendString(std::vector<uint8_t>* output, const std::string& value) {
  AppendScalar<uint64_t>(output, value.size());
  output->insert(output->end(), value.begin(), value.end());
}

inline void AppendStringMetadata(std::vector<uint8_t>* output,
                                 const std::string& key, const std::string& value) {
  AppendString(output, key);
  AppendScalar<uint32_t>(output, 8);
  AppendString(output, value);
}

inline void AppendStringArrayMetadata(std::vector<uint8_t>* output,
                                      const std::string& key,
                                      const std::vector<std::string>& values) {
  AppendString(output, key);
  AppendScalar<uint32_t>(output, 9);
  AppendScalar<uint32_t>(output, 8);
  AppendScalar<uint64_t>(output, values.size());
  for (const std::string& value : values) {
    AppendString(output, value);
  }
}

inline void AppendIntegerMetadata(std::vector<uint8_t>* output,
                                  const std::string& key, int32_t value) {
  AppendString(output, key);
  AppendScalar<uint32_t>(output, 5);
  AppendScalar<int32_t>(output, value);
}

inline const std::vector<std::string>& TinyTokens() {
  static const std::vector<std::string> tokens = {
      "<unk>", "<s>", "</s>", "h",     "e",           "l",     "o",
      "he",    "hel", "hell", "hello", "<|special|>", "Ġ",     "w",
      "r",     "d",   "Ġw",   "Ġwo",   "Ġwor",        "Ġworl", "Ġworld"};
  return tokens;
}

inline const std::vector<std::string>& TinyMerges() {
  static const std::vector<std::string> merges = {
      "h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d"};
  return merges;
}

inline void WriteTinyTokenizerGguf(const std::filesystem::path& path) {
  std::vector<uint8_t> bytes = {'G', 'G', 'U', 'F'};
  AppendScalar<uint32_t>(&bytes, 3);
  AppendScalar<uint64_t>(&bytes, 0);
  AppendScalar<uint64_t>(&bytes, 6);
  AppendStringArrayMetadata(&bytes, "tokenizer.ggml.tokens", TinyTokens());
  AppendStringArrayMetadata(&bytes, "tokenizer.ggml.merges", TinyMerges());
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.bos_token_id", 1);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.eos_token_id", 2);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.unknown_token_id", 0);
  AppendStringMetadata(
      &bytes, "tokenizer.chat_template",
      "{% for message in messages %}{{ message['content'] }}{% endfor %}");

  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output.write(reinterpret_cast<const char*>(bytes.data()),
                           static_cast<std::streamsize>(bytes.size())));
}

inline void WriteTinyBpeFiles(const std::filesystem::path& directory) {
  std::ofstream vocab(directory / "vocab.json");
  vocab << "{";
  for (size_t index = 0; index < TinyTokens().size(); ++index) {
    if (index != 0) {
      vocab << ",";
    }
    vocab << "\"" << TinyTokens()[index] << "\":" << index;
  }
  vocab << "}\n";

  std::ofstream merges(directory / "merges.txt");
  merges << "#version: 0.2\n";
  for (const std::string& merge : TinyMerges()) {
    merges << merge << '\n';
  }

  std::ofstream special(directory / "special_tokens.json");
  special << "{\"bos_id\":1,\"eos_id\":2,\"unk_id\":0,"
             "\"<|special|>\":11}\n";
  ASSERT_TRUE(vocab && merges && special);
}

inline void WriteBytes(const std::filesystem::path& path,
                       const std::vector<uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_TRUE(output);
}

inline void WriteTokenizerGguf(const std::filesystem::path& path,
                               const std::string& chat_template) {
  const std::vector<std::string> tokens = {"<unk>", "<s>",  "</s>",  "h",
                                           "e",     "l",    "o",     "he",
                                           "hel",   "hell", "hello", "<special>"};
  const std::vector<std::string> merges = {"h e", "he l", "hel l", "hell o"};
  std::vector<uint8_t> bytes = {'G', 'G', 'U', 'F'};
  AppendScalar<uint32_t>(&bytes, 3);
  AppendScalar<uint64_t>(&bytes, 0);
  AppendScalar<uint64_t>(&bytes, chat_template.empty() ? 5 : 6);
  AppendStringArrayMetadata(&bytes, "tokenizer.ggml.tokens", tokens);
  AppendStringArrayMetadata(&bytes, "tokenizer.ggml.merges", merges);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.bos_token_id", 1);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.eos_token_id", 2);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.unknown_token_id", 0);
  if (!chat_template.empty()) {
    AppendStringMetadata(&bytes, "tokenizer.chat_template", chat_template);
  }
  WriteBytes(path, bytes);
}

inline void WriteChatMetadataGguf(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, std::string>>& metadata) {
  std::vector<uint8_t> bytes = {'G', 'G', 'U', 'F'};
  AppendScalar<uint32_t>(&bytes, 3);
  AppendScalar<uint64_t>(&bytes, 0);
  AppendScalar<uint64_t>(&bytes, metadata.size());
  for (const auto& item : metadata) {
    AppendStringMetadata(&bytes, item.first, item.second);
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  WriteBytes(path, bytes);
}

}  // namespace mini_llama::test::gguf_tokenizer

#endif  // MINI_LLAMA_TESTS_GGUF_TOKENIZER_FIXTURE_H_
