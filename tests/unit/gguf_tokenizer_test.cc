#include "mini_llama/gguf_tokenizer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using mini_llama::CreateGgufTokenizer;
using mini_llama::GgufTokenizer;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static int sequence = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("mini_llama_gguf_test_" + std::to_string(++sequence));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  std::filesystem::path File(const std::string& name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

template <typename Value>
void AppendScalar(std::vector<uint8_t>* output, Value value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  output->insert(output->end(), bytes, bytes + sizeof(Value));
}

void AppendString(std::vector<uint8_t>* output, const std::string& value) {
  AppendScalar<uint64_t>(output, value.size());
  output->insert(output->end(), value.begin(), value.end());
}

void AppendStringMetadata(std::vector<uint8_t>* output, const std::string& key,
                          const std::string& value) {
  constexpr uint32_t kString = 8;
  AppendString(output, key);
  AppendScalar<uint32_t>(output, kString);
  AppendString(output, value);
}

void AppendStringArrayMetadata(std::vector<uint8_t>* output, const std::string& key,
                               const std::vector<std::string>& values) {
  constexpr uint32_t kString = 8;
  constexpr uint32_t kArray = 9;
  AppendString(output, key);
  AppendScalar<uint32_t>(output, kArray);
  AppendScalar<uint32_t>(output, kString);
  AppendScalar<uint64_t>(output, values.size());
  for (const std::string& value : values) {
    AppendString(output, value);
  }
}

void AppendIntegerMetadata(std::vector<uint8_t>* output, const std::string& key,
                           int32_t value) {
  constexpr uint32_t kInt32 = 5;
  AppendString(output, key);
  AppendScalar<uint32_t>(output, kInt32);
  AppendScalar<int32_t>(output, value);
}

void WriteBytes(const std::filesystem::path& path,
                const std::vector<uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_TRUE(output);
}

void WriteTokenizerGguf(const std::filesystem::path& path,
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

// 测试目的：验证 GGUF 元数据能够加载词表、BPE 合并规则、控制 token 和
// 特殊 token，并完成可逆的编码与解码。
TEST(GgufTokenizerTest, LoadsMetadataAndEncodesBpeTokens) {
  TemporaryDirectory directory;
  const auto path = directory.File("tokenizer.gguf");
  WriteTokenizerGguf(path, "qwen-template");
  GgufTokenizer tokenizer;

  ASSERT_TRUE(tokenizer.Load(path.string()));
  EXPECT_EQ(tokenizer.vocab_size(), 12);
  EXPECT_EQ(tokenizer.bos_id(), 1);
  EXPECT_EQ(tokenizer.eos_id(), 2);
  EXPECT_EQ(tokenizer.unk_id(), 0);
  EXPECT_EQ(tokenizer.Encode("hello<special>"), std::vector<int>({10, 11}));
  EXPECT_EQ(tokenizer.Decode({10, 11}), "hello<special>");
}

// 测试目的：验证 GGUF 工厂拒绝损坏文件，避免向聊天主流程暴露半初始化
// 的分词器对象。
TEST(GgufTokenizerTest, FactoryRejectsInvalidGguf) {
  TemporaryDirectory directory;
  const auto invalid_path = directory.File("invalid.gguf");
  WriteBytes(invalid_path, {'b', 'a', 'd'});

  EXPECT_EQ(CreateGgufTokenizer(invalid_path.string()), nullptr);
}

}  // namespace
