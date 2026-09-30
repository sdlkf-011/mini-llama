#include "mini_llama/tokenizer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/gguf_tokenizer.h"

namespace {

using mini_llama::AsciiTokenizer;
using mini_llama::BpeTokenizer;
using mini_llama::GgufTokenizer;
using mini_llama::ITokenizer;
using mini_llama::JsonVocabTokenizer;

void Check(bool condition, const std::string& message) {
  EXPECT_TRUE(condition) << message;
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static int sequence = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("mini_llama_tokenizer_test_" + std::to_string(++sequence));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  std::filesystem::path File(const std::string& name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

void WriteText(const std::filesystem::path& path, const std::string& content) {
  std::ofstream output(path, std::ios::binary);
  Check(static_cast<bool>(output), "无法创建测试文本文件");
  output << content;
  Check(static_cast<bool>(output), "无法写入测试文本文件");
}

template <typename Value>
void AppendScalar(std::vector<uint8_t>* output, Value value) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
  output->insert(output->end(), bytes, bytes + sizeof(Value));
}

void AppendString(std::vector<uint8_t>* output, const std::string& value) {
  AppendScalar<uint64_t>(output, value.size());
  output->insert(output->end(), value.begin(), value.end());
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

void WriteGguf(const std::filesystem::path& path,
               const std::vector<std::string>& tokens,
               const std::vector<std::string>& merges) {
  std::vector<uint8_t> bytes = {'G', 'G', 'U', 'F'};

  AppendScalar<uint32_t>(&bytes, 3);
  AppendScalar<uint64_t>(&bytes, 0);
  AppendScalar<uint64_t>(&bytes, 5);
  AppendStringArrayMetadata(&bytes, "tokenizer.ggml.tokens", tokens);
  AppendStringArrayMetadata(&bytes, "tokenizer.ggml.merges", merges);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.bos_token_id", 1);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.eos_token_id", 2);
  AppendIntegerMetadata(&bytes, "tokenizer.ggml.unknown_token_id", 0);

  std::ofstream output(path, std::ios::binary);
  Check(static_cast<bool>(output), "无法创建 GGUF 测试文件");
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  Check(static_cast<bool>(output), "无法写入 GGUF 测试文件");
}

const std::vector<std::string>& BpeTokens() {
  static const std::vector<std::string> tokens = {
      "<unk>", "<s>", "</s>", "h",    "e",     "l",
      "o",     "he",  "hel",  "hell", "hello", "<special>"};
  return tokens;
}

const std::vector<std::string>& BpeMerges() {
  static const std::vector<std::string> merges = {"h e", "he l", "hel l", "hell o"};
  return merges;
}

void WriteBpeFiles(const TemporaryDirectory& directory) {
  WriteText(
      directory.File("vocab.json"),
      R"({"<unk>":0,"<s>":1,"</s>":2,"h":3,"e":4,"l":5,"o":6,"he":7,"hel":8,"hell":9,"hello":10,"<special>":11})");
  WriteText(directory.File("merges.txt"),
            "#version: 0.2\nh e\nhe l\nhel l\nhell o\n");
  WriteText(directory.File("special_tokens.json"), R"({"<special>":11})");
}

// 测试 ASCII 分词器对合法字符、非 ASCII 字节和非法 token 的处理。
TEST(TokenizerTest, AsciiTokenizer) {
  AsciiTokenizer tokenizer;

  Check(tokenizer.vocab_size() == 128, "ASCII 词表大小错误");
  Check(
      tokenizer.bos_id() == 1 && tokenizer.eos_id() == 2 && tokenizer.unk_id() == 0,
      "ASCII 控制 token ID 错误");
  Check(tokenizer.Encode("Az") == std::vector<int>({'A', 'z'}), "ASCII 编码错误");
  Check(tokenizer.Encode("\xC3\xA9") == std::vector<int>({0, 0}),
        "非 ASCII 字节未回退到 UNK");
  Check(tokenizer.Decode({65, 66}) == "AB", "ASCII 解码错误");
  Check(tokenizer.DecodeToken(-1).empty() && tokenizer.DecodeToken(128).empty(),
        "非法 ASCII token 应解码为空串");
}

// 测试 JSON 词表的 BOS、最长匹配、逐字节 UNK 回退和双向解码。
TEST(TokenizerTest, JsonVocabTokenizer) {
  TemporaryDirectory directory;
  const auto vocab = directory.File("vocab.json");
  WriteText(
      vocab,
      R"([{"id":0,"content":"<unk>","special":true},{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true},{"id":3,"content":"a","special":false},{"id":4,"content":"ab","special":false},{"id":5,"content":"你","special":false}])");

  JsonVocabTokenizer tokenizer(vocab.string());

  Check(tokenizer.vocab_size() == 6, "JSON 词表大小错误");
  Check(
      tokenizer.bos_id() == 1 && tokenizer.eos_id() == 2 && tokenizer.unk_id() == 0,
      "JSON 控制 token ID 错误");
  Check(tokenizer.Encode("ab你x") == std::vector<int>({1, 4, 5, 0}),
        "JSON BOS、最长匹配或 UNK 回退错误");
  Check(tokenizer.Encode("好") == std::vector<int>({1, 0, 0, 0}),
        "JSON 未知 UTF-8 字符应逐字节回退");
  Check(tokenizer.Decode({4, 5, 0}) == "ab你<unk>", "JSON 解码错误");
  Check(tokenizer.DecodeToken(99) == "<unk>", "越界 JSON token 应解码为 UNK");
}

// 测试损坏或不存在的 JSON 词表会通过异常明确报告加载失败。
TEST(TokenizerTest, InvalidJsonVocabTokenizer) {
  TemporaryDirectory directory;
  const auto invalid = directory.File("invalid.json");
  WriteText(invalid, "not-json");

  EXPECT_THROW(JsonVocabTokenizer malformed(invalid.string()), std::runtime_error);
  EXPECT_THROW(JsonVocabTokenizer missing(directory.File("missing.json").string()),
               std::runtime_error);
}

// 测试独立 BPE 文件的加载、合并优先级、特殊 token 和解码流程。
TEST(TokenizerTest, BpeTokenizer) {
  TemporaryDirectory directory;
  WriteBpeFiles(directory);
  BpeTokenizer tokenizer;

  Check(tokenizer.Load(directory.File("vocab.json").string(),
                       directory.File("merges.txt").string(),
                       directory.File("special_tokens.json").string()),
        "BPE 文件加载失败");
  Check(tokenizer.vocab_size() == 12, "BPE 词表大小错误");
  Check(
      tokenizer.bos_id() == 1 && tokenizer.eos_id() == 2 && tokenizer.unk_id() == 0,
      "BPE 控制 token ID 错误");
  Check(tokenizer.Encode("hello<special>") == std::vector<int>({10, 11}),
        "BPE 合并或特殊 token 匹配错误");
  Check(tokenizer.Decode({10, 11}) == "hello<special>", "BPE 解码错误");
  Check(tokenizer.Encode("x") == std::vector<int>({0}), "BPE 未知符号未回退到 UNK");
  Check(tokenizer.DecodeToken(-1).empty(), "非法 BPE token 应解码为空串");
}

// 测试 BPE 加载失败和工厂函数的成功、失败返回约定。
TEST(TokenizerTest, BpeFactory) {
  TemporaryDirectory directory;
  WriteBpeFiles(directory);

  std::unique_ptr<ITokenizer> valid = mini_llama::CreateBpeTokenizer(
      directory.File("vocab.json").string(), directory.File("merges.txt").string(),
      directory.File("special_tokens.json").string());
  Check(valid != nullptr && valid->Encode("hello") == std::vector<int>({10}),
        "BPE 工厂创建结果错误");

  std::unique_ptr<ITokenizer> invalid = mini_llama::CreateBpeTokenizer(
      directory.File("missing.json").string(), directory.File("merges.txt").string(),
      directory.File("special_tokens.json").string());
  Check(invalid == nullptr, "BPE 工厂加载失败时应返回 nullptr");
}

// 测试通用工厂在有效 JSON 与缺失路径之间选择正确的分词器。
TEST(TokenizerTest, GenericFactory) {
  TemporaryDirectory directory;
  const auto vocab = directory.File("vocab.json");
  WriteText(
      vocab,
      R"([{"id":0,"content":"<unk>","special":true},{"id":3,"content":"ok","special":false}])");

  auto json = mini_llama::CreateTokenizer(vocab.string());
  const auto invalid = directory.File("invalid.json");
  WriteText(invalid, "not-json");
  auto missing =
      mini_llama::CreateTokenizer(directory.File("missing.json").string());
  auto malformed = mini_llama::CreateTokenizer(invalid.string());

  Check(dynamic_cast<JsonVocabTokenizer*>(json.get()) != nullptr,
        "有效路径应创建 JSON 分词器");
  Check(dynamic_cast<AsciiTokenizer*>(missing.get()) != nullptr,
        "缺失路径应回退到 ASCII 分词器");
  Check(dynamic_cast<AsciiTokenizer*>(malformed.get()) != nullptr,
        "损坏 JSON 应回退到 ASCII 分词器");
}

// 测试 GGUF 元数据加载、BPE 合并、特殊 token、控制 ID 和解码。
TEST(TokenizerTest, GgufTokenizer) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("tokenizer.gguf");
  WriteGguf(gguf, BpeTokens(), BpeMerges());
  GgufTokenizer tokenizer;

  Check(tokenizer.Load(gguf.string()), "GGUF 加载失败");
  Check(tokenizer.vocab_size() == 12, "GGUF 词表大小错误");
  Check(
      tokenizer.bos_id() == 1 && tokenizer.eos_id() == 2 && tokenizer.unk_id() == 0,
      "GGUF 控制 token ID 错误");
  Check(tokenizer.Encode("hello<special>") == std::vector<int>({10, 11}),
        "GGUF BPE 或特殊 token 编码错误");
  Check(tokenizer.Decode({10, 11}) == "hello<special>", "GGUF 解码错误");
  Check(tokenizer.DecodeToken(99).empty(), "越界 GGUF token 应解码为空串");
}

// 测试 GGUF 工厂拒绝无效文件，并能从有效文件创建统一接口对象。
TEST(TokenizerTest, GgufFactoryAndInvalidInput) {
  TemporaryDirectory directory;
  const auto valid_path = directory.File("valid.gguf");
  const auto invalid_path = directory.File("invalid.gguf");
  WriteGguf(valid_path, BpeTokens(), BpeMerges());
  WriteText(invalid_path, "not-gguf");

  auto valid = mini_llama::CreateGgufTokenizer(valid_path.string());
  auto invalid = mini_llama::CreateGgufTokenizer(invalid_path.string());

  Check(valid != nullptr && valid->Decode(valid->Encode("hello")) == "hello",
        "GGUF 工厂创建结果错误");
  Check(invalid == nullptr, "无效 GGUF 应返回 nullptr");
}

}  // namespace
