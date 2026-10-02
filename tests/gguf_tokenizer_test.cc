#include "mini_llama/gguf_tokenizer.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"
#include "mini_llama/tokenizer.h"

namespace {

using mini_llama::CreateBpeTokenizer;
using mini_llama::CreateGgufTokenizer;
using mini_llama::GgufTokenizer;

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
  AppendString(output, key);
  AppendScalar<uint32_t>(output, 8);
  AppendString(output, value);
}

void AppendStringArrayMetadata(std::vector<uint8_t>* output, const std::string& key,
                               const std::vector<std::string>& values) {
  AppendString(output, key);
  AppendScalar<uint32_t>(output, 9);
  AppendScalar<uint32_t>(output, 8);
  AppendScalar<uint64_t>(output, values.size());
  for (const std::string& value : values) {
    AppendString(output, value);
  }
}

void AppendIntegerMetadata(std::vector<uint8_t>* output, const std::string& key,
                           int32_t value) {
  AppendString(output, key);
  AppendScalar<uint32_t>(output, 5);
  AppendScalar<int32_t>(output, value);
}

const std::vector<std::string>& TinyTokens() {
  static const std::vector<std::string> tokens = {
      "<unk>", "<s>", "</s>", "h",     "e",           "l",     "o",
      "he",    "hel", "hell", "hello", "<|special|>", "Ġ",     "w",
      "r",     "d",   "Ġw",   "Ġwo",   "Ġwor",        "Ġworl", "Ġworld"};
  return tokens;
}

const std::vector<std::string>& TinyMerges() {
  static const std::vector<std::string> merges = {
      "h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d"};
  return merges;
}

void WriteTinyTokenizerGguf(const std::filesystem::path& path) {
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

void WriteTinyBpeFiles(const std::filesystem::path& directory) {
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

std::string ShellQuote(const std::filesystem::path& path) {
  std::string quoted = "'";
  for (char character : path.string()) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  return quoted + "'";
}

// 测试目的：验证 GgufTokenizer 能从 GGUF metadata 加载词表、控制 token
// ID 和 chat template。
TEST(GgufTokenizerIntegrationTest, gguf_tokenizer_loads_from_gguf) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("tiny.gguf");
  WriteTinyTokenizerGguf(gguf);

  GgufTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Load(gguf.string()));
  EXPECT_EQ(tokenizer.vocab_size(), 21);
  EXPECT_EQ(tokenizer.bos_id(), 1);
  EXPECT_EQ(tokenizer.eos_id(), 2);
  EXPECT_EQ(tokenizer.unk_id(), 0);
  std::string chat_template;
  EXPECT_TRUE(mini_llama::GgufGetMetadataString(
      gguf.string(), "tokenizer.chat_template", chat_template));
  EXPECT_FALSE(chat_template.empty());
}

// 测试目的：验证 GGUF 中的 BPE 合并规则可以正确编码文本，并能无损解码
// 回原始字符串。
TEST(GgufTokenizerIntegrationTest, gguf_tokenizer_encode_decode_roundtrip) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("tiny.gguf");
  WriteTinyTokenizerGguf(gguf);
  GgufTokenizer tokenizer;
  ASSERT_TRUE(tokenizer.Load(gguf.string()));

  const std::string text = "hello world";
  EXPECT_EQ(tokenizer.Encode(text), std::vector<int>({10, 20}));
  EXPECT_EQ(tokenizer.Decode(tokenizer.Encode(text)), text);
}

// 测试目的：验证特殊 token 会被整体识别，不会参与普通 BPE 拆分，并能正确
// 解码。
TEST(GgufTokenizerIntegrationTest, gguf_tokenizer_special_tokens) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("tiny.gguf");
  WriteTinyTokenizerGguf(gguf);
  auto tokenizer = CreateGgufTokenizer(gguf.string());
  ASSERT_NE(tokenizer, nullptr);

  EXPECT_EQ(tokenizer->Encode("hello<|special|>"), std::vector<int>({10, 11}));
  EXPECT_EQ(tokenizer->Decode({11}), "<|special|>");
}

// 测试目的：验证 GGUF 文件不存在时工厂函数安全返回空指针，chat template
// 读取也明确失败。
TEST(GgufTokenizerIntegrationTest, gguf_tokenizer_rejects_missing_file) {
  TemporaryDirectory directory;
  EXPECT_EQ(CreateGgufTokenizer(directory.File("missing.gguf").string()), nullptr);
  std::string chat_template;
  EXPECT_FALSE(
      mini_llama::GgufGetMetadataString(directory.File("missing.gguf").string(),
                                        "tokenizer.chat_template", chat_template));
}

// 测试目的：验证直接读取 GGUF metadata 与导出文件后使用 BpeTokenizer 的
// 编码和解码结果一致。
TEST(GgufTokenizerIntegrationTest, gguf_tokenizer_matches_bpe_tokenizer) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("tiny.gguf");
  WriteTinyTokenizerGguf(gguf);
  WriteTinyBpeFiles(directory.path());
  auto gguf_tokenizer = CreateGgufTokenizer(gguf.string());
  auto bpe_tokenizer = CreateBpeTokenizer(
      directory.File("vocab.json").string(), directory.File("merges.txt").string(),
      directory.File("special_tokens.json").string());
  ASSERT_NE(gguf_tokenizer, nullptr);
  ASSERT_NE(bpe_tokenizer, nullptr);

  for (const std::string& text :
       {"hello", "hello world", "hello<|special|> world"}) {
    EXPECT_EQ(gguf_tokenizer->Encode(text), bpe_tokenizer->Encode(text)) << text;
    EXPECT_EQ(gguf_tokenizer->Decode(gguf_tokenizer->Encode(text)),
              bpe_tokenizer->Decode(bpe_tokenizer->Encode(text)))
        << text;
  }
  EXPECT_EQ(bpe_tokenizer->DecodeToken(20), " world");

  const std::filesystem::path source_dir(MINI_LLAMA_TEST_SOURCE_DIR);
  const std::filesystem::path real_gguf =
      source_dir / "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::is_regular_file(real_gguf)) {
    return;
  }
  const std::string command =
      "python3 " + ShellQuote(source_dir / "scripts/export_gguf_tokenizer.py") +
      " " + ShellQuote(real_gguf) + " " + ShellQuote(directory.path());
  ASSERT_EQ(std::system(command.c_str()), 0);
  EXPECT_TRUE(std::filesystem::is_regular_file(directory.File("vocab.json")));
  EXPECT_TRUE(std::filesystem::is_regular_file(directory.File("merges.txt")));
  EXPECT_TRUE(
      std::filesystem::is_regular_file(directory.File("special_tokens.json")));
  EXPECT_TRUE(
      std::filesystem::is_regular_file(directory.File("chat_template.txt")));

  gguf_tokenizer = CreateGgufTokenizer(real_gguf.string());
  bpe_tokenizer = CreateBpeTokenizer(directory.File("vocab.json").string(),
                                     directory.File("merges.txt").string(),
                                     directory.File("special_tokens.json").string());
  ASSERT_NE(gguf_tokenizer, nullptr);
  ASSERT_NE(bpe_tokenizer, nullptr);
  for (const std::string& text : {"Hello, world!", "mini-llama tokenizer test",
                                  "<|im_start|>user\nHello<|im_end|>\n"}) {
    EXPECT_EQ(gguf_tokenizer->Encode(text), bpe_tokenizer->Encode(text)) << text;
  }
}

// 测试目的：本地存在真实 Qwen2 GGUF 时，验证词表大小、BOS/EOS、特殊 token
// ID 和 chat template 与模型 metadata 一致。
TEST(GgufTokenizerIntegrationTest, gguf_tokenizer_loads_real_model_when_available) {
  const std::filesystem::path gguf =
      std::filesystem::path(MINI_LLAMA_TEST_SOURCE_DIR) /
      "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::is_regular_file(gguf)) {
    GTEST_SKIP() << "real Qwen2 GGUF is not available";
  }

  auto tokenizer = CreateGgufTokenizer(gguf.string());
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_EQ(tokenizer->vocab_size(), 151936);
  EXPECT_EQ(tokenizer->bos_id(), 151643);
  EXPECT_EQ(tokenizer->eos_id(), 151645);
  EXPECT_EQ(tokenizer->Encode("<|im_start|>"), std::vector<int>({151644}));
  EXPECT_EQ(tokenizer->Encode("<|im_end|>"), std::vector<int>({151645}));
  std::string chat_template;
  EXPECT_TRUE(mini_llama::GgufGetMetadataString(
      gguf.string(), "tokenizer.chat_template", chat_template));
  EXPECT_FALSE(chat_template.empty());
}

}  // namespace
