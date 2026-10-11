// 测试目的：验证 GGUF 词表、BPE 编解码、特殊 token、工厂失败和模板元数据边界。
#include "mini_llama/gguf_tokenizer.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "gguf_tokenizer_fixture.h"
#include "main_test.h"
#include "mini_llama/gguf.h"
#include "mini_llama/tokenizer.h"

namespace {

using mini_llama::CreateBpeTokenizer;
using mini_llama::CreateGgufTokenizer;
using mini_llama::GgufTokenizer;

using namespace mini_llama::test::gguf_tokenizer;

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
  EXPECT_TRUE(std::filesystem::is_regular_file(directory.File("chat_template.txt")));

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

using mini_llama::CreateGgufTokenizer;
using mini_llama::GgufTokenizer;

using namespace mini_llama::test::gguf_tokenizer;

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

// 测试目的：显式 chat template 始终优先于架构 fallback，并且按原始
// 字节返回模板内容。
TEST(GgufTokenizerTest, ChatTemplateFromGgufPrefersExplicitTemplate) {
  TemporaryDirectory directory;
  const auto path = directory.File("template.gguf");
  const auto empty_path = directory.File("empty-template.gguf");
  const std::string chat_template =
      "{% for message in messages %}{{ message['content'] }}\n{% endfor %}";
  WriteChatMetadataGguf(path, {{"general.architecture", "qwen2"},
                               {"tokenizer.chat_template", chat_template}});
  WriteChatMetadataGguf(empty_path, {{"general.architecture", "qwen2"},
                                     {"tokenizer.chat_template", ""}});

  EXPECT_EQ(mini_llama::LoadChatTemplateFromGguf(path.string()), chat_template);
  EXPECT_TRUE(mini_llama::LoadChatTemplateFromGguf(empty_path.string()).empty());
}

// 测试目的：缺少 chat template 时仅 qwen2 架构启用内置标记，
// 其他架构保持空模板。
TEST(GgufTokenizerTest, MissingChatTemplateUsesQwen2ArchitectureOnly) {
  TemporaryDirectory directory;
  const auto qwen_path = directory.File("qwen.gguf");
  const auto other_path = directory.File("other.gguf");
  WriteChatMetadataGguf(qwen_path, {{"general.architecture", "qwen2"}});
  WriteChatMetadataGguf(other_path, {{"general.architecture", "llama"}});

  EXPECT_EQ(mini_llama::LoadChatTemplateFromGguf(qwen_path.string()), "qwen2");
  EXPECT_TRUE(mini_llama::LoadChatTemplateFromGguf(other_path.string()).empty());
}

// 测试目的：缺失文件和损坏 GGUF 的 metadata 读取只返回空结果，
// 不向聊天路径泄漏异常或崩溃。
TEST(GgufTokenizerTest, MalformedChatTemplateMetadataIsSafe) {
  TemporaryDirectory directory;
  const auto malformed_path = directory.File("malformed.gguf");
  WriteBytes(malformed_path, {'G', 'G', 'U', 'F', 3});

  EXPECT_NO_THROW(EXPECT_TRUE(
      mini_llama::LoadChatTemplateFromGguf(malformed_path.string()).empty()));
  EXPECT_NO_THROW(EXPECT_TRUE(
      mini_llama::LoadChatTemplateFromGguf(directory.File("missing.gguf").string())
          .empty()));
}

}  // namespace
