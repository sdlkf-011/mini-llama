#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/prompt_builder.h"

namespace mini_llama {

// main.cc 中的资源装配辅助函数没有独立公开头文件。
std::string LoadChatTemplateFromGguf(const std::string& gguf_path);

}  // namespace mini_llama

namespace {

using mini_llama::ChatMessage;
using mini_llama::PromptBuilder;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("mini_llama_chat_template_" + std::to_string(unique));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

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

void WriteMetadataGguf(
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

  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_TRUE(output);
}

PromptBuilder LoadBuilder(const std::filesystem::path& gguf) {
  PromptBuilder builder;
  builder.SetChatTemplate(mini_llama::LoadChatTemplateFromGguf(gguf.string()));
  return builder;
}

const std::string& Qwen2JinjaTemplate() {
  static const std::string chat_template =
      "{% for message in messages %}"
      "{% if loop.first and messages[0]['role'] != 'system' %}"
      "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
      "{% endif %}"
      "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
      "'<|im_end|>\\n' }}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
  return chat_template;
}

// 测试目的：验证 GGUF 原始 Qwen2 Jinja 模板从 metadata 加载后，
// 能完整执行默认 system 注入与 assistant generation prefix。
TEST(ChatTemplateIntegrationTest, ChatTemplateFromGgufQwen2) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("qwen-template.gguf");
  WriteMetadataGguf(gguf, {{"general.architecture", "qwen2"},
                           {"tokenizer.chat_template", Qwen2JinjaTemplate()}});

  const PromptBuilder builder = LoadBuilder(gguf);
  EXPECT_EQ(builder.Build({{"user", "你好"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "你好<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// 测试目的：验证调用方已提供 system 时保留原内容，不额外注入
// 默认 system。
TEST(ChatTemplateIntegrationTest, ChatTemplateKeepsSystemMessage) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("system-template.gguf");
  WriteMetadataGguf(gguf, {{"tokenizer.chat_template", Qwen2JinjaTemplate()}});

  const PromptBuilder builder = LoadBuilder(gguf);
  EXPECT_EQ(builder.Build({{"system", "Be concise."}, {"user", "hello"}}),
            "<|im_start|>system\n"
            "Be concise.<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// 测试目的：缺少模板时只有 qwen2 架构使用内置 fallback，其他
// 架构使用纯文本 prompt。
TEST(ChatTemplateIntegrationTest, MissingTemplateUsesArchitectureFallbackOnly) {
  TemporaryDirectory directory;
  const auto qwen_gguf = directory.File("qwen-fallback.gguf");
  const auto llama_gguf = directory.File("llama-fallback.gguf");
  WriteMetadataGguf(qwen_gguf, {{"general.architecture", "qwen2"}});
  WriteMetadataGguf(llama_gguf, {{"general.architecture", "llama"}});

  EXPECT_EQ(LoadBuilder(qwen_gguf).Build({{"user", "hello"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");
  EXPECT_EQ(LoadBuilder(llama_gguf).Build({{"user", "hello"}}),
            "User: hello\nAssistant: ");
}

// 测试目的：未闭合的 Jinja 控制块和空历史下标访问不会崩溃、
// 越界或死循环。
TEST(ChatTemplateIntegrationTest, MalformedTemplateIsSafe) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("malformed-template.gguf");
  WriteMetadataGguf(gguf,
                    {{"tokenizer.chat_template",
                      "{% for message in messages %}{{ message['content'] }}"}});

  const PromptBuilder builder = LoadBuilder(gguf);
  EXPECT_NO_THROW(EXPECT_EQ(builder.Build({{"user", "hello"}}), "hello"));

  PromptBuilder indexed_builder;
  indexed_builder.SetChatTemplate("{{ messages[0]['content'] }}");
  EXPECT_NO_THROW(EXPECT_EQ(indexed_builder.Build({}), ""));
}

// 测试目的：验证轻量执行器能识别 Jinja trim marker，不把标记中的
// '-' 当作语句或表达式的一部分。
TEST(ChatTemplateIntegrationTest, SupportsJinjaTrimMarkers) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("trim-template.gguf");
  WriteMetadataGguf(
      gguf,
      {{"tokenizer.chat_template",
        "{%- for message in messages -%}{{- message['content'] -}}"
        "{%- endfor -%}{%- if add_generation_prompt -%}assistant{%- endif -%}"}});

  EXPECT_EQ(LoadBuilder(gguf).Build({{"user", "one"}, {"assistant", "two"}}),
            "onetwoassistant");
}

// 测试目的：当 GGUF 没有模板和 qwen2 架构标记时，保持纯文本
// 多轮格式与未知角色过滤行为。
TEST(ChatTemplateIntegrationTest, PlainTextFallback) {
  TemporaryDirectory directory;
  const auto gguf = directory.File("plain.gguf");
  WriteMetadataGguf(gguf, {});

  const std::vector<ChatMessage> messages = {{"system", "rules"},
                                             {"user", "hello"},
                                             {"tool", "ignored"},
                                             {"assistant", "hi"},
                                             {"user", "again"}};
  EXPECT_EQ(LoadBuilder(gguf).Build(messages),
            "System: rules\n"
            "User: hello\n"
            "Assistant: hi\n"
            "User: again\n"
            "Assistant: ");
}

// 测试目的：本地存在真实 Qwen2 GGUF 时，直接验证其 metadata 中的
// 模板能产生与内置 Qwen2 相同的精确 prompt。
TEST(ChatTemplateIntegrationTest, RealQwen2TemplateWhenAvailable) {
#ifndef MINI_LLAMA_TEST_SOURCE_DIR
  GTEST_SKIP() << "MINI_LLAMA_TEST_SOURCE_DIR is not configured";
#else
  const std::filesystem::path gguf =
      std::filesystem::path(MINI_LLAMA_TEST_SOURCE_DIR) /
      "models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf";
  if (!std::filesystem::is_regular_file(gguf)) {
    GTEST_SKIP() << "real Qwen2 GGUF is not available";
  }

  EXPECT_EQ(LoadBuilder(gguf).Build({{"user", "hello"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");
#endif
}

}  // namespace
