// 测试目的：覆盖会话状态、交互命令、流式输出以及 GGUF/Jinja 聊天模板装配。
#include "mini_llama/chat.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "main_test.h"
#include "mini_llama/model.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/tensor.h"
#include "mini_llama/tokenizer.h"

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

using mini_llama::ChatMessage;
using mini_llama::ChatSession;
using mini_llama::ITokenizer;
using mini_llama::MiniLlamaModel;
using mini_llama::PromptBuilder;
using mini_llama::Tensor;

class FixedTokenizer : public ITokenizer {
 public:
  std::vector<int> Encode(const std::string& text) const override {
    return text.empty() ? std::vector<int>() : std::vector<int>({0});
  }

  std::string DecodeToken(int token) const override { return token == 2 ? "x" : ""; }

  std::string Decode(const std::vector<int>& tokens) const override {
    std::string result;
    for (int token : tokens) {
      result += DecodeToken(token);
    }
    return result;
  }

  int vocab_size() const override { return 3; }
  int bos_id() const override { return 0; }
  int eos_id() const override { return 1; }
  int unk_id() const override { return 0; }
};

class CountingBuffer : public std::stringbuf {
 public:
  int sync() override {
    ++flush_count;
    return std::stringbuf::sync();
  }

  int flush_count = 0;
};

MiniLlamaModel MakeTestModel() {
  MiniLlamaModel model;
  model.config.vocab_size = 3;
  model.config.dim = 2;
  model.config.hidden_dim = 2;
  model.config.n_layers = 1;
  model.config.n_heads = 1;
  model.config.n_kv_heads = 1;
  model.config.head_dim = 2;
  model.config.max_seq_len = 16;
  model.config.rms_norm_eps = 0.0f;
  model.loaded = true;

  model.token_embedding = Tensor({3, 2}, 0.0f);
  model.token_embedding.data = {3.0f, 4.0f, 4.0f, 3.0f, 1.0f, -1.0f};
  model.layers.resize(1);
  auto& layer = model.layers[0];
  layer.attention_norm = Tensor({2}, 1.0f);
  // 测试模型仍使用精确 F32 数值，但通过 kF32 分支匹配生产模型的权重类型。
  layer.wq = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));
  layer.wq.f32_data = {1.0f, 0.0f, 0.0f, 1.0f};
  layer.wk = layer.wq;
  layer.wv = layer.wq;
  layer.wo = layer.wq;
  layer.ffn_norm = Tensor({2}, 1.0f);
  layer.w_gate = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));
  layer.w_up = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));
  layer.w_down = mini_llama::ToQuantizedTensor(Tensor({2, 2}, 0.0f));
  model.final_norm = Tensor({2}, 1.0f);
  model.lm_head = mini_llama::ToQuantizedTensor(Tensor({3, 2}, 0.0f));
  model.lm_head.f32_data = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  return model;
}

// 测试目的：验证会话可以维护消息、已评估 token、前缀缓存和累计统计，
// 并由 /clear 所需的 Clear 一次性恢复初始状态。
TEST(ChatTest, SessionTracksAndClearsState) {
  ChatSession session;
  session.AddMessage("user", "hello");
  session.SetTokenHistory({1, 2});
  session.AppendToken(3);
  session.RecordPrefix({1, 2, 3});
  session.RecordTurn(3, 1, 5.0);

  ASSERT_EQ(session.messages.size(), 1u);
  EXPECT_EQ(session.messages[0].content, "hello");
  EXPECT_EQ(session.token_history, std::vector<int>({1, 2, 3}));
  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3, 4}), 3u);
  EXPECT_EQ(session.total_prompt_tokens, 3);
  EXPECT_EQ(session.total_generated_tokens, 1);

  session.Clear();
  EXPECT_TRUE(session.messages.empty());
  EXPECT_TRUE(session.token_history.empty());
  EXPECT_EQ(session.LongestCachedPrefix({1, 2, 3}), 0u);
  EXPECT_EQ(session.total_prompt_tokens, 0);
  EXPECT_EQ(session.total_generated_tokens, 0);
}

// 测试目的：验证默认纯文本与 Qwen2 模板都保留消息顺序，并在末尾添加
// assistant 生成前缀。
TEST(ChatTest, PromptBuilderFormatsPlainAndQwenPrompts) {
  const std::vector<ChatMessage> messages = {{"system", "be concise"},
                                             {"user", "hello"}};
  PromptBuilder builder;

  EXPECT_EQ(builder.Build(messages), "System: be concise\nUser: hello\nAssistant: ");

  builder.SetChatTemplate("qwen2");
  EXPECT_EQ(builder.Build(messages),
            "<|im_start|>system\nbe concise<|im_end|>\n"
            "<|im_start|>user\nhello<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// 测试目的：纯文本模式在空历史、多轮、空内容和未知角色下
// 仍产生字节级稳定的 assistant 生成前缀。
TEST(ChatTest, PromptBuilderPlainModeHandlesBoundaries) {
  PromptBuilder builder;
  EXPECT_EQ(builder.Build({}), "Assistant: ");

  const std::vector<ChatMessage> messages = {{"system", "rules"},
                                             {"user", ""},
                                             {"tool", "ignored"},
                                             {"assistant", "answer"},
                                             {"user", "again"}};
  EXPECT_EQ(builder.Build(messages),
            "System: rules\n"
            "User: \n"
            "Assistant: answer\n"
            "User: again\n"
            "Assistant: ");
}

// 测试目的：内置 Qwen2 仅在第一条是 system 时保留调用方
// 设定；第一条非 system 时即使后续有 system 也必须注入默认设定。
TEST(ChatTest, PromptBuilderQwen2UsesFirstMessageForSystemInjection) {
  PromptBuilder builder;
  builder.SetChatTemplate("qwen2");

  EXPECT_EQ(builder.Build({}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>assistant\n");

  EXPECT_EQ(builder.Build({{"user", "你好"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "你好<|im_end|>\n"
            "<|im_start|>assistant\n");

  EXPECT_EQ(builder.Build({{"system", "custom"}, {"user", "hello"}}),
            "<|im_start|>system\n"
            "custom<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");

  EXPECT_EQ(builder.Build({{"user", "first"}, {"system", "late"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "first<|im_end|>\n"
            "<|im_start|>system\n"
            "late<|im_end|>\n"
            "<|im_start|>assistant\n");

  EXPECT_EQ(builder.Build({{"tool", "ignored"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>assistant\n");
}

// 测试目的：验证轻量模板执行器支持消息循环、条件、逻辑运算、消息下标、
// loop.first、add_generation_prompt、字符串拼接和 trim marker。
TEST(ChatTest, PromptBuilderExecutesJinjaSubset) {
  const std::vector<ChatMessage> messages = {
      {"system", "brief"}, {"user", "hello"}, {"assistant", "hi"}};
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{%- for message in messages -%}"
      "{% if loop.first and message['role'] == 'system' %}"
      "{{- 'first=' + message['content'] + ';' -}}"
      "{% endif %}"
      "{% if message['role'] != 'system' or loop.first %}"
      "{{ message['role'] + ':' + message['content'] + ';' }}"
      "{% endif %}"
      "{%- endfor -%}"
      "{% if add_generation_prompt %}assistant;{% endif %}"
      "{{ messages[1]['role'] + '=' + messages[1]['content'] }}");

  EXPECT_EQ(builder.Build(messages),
            "first=brief;system:brief;user:hello;assistant:hi;"
            "assistant;user=hello");
}

// 测试目的：验证 GGUF 中常见的 Qwen2 Jinja 结构、双引号字面量、
// 转义换行和带前置文本的模板都按预期执行。
TEST(ChatTest, PromptBuilderExecutesQwen2StyleJinjaExactly) {
  PromptBuilder builder;
  builder.SetChatTemplate(
      "{% for message in messages %}"
      "{% if loop.first and messages[0][\"role\"] != \"system\" %}"
      "{{ \"<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n\" }}"
      "{% endif %}"
      "{{ '<|im_start|>' + message[\"role\"] + '\\n' + "
      "message[\"content\"] + '<|im_end|>\\n' }}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}");

  EXPECT_EQ(builder.Build({{"user", "hello"}}),
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "hello<|im_end|>\n"
            "<|im_start|>assistant\n");

  builder.SetChatTemplate(
      "prefix:{% if add_generation_prompt or messages[0]['role'] == 'system' "
      "and messages[0]['content'] == 'no' %}ok{% endif %}");
  EXPECT_EQ(builder.Build({{"user", "hello"}}), "prefix:ok");
}

// 测试目的：验证未知模板回退纯文本，且缺失闭合标签、非法下标等异常语法
// 只产生可控结果，不发生越界访问或崩溃。
TEST(ChatTest, PromptBuilderSafelyHandlesUnsupportedAndMalformedTemplates) {
  const std::vector<ChatMessage> messages = {{"user", "hello"}};
  PromptBuilder builder;
  builder.SetChatTemplate("unsupported-template");
  EXPECT_EQ(builder.Build(messages), "User: hello\nAssistant: ");

  builder.SetChatTemplate(
      "{% if add_generation_prompt %}ok:{{ messages[bad]['role'] }}");
  EXPECT_NO_THROW(EXPECT_EQ(builder.Build(messages), "ok:"));

  builder.SetChatTemplate("{% for message in messages %}{{ message['content'] }}");
  EXPECT_NO_THROW(EXPECT_EQ(builder.Build(messages), "hello"));

  builder.SetChatTemplate("{{ messages[0]['role'] }}");
  EXPECT_NO_THROW(EXPECT_EQ(builder.Build({}), ""));

  builder.SetChatTemplate("prefix{{");
  EXPECT_NO_THROW(EXPECT_EQ(builder.Build(messages), "prefix"));
}

// 测试目的：验证 run 模式识别 /help、/stats、/clear、/exit，且 clear 后
// 会话统计归零，不把命令作为用户消息送入模型。
TEST(ChatTest, RunModeHandlesInteractiveCommands) {
  MiniLlamaModel model = MakeTestModel();
  FixedTokenizer tokenizer;
  std::istringstream input("hello\n/stats\n/clear\n/stats\n/help\n/exit\n");
  std::ostringstream output;
  std::ostringstream error;
  const char* argv[] = {"mini-llama", "run", "--max-tokens", "1"};

  EXPECT_EQ(mini_llama::Run(4, argv, model, tokenizer, input, output, error), 0);
  EXPECT_TRUE(error.str().empty());
  EXPECT_NE(output.str().find("Messages: 2"), std::string::npos);
  EXPECT_NE(output.str().find("Context tokens: 2"), std::string::npos);
  EXPECT_NE(output.str().find("Generated tokens: 1"), std::string::npos);
  EXPECT_NE(output.str().find("Session cleared."), std::string::npos);
  EXPECT_NE(output.str().find("Generated tokens: 0"), std::string::npos);
  EXPECT_NE(output.str().find("/help"), std::string::npos);
  EXPECT_NE(output.str().find("Bye."), std::string::npos);
}

// 测试目的：验证 decode 结果按 token 立即写入并刷新输出流，保证终端在
// 一轮生成结束前即可看到内容。
TEST(ChatTest, RunModeStreamsAndFlushesGeneratedTokens) {
  MiniLlamaModel model = MakeTestModel();
  FixedTokenizer tokenizer;
  std::istringstream input("hello\n/exit\n");
  CountingBuffer buffer;
  std::ostream output(&buffer);
  std::ostringstream error;
  const char* argv[] = {"mini-llama", "run", "--max-tokens", "1"};

  EXPECT_EQ(mini_llama::Run(4, argv, model, tokenizer, input, output, error), 0);
  EXPECT_TRUE(error.str().empty());
  EXPECT_NE(buffer.str().find("assistant > x"), std::string::npos);
  EXPECT_GE(buffer.flush_count, 3);
}

// 测试目的：prompt 恰好占满上下文时应在 prefill 前拒绝该轮输入，
// 不输出空的 assistant 回复，也不把失败轮次计入会话统计。
TEST(ChatTest, RunModeRejectsPromptThatFillsContext) {
  MiniLlamaModel model = MakeTestModel();
  model.config.max_seq_len = 1;
  FixedTokenizer tokenizer;
  std::istringstream input("hello\n/stats\n/exit\n");
  std::ostringstream output;
  std::ostringstream error;
  const char* argv[] = {"mini-llama", "run", "--max-tokens", "1"};

  EXPECT_EQ(mini_llama::Run(4, argv, model, tokenizer, input, output, error), 0);
  EXPECT_TRUE(error.str().empty());
  EXPECT_NE(output.str().find("prompt must be shorter than model context length"),
            std::string::npos);
  EXPECT_NE(output.str().find("Prompt tokens: 0"), std::string::npos);
  EXPECT_NE(output.str().find("Generated tokens: 0"), std::string::npos);
  EXPECT_EQ(output.str().find("assistant >"), std::string::npos);
}

// 测试目的：run 的位置模型参数应在资源装配阶段被识别，而不是误传给生成参数
// 解析器；即使路径不存在，trace 也必须以明确的 error 状态收尾。
TEST(ChatTest, RunMainAcceptsPositionalModelPathAndTracesLoadFailure) {
  std::istringstream input;
  std::ostringstream output;
  std::ostringstream error;
  const char* argv[] = {"mini-llama", "run", "missing-model-directory", "-n", "1"};

  EXPECT_EQ(mini_llama::RunMain(5, argv, input, output, error), 1);
  EXPECT_TRUE(output.str().empty());
  EXPECT_NE(error.str().find("model path does not exist: missing-model-directory"),
            std::string::npos);
  EXPECT_NE(error.str().find("mode=run status=error"), std::string::npos);
  EXPECT_NE(error.str().find("model=missing-model-directory"), std::string::npos);
}

}  // namespace
