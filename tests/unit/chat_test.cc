#include "mini_llama/chat.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

#include "mini_llama/model.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/tensor.h"
#include "mini_llama/tokenizer.h"

namespace mini_llama {

// main.cc 的可测试入口沿用项目既有签名，不额外增加公开头文件。
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::istream& input, std::ostream& output,
        std::ostream& error);
int RunMain(int argc, const char* const argv[], std::istream& input,
            std::ostream& output, std::ostream& error);

}  // namespace mini_llama

namespace {

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
