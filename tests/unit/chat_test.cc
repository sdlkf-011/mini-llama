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
  layer.wq = Tensor({2, 2}, 0.0f);
  layer.wq.data = {1.0f, 0.0f, 0.0f, 1.0f};
  layer.wk = layer.wq;
  layer.wv = layer.wq;
  layer.wo = layer.wq;
  layer.ffn_norm = Tensor({2}, 1.0f);
  layer.w_gate = Tensor({2, 2}, 0.0f);
  layer.w_up = Tensor({2, 2}, 0.0f);
  layer.w_down = Tensor({2, 2}, 0.0f);
  model.final_norm = Tensor({2}, 1.0f);
  model.lm_head = Tensor({3, 2}, 0.0f);
  model.lm_head.data = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
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

}  // namespace
