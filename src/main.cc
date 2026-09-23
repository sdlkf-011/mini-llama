#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/model.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/tokenizer.h"

// 本模块提供命令行运行入口：generate 执行单轮生成，run 维护多轮会话
// 两种模式共享参数校验、分词、采样和前向推理流程，交互模式额外负责命令、
// 流式输出和 KV Cache 前缀复用
namespace mini_llama {

// 可注入输入流的重载供 run 模式和单元测试共用
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::istream& input, std::ostream& output,
        std::ostream& error);

namespace {

// generate 与 run 共用采样参数，run 可将 prompt 作为首轮输入
struct GenerateOptions {
  std::string prompt;
  int max_tokens = 32;
  SamplingParams sampling;
};

// 解析十进制整数选项，并拒绝部分转换和溢出等无效输入
int ParseInt(const std::string& value, const std::string& option) {
  int parsed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for " + option + ": " + value);
  }
  return parsed;
}

// 单独处理无符号随机种子，避免负值被隐式转换为大整数
unsigned int ParseSeed(const std::string& value) {
  unsigned int seed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), seed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for --seed: " + value);
  }
  return seed;
}

// 解析温度并约束为有限的非负数，避免采样阶段出现非法浮点值
float ParseTemperature(const std::string& value) {
  size_t consumed = 0;
  float temperature = 0.0f;
  try {
    temperature = std::stof(value, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid value for --temperature: " + value);
  }
  if (consumed != value.size() || !std::isfinite(temperature) ||
      temperature < 0.0f || temperature > 10000.0f) {
    throw std::invalid_argument("invalid value for --temperature: " + value);
  }
  return temperature;
}

// 解析 generate/run 共用参数；每个带值选项都在此处完成缺值检查
GenerateOptions ParseGenerateOptions(int argc, const char* const argv[]) {
  GenerateOptions options;
  for (int index = 0; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--prompt") {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for --prompt");
      }
      options.prompt = argv[index];
    } else if (option == "--max-tokens") {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for --max-tokens");
      }
      options.max_tokens = ParseInt(argv[index], option);
      if (options.max_tokens <= 0) {
        throw std::invalid_argument("--max-tokens must be positive");
      }
    } else if (option == "--temperature") {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for --temperature");
      }
      options.sampling.temperature = ParseTemperature(argv[index]);
    } else if (option == "--top-k") {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for --top-k");
      }
      options.sampling.top_k = ParseInt(argv[index], option);
      if (options.sampling.top_k < 0) {
        throw std::invalid_argument("--top-k must be non-negative");
      }
    } else if (option == "--seed") {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for --seed");
      }
      options.sampling.seed = ParseSeed(argv[index]);
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  return options;
}

// 在创建上下文前验证模型、分词器和采样参数之间的基本兼容性
void ValidateGeneration(const MiniLlamaModel& model, const ITokenizer& tokenizer,
                        int max_tokens, const SamplingParams& sampling) {
  if (!model.loaded) {
    throw std::invalid_argument("model is not loaded");
  }
  if (max_tokens <= 0) {
    throw std::invalid_argument("max_tokens must be positive");
  }
  if (!std::isfinite(sampling.temperature) || sampling.temperature < 0.0f ||
      sampling.top_k < 0) {
    throw std::invalid_argument("invalid sampling parameters");
  }
  if (tokenizer.vocab_size() > model.config.vocab_size) {
    throw std::invalid_argument("tokenizer vocabulary exceeds model vocabulary");
  }
}

// 将 prompt 编码为可前向计算的 token；空 prompt 使用 BOS 保证至少有一个输入
std::vector<int> EncodePrompt(const MiniLlamaModel& model,
                              const ITokenizer& tokenizer,
                              const std::string& prompt) {
  std::vector<int> tokens = tokenizer.Encode(prompt);
  if (tokens.empty()) {
    if (!prompt.empty()) {
      throw std::invalid_argument("tokenizer returned no tokens for prompt");
    }
    if (tokenizer.bos_id() < 0 || tokenizer.bos_id() >= model.config.vocab_size) {
      throw std::invalid_argument("empty prompt requires a valid BOS token");
    }
    tokens.push_back(tokenizer.bos_id());
  }

  if (model.config.max_seq_len <= 0 ||
      tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
    throw std::invalid_argument("prompt must be shorter than model context length");
  }
  return tokens;
}

// 计算两个 token 序列从起点开始的公共前缀长度
size_t CommonPrefixLength(const std::vector<int>& left,
                          const std::vector<int>& right) {
  const size_t limit = std::min(left.size(), right.size());
  size_t length = 0;
  while (length < limit && left[length] == right[length]) {
    ++length;
  }
  return length;
}

// 查找当前 KV Cache 可安全复用的最长前缀
// RadixTree 表示会话中曾完成前向计算的前缀；当前上下文公共前缀保证对应
size_t FindReusablePrefix(const std::vector<int>& tokens, const ChatSession& session,
                          const MiniLlamaContext& context) {
  const size_t cached_prefix = session.LongestCachedPrefix(tokens);
  const size_t context_prefix = CommonPrefixLength(tokens, context.token_history);
  return std::min(cached_prefix, context_prefix);
}

// 将上下文调整到可复用前缀末端，后续 prefill 会覆盖旧分支的 KV 数据
void PrepareContextForPrefix(size_t prefix_length, const MiniLlamaModel& model,
                             MiniLlamaContext* context) {
  if (prefix_length == 0) {
    *context = MiniLlamaContext(&model);
    return;
  }
  context->token_history.resize(prefix_length);
  context->pos = static_cast<int>(prefix_length) - 1;
}

// 处理 run 模式中的斜杠命令；返回 true 表示输入已经被命令消费
bool HandleCommand(const std::string& input, ChatSession* session,
                   MiniLlamaContext* context, const MiniLlamaModel& model,
                   const Terminal& terminal, bool* should_exit) {
  if (input.empty() || input.front() != '/') {
    return false;
  }

  if (input == "/help") {
    terminal.PrintHelp();
  } else if (input == "/clear") {
    // 清空逻辑消息和 RadixTree，并重建上下文以同步清除 KV Cache
    session->Clear();
    *context = MiniLlamaContext(&model);
    terminal.PrintMessage("Session cleared.");
  } else if (input == "/stats") {
    terminal.PrintStats(*session);
  } else if (input == "/exit") {
    *should_exit = true;
    terminal.PrintMessage("Bye.");
  } else {
    terminal.PrintMessage("Unknown command. Type /help.");
  }
  return true;
}

// 对一轮消息执行前缀复用、prefill、decode 和会话状态更新
void RunChatTurn(const std::string& input, const MiniLlamaModel& model,
                 const ITokenizer& tokenizer, const GenerateOptions& options,
                 const PromptBuilder& builder, Terminal* terminal,
                 ChatSession* session, MiniLlamaContext* context,
                 MiniSampler* sampler) {
  const auto start_time = std::chrono::steady_clock::now();
  std::vector<ChatMessage> candidate = session->messages;
  candidate.push_back({"user", input});
  const std::string prompt = builder.Build(candidate);
  const std::vector<int> tokens = EncodePrompt(model, tokenizer, prompt);

  size_t prefix_length = FindReusablePrefix(tokens, *session, *context);

  // 极端情况下 prompt 与缓存完全相同，重算最后一个 token 以取得 logits
  if (prefix_length == tokens.size() && prefix_length > 0) {
    --prefix_length;
  }

  PrepareContextForPrefix(prefix_length, model, context);
  const std::vector<int> prefill_tokens(tokens.begin() + prefix_length,
                                        tokens.end());

  // 仅对未命中的后缀 prefill，批次位置从复用前缀长度开始
  Tensor logits = ForwardBatch(
      *context, model,
      MiniBatch::FromTokens(prefill_tokens, static_cast<int>(prefix_length)));
  session->SetTokenHistory(context->token_history);
  session->RecordPrefix(context->token_history);

  terminal->PrintAssistantPrefix();
  terminal->Flush();
  std::string answer;
  int generated_tokens = 0;
  for (; generated_tokens < options.max_tokens;) {
    const int next_position = context->pos + 1;
    if (next_position >= model.config.max_seq_len) {
      break;
    }

    const int token = sampler->Sample(logits, options.sampling);
    if (token == tokenizer.eos_id()) {
      break;
    }

    const std::string token_text = tokenizer.DecodeToken(token);
    answer += token_text;
    terminal->PrintTokenText(token_text);
    terminal->Flush();
    ++generated_tokens;

    // 每个已输出 token 都立即前向计算，使下一轮 prompt 可以完整复用前缀
    logits = ForwardBatch(*context, model, MiniBatch::Single(token, next_position));
    session->AppendToken(token);
  }
  terminal->NewLine();

  session->AddMessage("user", input);
  session->AddMessage("assistant", answer);
  // 保存完整会话前缀并累计本轮指标，供 /stats 和下一轮复用使用
  session->RecordPrefix(session->token_history);
  const auto elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start_time);
  session->RecordTurn(static_cast<int>(tokens.size()), generated_tokens,
                      elapsed.count());
}

// 驱动终端交互循环；首轮可消费 --prompt，其余轮次从输入流读取
int RunInteractive(const MiniLlamaModel& model, const ITokenizer& tokenizer,
                   const GenerateOptions& options,
                   const PromptBuilder& prompt_builder, std::istream& input,
                   std::ostream& output) {
  ValidateGeneration(model, tokenizer, options.max_tokens, options.sampling);
  Terminal terminal(input, output);
  ChatSession session;
  session.sampling_params = options.sampling;
  MiniLlamaContext context(&model);
  MiniSampler sampler(options.sampling);

  std::string pending_input = options.prompt;
  bool has_pending_input = !pending_input.empty();
  bool should_exit = false;

  while (!should_exit) {
    std::string line;
    if (has_pending_input) {
      // --prompt 仅作为首轮用户输入，随后恢复正常终端读取
      line = pending_input;
      has_pending_input = false;
    } else {
      terminal.PrintUserPrompt();
      if (!terminal.ReadLine(&line)) {
        terminal.NewLine();
        break;
      }
    }

    if (line.empty()) {
      continue;
    }

    if (HandleCommand(line, &session, &context, model, terminal, &should_exit)) {
      continue;
    }

    try {
      RunChatTurn(line, model, tokenizer, options, prompt_builder, &terminal,
                  &session, &context, &sampler);
    } catch (const std::exception& exception) {
      terminal.PrintMessage(std::string("Error: ") + exception.what());
    }
  }
  return 0;
}

// 输出两个运行模式共用的命令行帮助，不承担模型加载职责
void PrintUsage(std::ostream& output) {
  output << "Usage:\n"
         << "  generate --prompt TEXT [--max-tokens N] [--temperature FLOAT] "
            "[--top-k N] [--seed N]\n"
         << "  run [--prompt TEXT] [--max-tokens N] [--temperature FLOAT] "
            "[--top-k N] [--seed N]\n";
}

}  // namespace

// 执行一次非交互生成：先 prefill prompt，再逐 token 采样和前向计算
std::string Generate(const MiniLlamaModel& model, const ITokenizer& tokenizer,
                     const std::string& prompt, int max_tokens,
                     const SamplingParams& sampling) {
  ValidateGeneration(model, tokenizer, max_tokens, sampling);
  const std::vector<int> tokens = EncodePrompt(model, tokenizer, prompt);

  MiniLlamaContext context(&model);
  Tensor logits = ForwardBatch(context, model, MiniBatch::FromTokens(tokens));
  MiniSampler sampler(sampling);
  std::string result;

  for (int generated = 0; generated < max_tokens; ++generated) {
    const int next_position = context.pos + 1;
    if (next_position >= model.config.max_seq_len) {
      break;
    }

    const int token = sampler.Sample(logits, sampling);
    if (token == tokenizer.eos_id()) {
      break;
    }

    result += tokenizer.DecodeToken(token);

    // 最后一个 token 无需再计算下一轮 logits，避免无效前向调用
    if (generated + 1 < max_tokens) {
      logits = ForwardBatch(context, model, MiniBatch::Single(token, next_position));
    }
  }
  return result;
}

// 默认运行入口使用标准输入，保持既有调用接口不变
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::ostream& output, std::ostream& error) {
  return Run(argc, argv, model, tokenizer, std::cin, output, error);
}

// 可注入流的运行入口用于交互运行和单元测试，统一处理命令行错误
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::istream& input, std::ostream& output,
        std::ostream& error) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--help") {
      PrintUsage(output);
      return 0;
    }

    std::string mode = "generate";
    int option_start = 1;
    if (argc > option_start && (std::string(argv[option_start]) == "generate" ||
                                std::string(argv[option_start]) == "run")) {
      mode = argv[option_start++];
    }

    if (argc > option_start && std::string(argv[option_start]) == "--help") {
      PrintUsage(output);
      return 0;
    }

    const GenerateOptions options =
        ParseGenerateOptions(argc - option_start, argv + option_start);
    if (mode == "run") {
      // run 将控制权交给会话循环；generate 保持一次性输出语义
      // 当前阶段使用 PromptBuilder 默认格式；后续模板来源只需在此处注入配置
      const PromptBuilder prompt_builder;
      return RunInteractive(model, tokenizer, options, prompt_builder, input,
                            output);
    }
    output << Generate(model, tokenizer, options.prompt, options.max_tokens,
                       options.sampling)
           << '\n';
    return 0;
  } catch (const std::exception& exception) {
    error << exception.what() << '\n';
    return 1;
  }
}

}  // namespace mini_llama
