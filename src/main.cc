#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/tokenizer.h"

// 本模块提供命令行运行入口：generate 执行单轮生成，run 维护多轮会话
// 两种模式共享参数校验、分词、采样和前向推理流程，交互模式额外负责命令、
// 流式输出和 KV Cache 前缀复用
namespace mini_llama {

// 可注入输入流的重载供 run 模式、进程装配和单元测试共用
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
    if (option == "--prompt" || option == "-p") {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for --prompt");
      }
      options.prompt = argv[index];
    } else if (option == "--max-tokens" || option == "--n-predict" ||
               option == "-n") {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for " + option);
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
         << "  mini-llama generate --model MODEL_OR_DIRECTORY --prompt TEXT "
            "[--max-tokens N|--n-predict N] [--temperature FLOAT] "
            "[--top-k N] [--seed N]\n"
         << "  mini-llama run --model MODEL_OR_DIRECTORY [--prompt TEXT] "
            "[--max-tokens N|--n-predict N] [--temperature FLOAT] "
            "[--top-k N] [--seed N]\n"
         << "  Add --tokenizer TOKENIZER_DIRECTORY_OR_VOCAB_JSON to override "
            "the GGUF tokenizer.\n"
         << "  mini-llama generate --config MODEL.json --weights MODEL.bin "
            "[--vocab VOCAB.json] --prompt TEXT [--max-tokens N]\n"
         << "  mini-llama inspect-gguf MODEL.gguf\n";
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
        std::ostream& error, const std::string& chat_template) {
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
      // run 将控制权交给会话循环；generate 保持一次性输出语义。
      PromptBuilder prompt_builder;
      prompt_builder.SetChatTemplate(chat_template);
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

int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::istream& input, std::ostream& output,
        std::ostream& error) {
  return Run(argc, argv, model, tokenizer, input, output, error, "");
}

namespace {

// 进程级参数分为资源路径和传递给 generate/run 的运行参数；后者保留原始
// argv 指针，避免再次拼装字符串存储
struct LaunchOptions {
  std::string model_path;
  std::string config_path;
  std::string weights_path;
  std::string vocab_path;
  std::string tokenizer_path;
  std::vector<const char*> forwarded_arguments;
};

// 模型、分词器和聊天模板必须来自同一次资源解析，防止运行阶段使用不匹配
// 的配置
struct LoadedResources {
  MiniLlamaModel model;
  std::unique_ptr<ITokenizer> tokenizer;
  std::string chat_template;
};

// 读取带值的路径选项，并统一拒绝重复、缺值和空值
void SetPathOption(std::string* destination, const std::string& option, int* index,
                   int argc, const char* const argv[]) {
  if (!destination->empty()) {
    throw std::invalid_argument("duplicate option: " + option);
  }
  if (++(*index) >= argc) {
    throw std::invalid_argument("missing value for " + option);
  }
  *destination = argv[*index];
  if (destination->empty()) {
    throw std::invalid_argument("empty value for " + option);
  }
}

// 提取资源加载选项；未消费的参数原样转发给 Run 解析生成配置
LaunchOptions ParseLaunchOptions(int argc, const char* const argv[]) {
  LaunchOptions options;
  options.forwarded_arguments.push_back(argv[0]);
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--model" || option == "-m") {
      SetPathOption(&options.model_path, option, &index, argc, argv);
    } else if (option == "--config") {
      SetPathOption(&options.config_path, option, &index, argc, argv);
    } else if (option == "--weights") {
      SetPathOption(&options.weights_path, option, &index, argc, argv);
    } else if (option == "--vocab") {
      SetPathOption(&options.vocab_path, option, &index, argc, argv);
    } else if (option == "--tokenizer") {
      SetPathOption(&options.tokenizer_path, option, &index, argc, argv);
    } else {
      options.forwarded_arguments.push_back(argv[index]);
    }
  }
  return options;
}

// 扩展名比较忽略大小写，仅用于判断模型候选，不代替 GGUF 内容校验
bool IsGgufPath(const std::string& path) {
  std::string extension = std::filesystem::path(path).extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return extension == ".gguf";
}

// 装配层内部根据目录或 vocab.json 提示选择现有 tokenizer 工厂，不扩展
// tokenizer 公共接口
std::unique_ptr<ITokenizer> CreateTokenizerFromVocabHint(
    const std::string& tokenizer_hint) {
  namespace fs = std::filesystem;
  if (tokenizer_hint.empty()) {
    return nullptr;
  }

  std::error_code error;
  fs::path hint(tokenizer_hint);
  fs::path vocab_path = hint;
  fs::path directory = hint.parent_path();
  if (fs::is_directory(hint, error)) {
    directory = hint;
    vocab_path = directory / "vocab.json";
  } else if (error) {
    return nullptr;
  }
  if (!fs::is_regular_file(vocab_path, error) || error) {
    return nullptr;
  }

  const fs::path merges_path = directory / "merges.txt";
  if (fs::is_regular_file(merges_path, error) && !error) {
    return CreateBpeTokenizer(vocab_path.string(), merges_path.string(),
                              (directory / "special_tokens.json").string());
  }
  try {
    return std::make_unique<JsonVocabTokenizer>(vocab_path.string());
  } catch (const std::exception&) {
    return nullptr;
  }
}

// chat template 只参与命令行资源装配，直接复用已有 GGUF metadata 接口
std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  std::string chat_template;
  if (!GgufGetMetadataString(gguf_path, "tokenizer.chat_template", chat_template)) {
    return {};
  }
  return chat_template;
}

std::filesystem::path ResolveGgufPath(const std::string& model_hint) {
  // --model 既可以指向单个 GGUF，也可以指向只包含一个 GGUF 的目录
  namespace fs = std::filesystem;
  std::error_code error;
  const fs::path hint(model_hint);
  if (fs::is_regular_file(hint, error)) {
    if (!IsGgufPath(hint.string())) {
      throw std::invalid_argument("--model file must have a .gguf extension");
    }
    return hint;
  }
  if (error || !fs::is_directory(hint, error) || error) {
    throw std::invalid_argument("model path does not exist: " + model_hint);
  }

  std::vector<fs::path> candidates;
  // 目录中必须只有一个明确候选，避免依赖文件系统遍历顺序选择模型
  for (fs::directory_iterator it(hint, error), end; !error && it != end;
       it.increment(error)) {
    if (it->is_regular_file(error) && !error && IsGgufPath(it->path().string())) {
      candidates.push_back(it->path());
    }
  }
  if (error) {
    throw std::runtime_error("failed to inspect model directory: " + model_hint);
  }
  std::sort(candidates.begin(), candidates.end());
  if (candidates.empty()) {
    throw std::runtime_error("model directory contains no .gguf file: " +
                             model_hint);
  }
  if (candidates.size() > 1) {
    throw std::runtime_error("model directory contains multiple .gguf files: " +
                             model_hint);
  }
  return candidates.front();
}

// 按命令行模式装配模型与 tokenizer：GGUF 优先读取内嵌 metadata，旧格式
// 继续使用 manifest；显式 tokenizer 始终覆盖自动选择结果
LoadedResources LoadModelAndTokenizer(const LaunchOptions& options) {
  LoadedResources resources;
  if (!options.model_path.empty()) {
    // GGUF 路径不能与旧版 config/weights/vocab 参数混用
    if (!options.config_path.empty() || !options.weights_path.empty() ||
        !options.vocab_path.empty()) {
      throw std::invalid_argument(
          "--model cannot be combined with --config, --weights, or --vocab");
    }
    const std::filesystem::path model_path = ResolveGgufPath(options.model_path);
    resources.model = LoadGgufModel(model_path.string());
    if (!resources.model.loaded) {
      throw std::runtime_error("failed to load GGUF model: " +
                               resources.model.load_error);
    }
    // 用户显式指定 tokenizer 时不读取 GGUF 内嵌 tokenizer metadata
    if (!options.tokenizer_path.empty()) {
      resources.tokenizer = CreateTokenizerFromVocabHint(options.tokenizer_path);
      if (!resources.tokenizer) {
        throw std::runtime_error("failed to load explicit tokenizer: " +
                                 options.tokenizer_path);
      }
    } else {
      resources.tokenizer = CreateGgufTokenizer(model_path.string());
    }
    // 内嵌 metadata 缺失时，尝试同目录的 vocab.json/merges.txt 导出文件
    if (!resources.tokenizer) {
      const std::filesystem::path tokenizer_directory = model_path.parent_path();
      resources.tokenizer =
          CreateTokenizerFromVocabHint(tokenizer_directory.string());
    }
    if (!resources.tokenizer) {
      throw std::runtime_error(
          "GGUF has no tokenizer metadata and no vocab.json/merges.txt fallback");
    }
    // chat template 独立于 tokenizer 来源，始终从最终选定的 GGUF 读取
    resources.chat_template = LoadChatTemplateFromGguf(model_path.string());
    return resources;
  }

  if (options.config_path.empty() || options.weights_path.empty()) {
    throw std::invalid_argument(
        "provide --model MODEL.gguf or both --config MODEL.json and --weights "
        "MODEL.bin");
  }

  const ModelManifest manifest = ParseManifest(options.config_path);
  resources.model = LoadModel(options.config_path, options.weights_path);
  if (!resources.model.loaded) {
    throw std::runtime_error("failed to load model: " + resources.model.load_error);
  }

  if (!options.tokenizer_path.empty()) {
    // 旧格式同样允许显式 tokenizer 覆盖 manifest 中的配置
    resources.tokenizer = CreateTokenizerFromVocabHint(options.tokenizer_path);
    if (!resources.tokenizer) {
      throw std::runtime_error("failed to load explicit tokenizer: " +
                               options.tokenizer_path);
    }
    return resources;
  }

  if (manifest.tokenizer.type == "ascii" && options.vocab_path.empty()) {
    resources.tokenizer = std::make_unique<AsciiTokenizer>();
    return resources;
  }

  std::filesystem::path vocab_path = options.vocab_path;
  // 未显式提供 --vocab 时，相对于 manifest 所在目录解析词表路径
  if (vocab_path.empty()) {
    vocab_path = std::filesystem::path(options.config_path).parent_path() /
                 manifest.tokenizer.path;
  }
  if (!std::filesystem::is_regular_file(vocab_path)) {
    throw std::runtime_error("tokenizer vocabulary does not exist: " +
                             vocab_path.string());
  }
  resources.tokenizer = CreateTokenizer(vocab_path.string());
  return resources;
}

// 帮助选项在加载大型模型前处理，避免仅查看用法时产生昂贵 I/O
bool HasHelpOption(int argc, const char* const argv[]) {
  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) == "--help" || std::string(argv[index]) == "-h") {
      return true;
    }
  }
  return false;
}

}  // namespace

// 进程级入口负责帮助、GGUF 检查和资源装配，再将剩余参数交给运行层
int RunMain(int argc, const char* const argv[], std::istream& input,
            std::ostream& output, std::ostream& error) {
  try {
    if (argc <= 1 || HasHelpOption(argc, argv)) {
      PrintUsage(output);
      return 0;
    }

    // inspect-gguf 是独立的只读命令，不需要加载模型权重和 tokenizer
    if (std::string(argv[1]) == "inspect-gguf") {
      if (argc != 3) {
        throw std::invalid_argument(
            "inspect-gguf requires exactly one GGUF file path");
      }
      GgufReader reader;
      if (!reader.Load(argv[2])) {
        throw std::runtime_error("failed to inspect GGUF model: " +
                                 reader.load_error);
      }
      InspectGguf(reader);
      return 0;
    }

    const LaunchOptions options = ParseLaunchOptions(argc, argv);
    LoadedResources resources = LoadModelAndTokenizer(options);
    return Run(static_cast<int>(options.forwarded_arguments.size()),
               options.forwarded_arguments.data(), resources.model,
               *resources.tokenizer, input, output, error, resources.chat_template);
  } catch (const std::exception& exception) {
    error << exception.what() << '\n';
    return 1;
  }
}

// 默认入口绑定标准输入输出；可注入流的重载保留给测试和嵌入调用
int RunMain(int argc, const char* const argv[]) {
  return RunMain(argc, argv, std::cin, std::cout, std::cerr);
}

}  // namespace mini_llama

#ifdef MINI_LLAMA_BUILD_EXECUTABLE
// 复用本文件已有的 RunMain 装配逻辑，避免为进程入口增加独立源文件。
int main(int argc, char* argv[]) { return mini_llama::RunMain(argc, argv); }
#endif
