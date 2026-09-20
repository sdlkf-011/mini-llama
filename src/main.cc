#include <charconv>
#include <cmath>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/sampler.h"
#include "mini_llama/tokenizer.h"

namespace mini_llama {
namespace {

// 当前阶段的命令行选项只覆盖一次生成所需的输入和采样参数。
// Chat、PromptBuilder 以及模型加载参数由后续阶段扩展，不在这里写死。
struct GenerateOptions {
  std::string prompt;
  int max_tokens = 32;
  SamplingParams sampling;
};

int ParseInt(const std::string& value, const std::string& option) {
  int parsed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for " + option + ": " + value);
  }
  return parsed;
}

unsigned int ParseSeed(const std::string& value) {
  unsigned int seed = 0;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), seed);
  if (result.ec != std::errc() || result.ptr != value.data() + value.size()) {
    throw std::invalid_argument("invalid value for --seed: " + value);
  }
  return seed;
}

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

}  // namespace

// 解析 generate/run 共用的命令行选项，并将采样相关值集中到
// SamplingParams，保证后续不同运行模式使用同一套参数语义。
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

// 使用现有 ForwardBatch 完成 prompt prefill，再用 MiniSampler 逐 token
// 解码。模型和 tokenizer 由调用方提供，main 不负责猜测或加载固定路径。
std::string Generate(const MiniLlamaModel& model, const ITokenizer& tokenizer,
                     const std::string& prompt, int max_tokens,
                     const SamplingParams& sampling) {
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
      tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
    throw std::invalid_argument("prompt exceeds model context length");
  }

  MiniLlamaContext context(&model);
  Tensor logits = ForwardBatch(context, model, MiniBatch::FromTokens(tokens));
  MiniSampler sampler(sampling);
  std::string result;
  for (int generated = 0; generated < max_tokens; ++generated) {
    const int token = sampler.Sample(logits, sampling);
    if (token == tokenizer.eos_id()) {
      break;
    }
    result += tokenizer.DecodeToken(token);
    if (generated + 1 == max_tokens || context.pos + 1 >= model.config.max_seq_len) {
      break;
    }
    logits = ForwardBatch(context, model, MiniBatch::Single(token, context.pos + 1));
  }
  return result;
}

// 当前阶段的统一入口：识别 generate/run 模式并把解析后的参数传入生成
// 流程。后续接入 Chat 后，只需替换对应模式的执行部分。
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::ostream& output, std::ostream& error) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--help") {
      output << "Usage: [generate|run] --prompt TEXT [--max-tokens N] "
                "[--temperature FLOAT] [--top-k N] [--seed N]\n";
      return 0;
    }
    int option_start = 1;
    if (argc > option_start && (std::string(argv[option_start]) == "generate" ||
                                std::string(argv[option_start]) == "run")) {
      ++option_start;
    }
    if (argc > option_start && std::string(argv[option_start]) == "--help") {
      output << "Usage: [generate|run] --prompt TEXT [--max-tokens N] "
                "[--temperature FLOAT] [--top-k N] [--seed N]\n";
      return 0;
    }
    const GenerateOptions options =
        ParseGenerateOptions(argc - option_start, argv + option_start);
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
