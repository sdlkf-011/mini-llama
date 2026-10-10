#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/backend.h"
#include "mini_llama/batch.h"
#include "mini_llama/chat.h"
#include "mini_llama/context.h"
#include "mini_llama/cuda_runtime.h"
#include "mini_llama/debug.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/gguf_loader.h"
#include "mini_llama/gguf_tokenizer.h"
#include "mini_llama/loader.h"
#include "mini_llama/model.h"
#include "mini_llama/prompt_builder.h"
#include "mini_llama/request_context.h"
#include "mini_llama/sampler.h"
#include "mini_llama/terminal.h"
#include "mini_llama/thread_pool.h"
#include "mini_llama/tokenizer.h"

// 命令行装配层：选择资源并组织现有模块，不在入口中实现计算后端
// 新命令在分发处接入，模型格式和分词器选择集中在加载辅助函数中扩展
namespace mini_llama {

// 保留既有模板读取接口，供集成测试和资源装配共同使用
std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  // 模板属于可选资源；读取失败时交给 PromptBuilder 使用纯文本格式
  GgufReader reader;
  if (!reader.Load(gguf_path)) {
    return {};
  }

  // 先查显式模板，同时记住架构信息，供缺少模板时判断是否可用内置格式
  const GgufMetadataKv* architecture = nullptr;
  for (const auto& item : reader.metadata) {
    if (item.key == "tokenizer.chat_template") {
      // 显式空模板也有意义，不能被架构回退覆盖
      return item.type == kGgufTypeString ? item.value_str : std::string();
    }
    if (item.key == "general.architecture") {
      architecture = &item;
    }
  }

  // 只有已支持的 qwen2 架构使用内置回退，其余架构返回空模板
  if (architecture != nullptr && architecture->type == kGgufTypeString &&
      architecture->value_str == "qwen2") {
    return "qwen2";
  }
  return {};
}

// ----------------------------------------------
// 参数解析与命令帮助
// ----------------------------------------------

// 保存原始命令行配置；路径解析和资源加载在后续阶段完成
struct CommandOptions {
  std::string command = "generate";
  bool help = false;
  std::string model_path;
  std::string config_path;
  std::string weights_path;
  std::string tokenizer_path;
  std::string vocab_path;
  std::string prompt;
  int n_predict = 16;  // generate 的总生成预算，或 run 的单轮响应上限
  int n_threads = -1;  // 未指定时保留当前配置；0 表示自动线程数
  SamplingParams sampling;
  std::string dump_logits_dir;
  std::string quant_type;
  bool verbose = false;
  BackendConfig backend;
};

// 数值转换完整消费字符串；错误在模型加载前拒绝
static bool ParseIntArg(const char* text, int& value) {
  if (!text || text[0] == '\0' || text[0] == '-' || text[0] == ' ') {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0' || errno == ERANGE || parsed < 0 ||
      parsed > 1000000) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

static bool ParseFloatArg(const char* text, float& value) {
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(parsed) ||
      parsed < 0.0 || parsed > 10000.0) {
    return false;
  }
  value = static_cast<float>(parsed);
  return true;
}

static bool ParseUintArg(const char* text, unsigned int& value) {
  if (!text || text[0] == '-' || text[0] == '\0' || text[0] == ' ') {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const uint64_t parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || errno == ERANGE ||
      parsed > std::numeric_limits<unsigned int>::max()) {
    return false;
  }
  value = static_cast<unsigned int>(parsed);
  return true;
}

static bool ParseBackendArg(const char* text, BackendConfig& config) {
  BackendKind kind;
  if (!ParseBackendKind(text, kind)) {
    return false;
  }
  config.kind = kind;
  return true;
}

// 使用 from_chars 拒绝负号、空白、尾随字符和溢出；不限制合法 seed 的范围
template <typename Integer>
Integer ParseInteger(const std::string& text, const std::string& option) {
  Integer value = 0;
  // 转换成功仍需检查是否消费了整个字符串，例如 "12abc" 不能接受
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() ||
      text.empty() || text.front() == '-') {
    throw std::invalid_argument("invalid value for " + option + ": " + text);
  }
  return value;
}

// 为 token 数和 top-k 增加命令行范围约束；随机种子单独使用无符号解析
static int ParseIntArg(const std::string& text, const std::string& option) {
  int value = 0;
  if (!ParseIntArg(text.c_str(), value)) {
    throw std::invalid_argument("invalid value for " + option + ": " + text);
  }
  return value;
}

static void ParseBackendArg(const std::string& text, BackendConfig& config) {
  if (!ParseBackendArg(text.c_str(), config)) {
    throw std::invalid_argument("invalid --backend: " + text +
                                "; expected cpu or cuda");
  }
}

// 设备编号按 int 全范围解析，由 Runtime 判断设备是否存在
static bool ParseDeviceArg(const char* text, BackendConfig& config) {
  int value = 0;
  const auto end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, value);
  if (result.ec != std::errc() || result.ptr != end || text == end || value < 0) {
    return false;
  }
  config.device_id = value;
  config.device_id_set = true;
  return true;
}

static void ParseDeviceArg(const std::string& text, BackendConfig& config) {
  if (config.device_id_set) {
    throw std::invalid_argument("duplicate option: --device");
  }
  if (!ParseDeviceArg(text.c_str(), config)) {
    throw std::invalid_argument("invalid --device: " + text +
                                "; expected a non-negative integer GPU ID");
  }
}

// 温度必须是有限的非负数；范围错误在转换为 float 前检查
static float ParseFloatArg(const std::string& text, const std::string& option) {
  float value = 0.0f;
  if (!ParseFloatArg(text.c_str(), value)) {
    throw std::invalid_argument("invalid value for " + option + ": " + text);
  }
  return value;
}

// 生成类命令共用采样选项，检查类命令只接受资源路径和帮助选项
static bool IsGenerationCommand(const std::string& command) {
  return command == "generate" || command == "run";
}

// 需要模型、分词器和 prompt 的命令共用资源及基础推理参数
static bool IsInferenceCommand(const std::string& command) {
  return IsGenerationCommand(command) || command == "bench";
}

// 诊断使用统一 Runtime 接口；错误写入注入的流，保持 CLI 的非零退出约定。
static int RunCudaInfo(std::ostream& output, std::ostream& error) {
  output << "CUDA backend built: " << (CudaRuntimeBuilt() ? "yes" : "no") << '\n';
  try {
    const int count = CudaDeviceCount();
    output << "CUDA GPU count: " << count << '\n';
    for (int device_id = 0; device_id < count; ++device_id) {
      output << CudaFormatDeviceInfo(CudaGetDeviceInfo(device_id)) << '\n';
    }
    if (count == 0) {
      output << "No CUDA devices are available.\n";
    }
    return 0;
  } catch (const std::exception& exception) {
    error << "CUDA info failed: " << exception.what() << '\n';
    return 1;
  }
}

// 后端校验只报告实际错误，CUDA 配置失败时不执行 CPU 推理
static bool ValidateBackendOrPrint(const BackendConfig& config,
                                   std::ostream& error = std::cerr) {
  try {
    ValidateBackend(config);
    return true;
  } catch (const std::exception& e) {
    error << "Backend setup failed: " << e.what() << '\n';
    return false;
  }
}

static void PrintBackendInfo(const BackendConfig& config,
                             std::ostream& out = std::cout) {
  out << "backend: " << BackendKindName(config.kind) << '\n';
  out << BackendExecutionNote(config) << '\n';
  if (config.kind == BackendKind::kCuda) {
    out << "cuda: " << CudaDeviceSummary(config.device_id) << '\n';
  }
}

static std::string FormatMb(size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MB";
  return out.str();
}

// 权重准备单独完成，失败保留原设备快照并给出上传阶段诊断
static bool PrepareCudaWeightsOrPrint(MiniLlamaModel& model,
                                      const BackendConfig& config,
                                      std::ostream& error = std::cerr) {
  if (config.kind != BackendKind::kCuda) {
    return true;
  }
  try {
    UploadModelWeightsToCuda(model, config.device_id);
    return true;
  } catch (const std::exception& e) {
    error << "CUDA weight upload failed: " << e.what() << '\n';
    return false;
  }
}

static void PrintCudaWeightSummary(const MiniLlamaModel& model,
                                   const std::string& indent = "",
                                   std::ostream& out = std::cout) {
  out << indent << "uploaded weights: " << ModelCudaUploadedWeightCount(model)
      << '\n';
  out << indent << "gpu memory used: " << ModelCudaMemoryBytes(model) << " bytes ("
      << FormatMb(ModelCudaMemoryBytes(model)) << ")\n";
}

// 分类型统计只包含 Linear；稠密 Embedding/Norm/Bias 另计上传总量
struct LinearWeightTypeCounts {
  size_t f32 = 0;
  size_t q8_0 = 0;
  size_t q4_0 = 0;
  size_t q4_1 = 0;
  size_t total() const { return f32 + q8_0 + q4_0 + q4_1; }
  size_t quantized() const { return q8_0 + q4_0 + q4_1; }
};

static void CountLinearWeight(const QuantizedTensor& weight,
                              LinearWeightTypeCounts& counts) {
  switch (weight.type) {
    case QuantType::kF32:
      ++counts.f32;
      return;
    case QuantType::kQ80:
      ++counts.q8_0;
      return;
    case QuantType::kQ40:
      ++counts.q4_0;
      return;
    case QuantType::kQ41:
      ++counts.q4_1;
      return;
  }
  throw std::runtime_error("unknown Linear weight type");
}

static LinearWeightTypeCounts ModelLinearWeightTypeCounts(
    const MiniLlamaModel& model) {
  LinearWeightTypeCounts counts;
  CountLinearWeight(model.lm_head, counts);
  for (const auto& lw : model.layers) {
    for (const auto* w :
         {&lw.wq, &lw.wk, &lw.wv, &lw.wo, &lw.w_gate, &lw.w_up, &lw.w_down}) {
      CountLinearWeight(*w, counts);
    }
  }
  return counts;
}

static void PrintCudaExecutionSummary(const MiniLlamaModel& model,
                                      const std::string& indent = "",
                                      std::ostream& out = std::cout) {
  PrintCudaWeightSummary(model, indent, out);
  const auto counts = ModelLinearWeightTypeCounts(model);
  out << indent
      << "cuda f32 Linear weights: " << ModelCudaUploadedF32WeightCount(model) << "/"
      << counts.total() << '\n';
  if (counts.q8_0 > 0) {
    out << indent
        << "cuda q8_0 Linear weights: " << ModelCudaUploadedQ80WeightCount(model)
        << "/" << counts.q8_0 << '\n';
  }
  if (counts.q4_0 > 0) {
    out << indent
        << "cuda q4_0 Linear weights: " << ModelCudaUploadedQ40WeightCount(model)
        << "/" << counts.q4_0 << '\n';
  }
  if (counts.q4_1 > 0) {
    out << indent
        << "cuda q4_1 Linear weights: " << ModelCudaUploadedQ41WeightCount(model)
        << "/" << counts.q4_1 << '\n';
  }

  // 缺失量化权重会在 Forward 明确报错，不将其显示为可用的 CPU 回退
  if (ModelCudaUploadedF32WeightCount(model) +
          ModelCudaUploadedQ80WeightCount(model) +
          ModelCudaUploadedQ40WeightCount(model) +
          ModelCudaUploadedQ41WeightCount(model) !=
      counts.total()) {
    out << indent << "cuda weights incomplete: CUDA forward will report an error\n";
  }
}

// 执行摘要累计当前推理的 Prefill + Decode，不包含模型权重上传
static void PrintCudaRuntimeSummary(const MiniLlamaModel& model,
                                    const std::string& indent = "",
                                    std::ostream& out = std::cout) {
  out << indent << "cuda Linear calls: " << ModelCudaLinearCalls(model) << '\n';
  out << indent << "cuda activation calls: " << ModelCudaActivationCalls(model)
      << '\n';
  out << indent << "cuda attention calls: " << ModelCudaAttentionCalls(model)
      << '\n';
  out << indent
      << "cpu attention fallback calls: " << ModelCudaAttentionCpuFallbacks(model)
      << '\n';
  out << indent << "cuda kv write bytes: " << ModelCudaKvCacheWriteBytes(model)
      << '\n';
  out << indent << "cuda kv read bytes: " << ModelCudaKvCacheReadBytes(model)
      << '\n';
  out << indent << "host->device copies: " << ModelCudaHostToDeviceCopies(model)
      << " (" << ModelCudaHostToDeviceBytes(model) << " bytes)\n";
  out << indent << "device->host copies: " << ModelCudaDeviceToHostCopies(model)
      << " (" << ModelCudaDeviceToHostBytes(model) << " bytes)\n";
}

// 显式覆盖只转换 Linear，Embedding、Norm 和 Bias 保持原有 F32 格式
static void ApplyQuantOverride(MiniLlamaModel& model,
                               const std::string& quant_type) {
  if (quant_type.empty()) {
    return;
  }
  if (quant_type == "q8_0") {
    QuantizeModelToQ80(model);
    return;
  }
  if (quant_type == "q4_0") {
    QuantizeModelToQ40(model);
    return;
  }
  throw std::runtime_error("unsupported quant type: " + quant_type);
}

static Tensor RunLogitsForTokens(const MiniLlamaModel& model,
                                 const std::vector<int>& tokens) {
  MiniLlamaContext ctx(&model);
  return ForwardBatch(ctx, model, MiniBatch::FromTokens(tokens, 0));
}

static std::pair<float, float> LogitsError(const Tensor& baseline,
                                           const Tensor& candidate) {
  if (baseline.shape != candidate.shape || baseline.size() != candidate.size() ||
      baseline.size() == 0) {
    throw std::runtime_error("LogitsError: shape/storage mismatch");
  }
  float max_error = 0.0f;
  double sum_error = 0.0;
  for (size_t i = 0; i < baseline.size(); ++i) {
    const float error = std::abs(baseline.data[i] - candidate.data[i]);
    max_error = std::max(max_error, error);
    sum_error += error;
  }
  return {max_error, static_cast<float>(sum_error / baseline.size())};
}

// 全局帮助只列出当前已接入的命令，具体参数交给对应命令帮助说明
static void PrintGlobalUsage(const char* prog, std::ostream& out) {
  out << "Usage: " << prog << " <command> [options]\n"
      << "Commands:\n"
      << "  generate      One-shot text generation\n"
      << "  run           Interactive chat mode\n"
      << "  bench         Benchmark prefill and token decoding\n"
      << "  inspect       Inspect model configuration and tensors\n"
      << "  inspect-gguf  Inspect GGUF metadata and tensor directory\n"
      << "  --cuda-info   Diagnose the compiled CUDA backend and GPUs\n"
      << "\nRun \"" << prog << " <command> --help\" for details.\n";
}

// 按命令输出有效选项和默认值；查看帮助不需要加载任何模型资源
static void PrintCommandUsage(const char* prog, const std::string& command,
                              std::ostream& out) {
  if (command == "inspect-gguf") {
    // 原始 GGUF 检查只需要文件路径，不涉及分词器或生成参数
    out << "Usage: " << prog << " inspect-gguf <path>\n";
    return;
  }
  out << "Usage: " << prog << " " << command;
  if (command == "generate") {
    out << " [options]\n";
  } else {
    out << " <model-path|dir> [options]\n";
  }
  out << "Options:\n"
      << "  -m, --model <path|dir> Model binary, GGUF file, or model directory\n"
      << "  --config <path>       JSON manifest for a binary model\n";
  out << "  --backend cpu|cuda    Compute backend (default: cpu)\n"
      << "  --device <id>         Non-negative GPU ID (CUDA only; default: 0)\n";
  if (command == "inspect") {
    out << "                        Metadata only: backend/device are validated,\n"
        << "                        but do not query GPUs or upload weights\n";
  }

  if (IsInferenceCommand(command)) {
    out << "  --weights <path>      Alias for --model with binary weights\n"
        << "  --tokenizer <path>    Tokenizer directory or vocab.json override\n"
        << "  --vocab <path>        Legacy alias for --tokenizer\n"
        << "  -p, --prompt <str>    "
        << (command == "run" ? "Optional first user message"
                             : "Input prompt (default: hello)")
        << "\n"
        << "  -n, --n-predict <n>   "
        << (command == "run" ? "Maximum response tokens (default: 64)"
                             : "Tokens to generate (default: 16; 0 = prefill only)")
        << "\n"
        << "  --max-tokens <n>      Legacy alias for --n-predict\n"
        << "  --threads <N>         CPU threads (0 = automatic; default: "
           "automatic)\n"
        << "  --seed <S>            Random seed (default: 0 = random)\n";

    if (IsGenerationCommand(command)) {
      out << "  --temperature <T>     Sampling temperature (default: 0 = greedy)\n"
          << "  --top-k <k>           Top-k sampling (default: 0 = disabled)\n";
    } else {
      out << "  --verbose             Print logits and KV cache details\n";
    }

    out << "  --quant q8_0|q4_0     Quantize Linear weights before inference\n";
    if (command == "generate") {
      out << "  --dump-logits <dir>   Dump step logits and generation_tokens.txt\n"
          << "                        Token-by-token prefill; steps start at 0\n"
          << "  Default model: models/tiny/model.bin\n";
    }
  }
  out << "  -h, --help            Show this help\n";
}

// 每个选项只消费自己的值，避免把 prompt 内的 --help/--model 误当作选项
static CommandOptions ParseOptions(int argc, const char* const argv[],
                                   bool allow_resource_options) {
  CommandOptions options;
  bool backend_set = false;
  int start = 1;
  // 首参数是选项时沿用旧版调用方式，默认进入 generate；否则消费命令名
  if (argc > 1 && argv[1][0] != '-') {
    options.command = argv[1];
    ++start;
  }
  if (!IsInferenceCommand(options.command) && options.command != "inspect" &&
      options.command != "inspect-gguf") {
    throw std::invalid_argument("unknown command: " + options.command);
  }

  // 各推理模式使用对应默认值；run 的 prompt 仅作为可选首轮输入
  options.n_predict = options.command == "run" ? 64 : 16;
  options.prompt =
      options.command == "generate" || options.command == "bench" ? "hello" : "";

  // 消费下一个 argv 并推进索引；文本值允许以 '-' 开头
  const auto read_value = [&](int& index, const std::string& option) {
    if (++index >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    return std::string(argv[index]);
  };

  const auto read_backend_value = [&](int& index, const std::string& option) {
    const std::string value = read_value(index, option);
    if (value.empty() || value.rfind("--", 0) == 0 || value == "-h") {
      throw std::invalid_argument("missing value for " + option);
    }
    return value;
  };

  // 资源路径禁止重复和空值，避免多个来源覆盖后难以判断最终加载了什么
  const auto set_path = [&](std::string& path, int& index,
                            const std::string& option) {
    if (!path.empty()) {
      throw std::invalid_argument("duplicate option: " + option);
    }
    path = read_value(index, option);
    if (path.empty()) {
      throw std::invalid_argument("empty value for " + option);
    }
  };

  // 库入口已经收到模型和分词器，因此通过 allow_resource_options 禁用路径选项
  for (int i = start; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "-h" || option == "--help") {
      options.help = true;
    } else if (options.command != "inspect-gguf" && option == "--backend") {
      if (backend_set) {
        throw std::invalid_argument("duplicate option: --backend");
      }
      ParseBackendArg(read_backend_value(i, option), options.backend);
      backend_set = true;
    } else if (options.command != "inspect-gguf" && option == "--device") {
      ParseDeviceArg(read_backend_value(i, option), options.backend);
    } else if (allow_resource_options && i == start && !option.empty() &&
               option.front() != '-') {
      // 位置模型参数只允许紧跟命令，后续裸参数不会被误当成另一个模型
      options.model_path = option;
    } else if (allow_resource_options && options.command != "inspect-gguf" &&
               (option == "--model" || option == "-m")) {
      set_path(options.model_path, i, option);
    } else if (allow_resource_options && options.command != "inspect-gguf" &&
               option == "--config") {
      set_path(options.config_path, i, option);
    } else if (allow_resource_options && IsInferenceCommand(options.command) &&
               option == "--weights") {
      set_path(options.weights_path, i, option);
    } else if (allow_resource_options && IsInferenceCommand(options.command) &&
               option == "--tokenizer") {
      set_path(options.tokenizer_path, i, option);
    } else if (allow_resource_options && IsInferenceCommand(options.command) &&
               option == "--vocab") {
      set_path(options.vocab_path, i, option);
    } else if (IsInferenceCommand(options.command) &&
               (option == "-p" || option == "--prompt")) {
      options.prompt = read_value(i, option);
    } else if (IsInferenceCommand(options.command) &&
               (option == "-n" || option == "--n-predict" ||
                option == "--max-tokens")) {
      options.n_predict = ParseIntArg(read_value(i, option), option);
    } else if (IsInferenceCommand(options.command) && option == "--threads") {
      options.n_threads = ParseIntArg(read_value(i, option), option);
    } else if (IsGenerationCommand(options.command) && option == "--temperature") {
      options.sampling.temperature = ParseFloatArg(read_value(i, option), option);
    } else if (IsGenerationCommand(options.command) && option == "--top-k") {
      options.sampling.top_k = ParseIntArg(read_value(i, option), option);
    } else if (IsInferenceCommand(options.command) && option == "--seed") {
      const auto value = read_value(i, option);
      if (!ParseUintArg(value.c_str(), options.sampling.seed)) {
        throw std::invalid_argument("invalid value for " + option + ": " + value);
      }
    } else if (IsInferenceCommand(options.command) && option == "--quant") {
      set_path(options.quant_type, i, option);
      if (options.quant_type != "q8_0" && options.quant_type != "q4_0") {
        throw std::invalid_argument("unsupported quant type: " + options.quant_type);
      }
    } else if (options.command == "bench" && option == "--verbose") {
      options.verbose = true;
    } else if (options.command == "generate" && option == "--dump-logits") {
      set_path(options.dump_logits_dir, i, option);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + option);
    }
  }

  // generate 允许只做 prefill；聊天模式必须留出至少一个响应 token 的预算
  if (options.command == "run" && options.n_predict == 0 && !options.help) {
    throw std::invalid_argument("--n-predict must be positive in run mode");
  }
  // 所有命令都拒绝 CPU 与显式设备参数冲突；这里只检查 CPU，不查询 GPU。
  if (options.backend.kind == BackendKind::kCpu) {
    ValidateBackend(options.backend);
  }
  return options;
}

// ----------------------------------------------
// 模型路径解析与分词器装配
// ----------------------------------------------

// 保存解析后的实际文件路径；GGUF 自带配置，因此 config 仅用于 JSON+BIN
struct ModelPaths {
  std::filesystem::path model;
  std::filesystem::path config;
  bool gguf = false;
};

// 将同一模型对应的权重、分词器和模板一起传递，避免运行阶段混用资源
struct LoadedModel {
  MiniLlamaModel model;
  std::unique_ptr<ITokenizer> tokenizer;
  std::string chat_template;
};

// 根据文件魔数判断格式，不能只依赖后缀
static bool IsGgufFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  char magic[4] = {};
  // 魔数匹配仅用于格式分发，文件版本与数据完整性由 GGUF 加载器继续校验
  return file.read(magic, sizeof(magic)) && std::memcmp(magic, "GGUF", 4) == 0;
}

// 扫描目录中的普通文件；无候选返回空路径，多个候选要求用户指定文件
static std::filesystem::path FindGgufInDirectory(
    const std::filesystem::path& directory) {
  std::vector<std::filesystem::path> candidates;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && IsGgufFile(entry.path())) {
      candidates.push_back(entry.path());
    }
  }

  // 多个候选必须由调用方明确选择，避免加载目录中碰巧排在首位的模型
  if (candidates.size() > 1) {
    throw std::runtime_error("model directory contains multiple GGUF files: " +
                             directory.string() + "; use --model FILE");
  }
  return candidates.empty() ? std::filesystem::path() : candidates.front();
}

// generate、run、inspect 共享资源解析；模型格式选择与执行命令解耦
static ModelPaths ResolveModelPaths(const CommandOptions& options) {
  namespace fs = std::filesystem;
  // --weights 是旧版二进制权重选项，不能与 --model 同时决定模型来源
  if (!options.model_path.empty() && !options.weights_path.empty()) {
    throw std::invalid_argument("--model and --weights cannot be combined");
  }
  ModelPaths paths;
  paths.model =
      options.model_path.empty() ? options.weights_path : options.model_path;
  paths.config = options.config_path;
  if (paths.model.empty()) {
    // 显式清单推导同目录 model.bin；只有 generate 提供教学模型默认路径
    if (!paths.config.empty()) {
      paths.model = paths.config.parent_path() / "model.bin";
    } else if (options.command == "generate") {
      paths.model = "models/tiny/model.bin";
    } else {
      throw std::invalid_argument("missing model path");
    }
  }

  if (!fs::exists(paths.model)) {
    // inspect 可以只读取清单，model.bin 不必存在
    if (options.command == "inspect" && !paths.config.empty()) {
      return paths;
    }
    throw std::invalid_argument("model path does not exist: " +
                                paths.model.string());
  }

  if (fs::is_directory(paths.model)) {
    const fs::path directory = paths.model;
    if (paths.config.empty()) {
      paths.config = directory / "model.json";
    }

    // 目录中 JSON+BIN 配对完整时优先使用旧格式；inspect 仅要求清单存在
    if (fs::is_regular_file(paths.config) &&
        (fs::is_regular_file(directory / "model.bin") ||
         options.command == "inspect")) {
      paths.model = directory / "model.bin";
    } else {
      // 旧格式资源不完整时查找 GGUF，丢弃自动推导的 JSON 清单路径
      paths.model = FindGgufInDirectory(directory);
      if (paths.model.empty()) {
        throw std::runtime_error(
            "model directory contains no model.json/model.bin "
            "pair or GGUF file: " +
            directory.string());
      }
      paths.config = options.config_path;
    }
  }

  paths.gguf = IsGgufFile(paths.model);
  if (paths.gguf) {
    // GGUF 配置与权重属于同一个文件，拒绝额外清单或旧版权重参数混用
    if (!options.config_path.empty() || !options.weights_path.empty()) {
      throw std::invalid_argument(
          "GGUF cannot be combined with --config or --weights");
    }
    paths.config.clear();
  } else if (options.command == "inspect" && paths.model.extension() == ".json") {
    // 支持直接检查清单文件，不必先找到二进制权重
    paths.config = paths.model;
  } else if (paths.config.empty()) {
    // 单独传入 model.bin 时，默认读取权重旁边的 model.json
    paths.config = paths.model.parent_path() / "model.json";
  }
  return paths;
}

// 外部词表可指向目录或 vocab.json；有 merges.txt 时使用 BPE
// 显式词表加载失败必须报错，不能由工厂的 ASCII 回退掩盖配置错误
static std::unique_ptr<ITokenizer> CreateTokenizerFromVocabHint(
    const std::string& hint) {
  namespace fs = std::filesystem;
  fs::path vocab(hint);
  if (fs::is_directory(vocab)) {
    // 目录提示统一转换为词表文件，合并规则和特殊 token 文件位于同目录
    vocab /= "vocab.json";
  }

  if (!fs::is_regular_file(vocab)) {
    throw std::runtime_error("tokenizer vocabulary does not exist: " +
                             vocab.string());
  }

  const fs::path directory = vocab.parent_path();
  if (fs::is_regular_file(directory / "merges.txt")) {
    // BPE 工厂通过空指针表示失败，转换为异常后由入口统一报告
    auto tokenizer =
        CreateBpeTokenizer(vocab.string(), (directory / "merges.txt").string(),
                           (directory / "special_tokens.json").string());
    if (!tokenizer) {
      throw std::runtime_error("failed to load BPE tokenizer: " + hint);
    }
    return tokenizer;
  }

  // 没有合并规则时使用项目的 JSON 词表格式；构造失败直接向上抛出
  return std::make_unique<JsonVocabTokenizer>(vocab.string());
}

// JSON 清单与实际分词器必须采用相同的特殊 ID，避免 BOS、EOS 和 UNK 语义错位
// 显式替换词表只改变资源来源，仍须遵守模型清单声明的特殊 token 约定
static void ValidateTokenizerSpecialIds(const TokenizerInfo& expected,
                                        const ITokenizer& tokenizer) {
  const auto require_same_id = [](const char* name, int expected_id, int actual_id) {
    if (expected_id != actual_id) {
      throw std::runtime_error(std::string("tokenizer ") + name +
                               " mismatch: manifest=" + std::to_string(expected_id) +
                               ", actual=" + std::to_string(actual_id));
    }
  };
  require_same_id("bos_id", expected.bos_id, tokenizer.bos_id());
  require_same_id("eos_id", expected.eos_id, tokenizer.eos_id());
  require_same_id("unk_id", expected.unk_id, tokenizer.unk_id());
}

// 按“显式指定、格式内元数据、同目录回退”的规则装配分词器，再校验清单约定
static LoadedModel LoadModelAndTokenizer(const ModelPaths& paths,
                                         const CommandOptions& options) {
  if (!options.tokenizer_path.empty() && !options.vocab_path.empty()) {
    throw std::invalid_argument("--tokenizer and --vocab cannot be combined");
  }

  // 先取得 JSON 清单，使自动选择和显式覆盖的分词器共用同一套 ID 校验
  // GGUF 分词器从文件元数据恢复特殊 ID，不使用 JSON 清单默认值
  ModelManifest manifest;
  if (!paths.gguf) {
    manifest = ParseManifest(paths.config.string());
  }

  // 两种格式都返回统一的 MiniLlamaModel，后续推理不再关心文件格式
  LoadedModel result;
  result.model = paths.gguf ? LoadGgufModel(paths.model.string())
                            : LoadModel(paths.config.string(), paths.model.string());
  if (!result.model.loaded) {
    throw std::runtime_error("failed to load model: " + result.model.load_error);
  }

  const std::string explicit_tokenizer =
      options.tokenizer_path.empty() ? options.vocab_path : options.tokenizer_path;
  if (!explicit_tokenizer.empty()) {
    // 用户提供的分词器优先级最高，加载失败时不改用其他来源
    result.tokenizer = CreateTokenizerFromVocabHint(explicit_tokenizer);
  } else if (paths.gguf) {
    // GGUF 优先恢复内嵌词表；缺失元数据时才查找权重同目录的外部词表
    result.tokenizer = CreateGgufTokenizer(paths.model.string());
    if (!result.tokenizer) {
      const auto directory = paths.model.has_parent_path()
                                 ? paths.model.parent_path()
                                 : std::filesystem::path(".");
      result.tokenizer = CreateTokenizerFromVocabHint(directory.string());
    }
  } else {
    // JSON 模型按清单中的 tokenizer.type 选择 ASCII 或 JSON 词表
    if (manifest.tokenizer.type == "ascii") {
      result.tokenizer = std::make_unique<AsciiTokenizer>();
    } else {
      std::filesystem::path vocab(manifest.tokenizer.path);
      if (vocab.empty()) {
        throw std::runtime_error("JSON manifest has no tokenizer vocabulary path");
      }
      if (vocab.is_relative()) {
        // 清单相对路径以清单目录为基准，而不是进程的当前工作目录
        vocab = paths.config.parent_path() / vocab;
      }
      result.tokenizer = CreateTokenizerFromVocabHint(vocab.string());
    }
  }
  if (!paths.gguf) {
    ValidateTokenizerSpecialIds(manifest.tokenizer, *result.tokenizer);
  }
  if (paths.gguf) {
    // 聊天模板始终来自模型文件，不随显式分词器覆盖而切换来源
    result.chat_template = LoadChatTemplateFromGguf(paths.model.string());
  }
  return result;
}

// ----------------------------------------------
// 推理校验、请求追踪与结果导出
// ----------------------------------------------

// 在分配上下文和运行采样前检查模型状态、参数以及词表容量的兼容性
static void ValidateGeneration(const MiniLlamaModel& model,
                               const ITokenizer& tokenizer, int n_predict,
                               const SamplingParams& sampling) {
  if (!model.loaded) {
    throw std::invalid_argument("model is not loaded");
  }
  if (n_predict < 0 || !std::isfinite(sampling.temperature) ||
      sampling.temperature < 0.0f || sampling.top_k < 0) {
    throw std::invalid_argument("invalid generation or sampling parameters");
  }
  if (tokenizer.vocab_size() > model.config.vocab_size) {
    // 允许模型词表更大，但分词器不能编码出模型词表范围之外的 ID
    throw std::invalid_argument("tokenizer vocabulary exceeds model vocabulary");
  }
}

// 编码并校验输入长度；needs_response 决定是否必须为生成保留剩余位置
static std::vector<int> EncodePrompt(const MiniLlamaModel& model,
                                     const ITokenizer& tokenizer,
                                     const std::string& prompt,
                                     bool needs_response) {
  std::vector<int> tokens = tokenizer.Encode(prompt);
  if (tokens.empty()) {
    // 空 prompt 使用合法 BOS；非空文本编码失败则明确报错
    if (!prompt.empty() || tokenizer.bos_id() < 0 ||
        tokenizer.bos_id() >= model.config.vocab_size) {
      throw std::invalid_argument("prompt requires at least one valid input token");
    }
    tokens.push_back(tokenizer.bos_id());
  }

  // 仅做 prefill 时允许占满窗口；生成或聊天时至少还需一个可用位置
  if (model.config.max_seq_len <= 0 ||
      tokens.size() > static_cast<size_t>(model.config.max_seq_len) ||
      (needs_response &&
       tokens.size() == static_cast<size_t>(model.config.max_seq_len))) {
    throw std::invalid_argument("prompt must be shorter than model context length");
  }
  return tokens;
}

// 先输出按记录顺序排列的事件，再输出摘要，便于脚本读取最终状态
static void PrintRequestTrace(const RequestContext& request, std::ostream& out) {
  for (const auto& line : FormatRequestTraceEvents(request)) {
    out << line << '\n';
  }
  out << FormatRequestTraceSummary(request) << '\n';
}

// 将整数序列统一格式化为方括号列表，供 token 结果和张量形状共用
static void PrintTokens(const std::vector<int>& tokens, std::ostream& out) {
  out << '[';
  for (size_t i = 0; i < tokens.size(); ++i) {
    out << (i == 0 ? "" : ", ") << tokens[i];
  }
  out << ']';
}

// 原始 token 与追踪保留 EOS，展示文本排除结束标记，延续既有库接口语义
static std::string DecodeGeneratedTokens(const ITokenizer& tokenizer,
                                         const std::vector<int>& tokens) {
  if (!tokens.empty() && tokens.back() == tokenizer.eos_id()) {
    return tokenizer.Decode(std::vector<int>(tokens.begin(), tokens.end() - 1));
  }
  return tokenizer.Decode(tokens);
}

// 导出采用本机 float 二进制布局，与逐步 logits 对照工具保持一致
static void DumpLogits(const Tensor& logits, const std::filesystem::path& path) {
  static_assert(sizeof(float) == 4, "logits dumps require float32");
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(logits.data.data()),
            static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
  // 显式关闭后再检查状态，同时覆盖写入与最终刷新失败
  out.close();
  if (!out) {
    throw std::runtime_error("failed to write logits dump: " + path.string());
  }
}

// 生成 ID 使用空格分隔的文本格式，包含 EOS，空结果也保留结尾换行
static void DumpGeneratedTokens(const std::vector<int>& tokens,
                                const std::filesystem::path& path) {
  std::ofstream out(path);
  for (size_t i = 0; i < tokens.size(); ++i) {
    out << (i == 0 ? "" : " ") << tokens[i];
  }
  out << '\n';
  out.close();
  if (!out) {
    throw std::runtime_error("failed to write generation tokens: " + path.string());
  }
}

// ----------------------------------------------
// generate：一次性生成
// ----------------------------------------------

// 完成 tokenize、prefill、采样和 decode，返回原始生成 ID 并累加请求指标
// require_full_budget 区分命令行的严格预算检查与既有库调用的截断语义
static std::vector<int> GenerateTokens(const MiniLlamaModel& model,
                                       const ITokenizer& tokenizer,
                                       const CommandOptions& options,
                                       RequestContext& request,
                                       bool require_full_budget = false) {
  ValidateGeneration(model, tokenizer, options.n_predict, options.sampling);

  // 先编码输入再计算剩余窗口，token 数不能用 prompt 的字符数代替
  auto stage_start = RequestClock::now();
  const std::vector<int> prompt =
      EncodePrompt(model, tokenizer, options.prompt, options.n_predict > 0);
  request.tokenize_ms = ElapsedMs(stage_start);
  request.prompt_tokens = static_cast<int>(prompt.size());
  request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                      "prompt");

  const int capacity = model.config.max_seq_len - request.prompt_tokens;

  // 命令行要求完整预算可容纳；既有库调用继续按剩余容量截断
  if (require_full_budget && options.n_predict > capacity) {
    throw std::invalid_argument("requested tokens exceed context window");
  }

  const int limit = std::min(options.n_predict, capacity);
  const std::filesystem::path dump_dir(options.dump_logits_dir);
  if (!dump_dir.empty()) {
    // 在推理前准备导出目录，避免完成昂贵计算后才发现目标路径不可用
    std::filesystem::create_directories(dump_dir);
    if (!std::filesystem::is_directory(dump_dir)) {
      throw std::runtime_error("logits dump path is not a directory: " +
                               dump_dir.string());
    }
  }

  // 单轮请求创建独立 KV Cache；step 连续编号 prefill 与 decode 的导出文件
  MiniLlamaContext context(&model);
  Tensor logits;
  int step = 0;
  stage_start = RequestClock::now();
  if (dump_dir.empty()) {
    // 普通路径一次提交完整 prompt，返回最后一个输入位置的 logits
    logits = ForwardBatch(context, model, MiniBatch::FromTokens(prompt));
  } else {
    // 开启导出时逐 token prefill，确保每个输入位置都有对应 logits
    for (size_t i = 0; i < prompt.size(); ++i) {
      logits = ForwardBatch(context, model,
                            MiniBatch::Single(prompt[i], static_cast<int>(i)));
      DumpLogits(logits,
                 dump_dir / ("logits_step" + std::to_string(step++) + ".bin"));
    }
  }

  // prefill 统计输入 token 数，与之后采样得到的生成 token 数分别记录
  context.n_prefill_tokens += request.prompt_tokens;
  request.prefill_tokens = request.prompt_tokens;
  request.prefill_ms = ElapsedMs(stage_start);
  request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                      dump_dir.empty() ? "batch" : "step_dump");

  // 一个请求内复用同一采样器，保持随机数序列连续并支持固定 seed 复现
  MiniSampler sampler(options.sampling);
  std::vector<int> generated;

  for (int i = 0; i < limit; ++i) {
    // 首个 token 使用 prefill 的 logits，后续 token 使用上轮 decode 的 logits
    stage_start = RequestClock::now();
    const int token = sampler.Sample(logits, options.sampling);
    request.sample_ms += ElapsedMs(stage_start);
    generated.push_back(token);
    request.generated_tokens = static_cast<int>(generated.size());
    // 首个生成 token 也可能是 EOS；停止判断必须早于下一次前向计算
    if (token == tokenizer.eos_id() || i + 1 == limit) {
      break;
    }

    // 只有还需要继续采样时才计算下一份 logits，最后一个 token 不再前向
    // ForwardBatch 会同步更新 context.pos 和 token_history
    stage_start = RequestClock::now();
    logits = ForwardBatch(context, model, MiniBatch::Single(token, context.pos + 1));
    const double elapsed = ElapsedMs(stage_start);
    request.decode_ms += elapsed;
    ++request.decode_tokens;
    ++context.n_decode_tokens;
    request.RecordEvent("decode", elapsed, 1, "pos=" + std::to_string(context.pos));
    if (!dump_dir.empty()) {
      DumpLogits(logits,
                 dump_dir / ("logits_step" + std::to_string(step++) + ".bin"));
    }
  }
  request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                      "generated_tokens");

  if (!dump_dir.empty()) {
    // token 结果与 logits 导出配套，便于检查每步 ArgMax 或采样输出
    DumpGeneratedTokens(generated, dump_dir / "generation_tokens.txt");
  }
  return generated;
}

// 命令行单轮生成入口：使用严格预算检查，并展示采样配置、原始 ID 与文本
static int RunGenerate(const LoadedModel& loaded, const CommandOptions& options,
                       RequestContext& request, std::ostream& output) {
  if (loaded.model.cuda_weights) {
    PrintCudaExecutionSummary(loaded.model, "", output);
  }
  const auto generated =
      GenerateTokens(loaded.model, *loaded.tokenizer, options, request, true);
  PrintBackendInfo(options.backend, output);
  output << "mini-llama.cpp\n==============\n\n"
         << "threads: " << GetThreadCount() << '\n'
         << "prompt: " << options.prompt << '\n'
         << "sampling: temperature=" << options.sampling.temperature
         << ", top_k=" << options.sampling.top_k
         << ", seed=" << options.sampling.seed << "\ngenerated tokens: ";
  PrintTokens(generated, output);
  output << "\ngenerated text: \""
         << DecodeGeneratedTokens(*loaded.tokenizer, generated) << "\"\n";
  if (loaded.model.cuda_weights) {
    output << "CUDA runtime (prefill + decode):\n";
    PrintCudaRuntimeSummary(loaded.model, "", output);
  }
  return 0;
}

// 执行独立基准测试，并把基准结果同步到请求追踪和命令行输出
static int RunBench(const LoadedModel& loaded, const CommandOptions& options,
                    RequestContext& request, std::ostream& output,
                    const MiniLlamaModel* baseline_model = nullptr) {
  const auto tokenize_start = RequestClock::now();
  const std::vector<int> prompt = EncodePrompt(
      loaded.model, *loaded.tokenizer, options.prompt, options.n_predict > 0);
  request.tokenize_ms = ElapsedMs(tokenize_start);
  request.prompt_tokens = static_cast<int>(prompt.size());
  request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                      "benchmark_prompt");

  if (loaded.model.cuda_weights) {
    PrintCudaExecutionSummary(loaded.model, "  ", output);
  }
  const BenchmarkResult result =
      RunBenchmark(loaded.model, prompt, options.n_predict, options.sampling.seed,
                   options.verbose, output);
  PrintBackendInfo(options.backend, output);
  request.prefill_tokens = result.n_prompt_tokens;
  request.generated_tokens = result.n_generated_tokens;
  request.decode_tokens = result.n_decode_tokens;
  request.prefill_ms = result.prefill_ms;
  request.decode_ms = result.decode_ms;
  request.RecordEvent("prefill", result.prefill_ms, result.n_prompt_tokens,
                      "benchmark");
  if (result.n_decode_tokens > 0) {
    request.RecordEvent("decode", result.decode_ms, result.n_decode_tokens,
                        "benchmark");
  }

  // 临时设置统计数值格式，输出后恢复调用方流的原有状态
  const std::ios::fmtflags old_flags = output.flags();
  const std::streamsize old_precision = output.precision();
  output << "mini-llama benchmark\n"
         << "====================\n"
         << "threads: " << GetThreadCount() << '\n'
         << "prompt tokens: " << result.n_prompt_tokens << '\n'
         << "generated tokens: " << result.n_generated_tokens << '\n'
         << "decode tokens: " << result.n_decode_tokens << '\n'
         << std::fixed << std::setprecision(3) << "prefill: " << result.prefill_ms
         << " ms\n"
         << "decode: " << result.decode_ms << " ms\n"
         << "throughput: " << result.tokens_per_sec() << " tokens/s\n"
         << "decode throughput: " << result.decode_tokens_per_sec() << " tokens/s\n"
         << "tokens/s (total): " << result.tokens_per_sec() << '\n'
         << "tokens/s (Decode): " << result.decode_tokens_per_sec() << '\n';

  if (loaded.model.cuda_weights) {
    output << "  cuda runtime (prefill + decode):\n";
    PrintCudaRuntimeSummary(loaded.model, "    ", output);
  }
  const size_t actual_bytes = ModelWeightBytes(loaded.model);
  const size_t f32_bytes = ModelWeightBytesF32(loaded.model);
  output << "  weight memory: actual=" << actual_bytes << " bytes ("
         << FormatMb(actual_bytes) << "), f32 equiv=" << f32_bytes << " bytes\n";

  if (baseline_model) {
    const auto error = LogitsError(RunLogitsForTokens(*baseline_model, prompt),
                                   RunLogitsForTokens(loaded.model, prompt));
    output << "  logits error vs model-native: max=" << std::scientific
           << error.first << ", mean=" << error.second << '\n';
  }
  output.flags(old_flags);
  output.precision(old_precision);
  return 0;
}

// ----------------------------------------------
// run：多轮会话、前缀复用和终端命令
// ----------------------------------------------

// 查找两个 token 序列从起点开始的相同部分，用于验证物理 KV Cache 的复用范围
static size_t CommonPrefixLength(const std::vector<int>& a,
                                 const std::vector<int>& b) {
  size_t i = 0;
  while (i < std::min(a.size(), b.size()) && a[i] == b[i]) {
    ++i;
  }
  return i;
}

// 返回 true 表示输入已由终端命令消费；普通用户文本交给后续推理处理
// /exit 通过 should_exit 通知外层循环退出，本函数不终止进程
static bool HandleCommand(const std::string& input, ChatSession& session,
                          MiniLlamaContext& context, const MiniLlamaModel& model,
                          const Terminal& terminal, bool& should_exit) {
  if (input.empty() || input.front() != '/') {
    return false;
  }
  if (input == "/help") {
    terminal.PrintHelp();
  } else if (input == "/clear") {
    // 逻辑消息、前缀索引和物理 KV Cache 一起清空，采样参数继续保留
    session.Clear();
    context = MiniLlamaContext(&model);
    terminal.PrintMessage("Session cleared.");
  } else if (input == "/stats") {
    terminal.PrintStats(session);
  } else if (input == "/params") {
    terminal.PrintParams(session.sampling_params);
  } else if (input == "/exit") {
    should_exit = true;
    terminal.PrintMessage("Bye.");
  } else {
    terminal.PrintMessage("Unknown command. Type /help.");
  }
  return true;
}

// 执行一轮会话：构建完整 prompt，只 prefill 缓存缺失后缀，再流式生成响应
// 模型与采样器跨轮复用；request 只记录当前这一轮的耗时与 token 数
static void RunChatTurn(const std::string& input, const MiniLlamaModel& model,
                        const ITokenizer& tokenizer, const CommandOptions& options,
                        const PromptBuilder& builder, Terminal& terminal,
                        ChatSession& session, MiniLlamaContext& context,
                        MiniSampler& sampler, RequestContext& request) {
  // 先在副本中追加用户消息，正常完成生成后再提交到会话历史
  auto candidate = session.messages;
  candidate.push_back({"user", input});
  auto stage_start = RequestClock::now();
  const std::string prompt = builder.Build(candidate);
  request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0,
                      "messages=" + std::to_string(candidate.size()));
  stage_start = RequestClock::now();
  const auto tokens = EncodePrompt(model, tokenizer, prompt, true);
  request.tokenize_ms = ElapsedMs(stage_start);
  request.prompt_tokens = static_cast<int>(tokens.size());
  request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens);

  const size_t cached_prefix = session.LongestCachedPrefix(tokens);

  // RadixTree 仅保存命中信息，实际 KV 数据以当前上下文的公共前缀为准
  size_t prefix =
      std::min(cached_prefix, CommonPrefixLength(tokens, context.token_history));
  if (prefix == tokens.size()) {
    // 当前不缓存 logits，完全命中时重算最后一个 token
    --prefix;
  }
  if (prefix == 0) {
    // 无公共前缀时重新创建上下文，旧 KV 数据不能用于当前 prompt
    context = MiniLlamaContext(&model);
  } else {
    // 将逻辑历史回退到命中末端；后续前向会覆盖旧分支的缓存位置
    context.token_history.resize(prefix);
    context.pos = static_cast<int>(prefix) - 1;
  }

  const std::vector<int> suffix(tokens.begin() + static_cast<std::ptrdiff_t>(prefix),
                                tokens.end());

  // suffix 的起始位置必须是 prefix，不能从零写入并覆盖复用部分
  stage_start = RequestClock::now();
  Tensor logits = ForwardBatch(
      context, model, MiniBatch::FromTokens(suffix, static_cast<int>(prefix)));
  request.prefill_ms = ElapsedMs(stage_start);
  request.prefill_tokens = static_cast<int>(suffix.size());
  context.n_prefill_tokens += request.prefill_tokens;
  request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                      "radix_hit=" + std::to_string(cached_prefix) +
                          ", prefix_reuse=" + std::to_string(prefix));

  // 会话跟踪的是实际已前向计算的 token，而非仅由模板构建出的逻辑消息
  session.SetTokenHistory(context.token_history);

  terminal.PrintAssistantPrefix();
  std::vector<int> generated;
  std::string streamed_reply;

  // 单轮响应预算根据剩余窗口截断，历史 prompt 本身不能超过窗口
  const int limit =
      std::min(options.n_predict, model.config.max_seq_len - request.prompt_tokens);
  for (int i = 0; i < limit; ++i) {
    stage_start = RequestClock::now();
    const int token = sampler.Sample(logits, session.sampling_params);
    request.sample_ms += ElapsedMs(stage_start);
    // EOS 计入生成统计，但既不展示，也不作为助手正文写入消息历史
    ++request.generated_tokens;
    if (token == tokenizer.eos_id()) {
      break;
    }
    generated.push_back(token);

    // 按完整 token 序列解码，保持跨 token 的 UTF-8 字节顺序
    const std::string reply = tokenizer.Decode(generated);
    if (reply.size() > streamed_reply.size()) {
      // 只输出相对上次结果新增的后缀，刷新流使用户立即看到生成内容
      terminal.PrintTokenText(reply.substr(streamed_reply.size()));
      terminal.Flush();
      streamed_reply = reply;
    }

    // 已输出 token 全部写入 KV Cache，下一轮可以复用完整上下文
    stage_start = RequestClock::now();
    logits = ForwardBatch(context, model, MiniBatch::Single(token, context.pos + 1));
    const double elapsed = ElapsedMs(stage_start);
    request.decode_ms += elapsed;
    ++request.decode_tokens;
    ++context.n_decode_tokens;
    request.RecordEvent("decode", elapsed, 1, "pos=" + std::to_string(context.pos));
    session.AppendToken(token);
  }
  terminal.NewLine();

  // 正常生成结束后提交用户与助手消息，并把已评估 token 前缀加入索引
  session.messages = std::move(candidate);
  session.AddMessage("assistant", tokenizer.Decode(generated));
  session.RecordPrefix(session.token_history);

  // 会话统计累加实际新增的 prefill 工作量，而非每轮完整 prompt 的长度
  session.RecordTurn(request.prefill_tokens, request.generated_tokens,
                     ElapsedMs(request.start_time));
  request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                      "generated_tokens");
}

// 驱动终端循环；trace_turns 决定是否展示逐轮追踪，库调用可保持简洁输出
static int RunChat(const MiniLlamaModel& model, const ITokenizer& tokenizer,
                   const std::string& chat_template, const CommandOptions& options,
                   std::istream& input, std::ostream& output,
                   const std::string& model_path, bool trace_turns) {
  ValidateGeneration(model, tokenizer, options.n_predict, options.sampling);

  // 终端使用注入的 I/O；会话、上下文和采样器在整个交互期间持续存在
  Terminal terminal(input, output);
  PromptBuilder builder;

  // 模板由装配层传入，空模板沿用 PromptBuilder 的纯文本格式
  builder.SetChatTemplate(chat_template);
  ChatSession session;
  session.sampling_params = options.sampling;
  MiniLlamaContext context(&model);
  MiniSampler sampler(options.sampling);
  if (trace_turns) {
    if (model.cuda_weights) {
      PrintCudaExecutionSummary(model, "", output);
    }
    output << "backend: " << BackendKindName(options.backend.kind) << '\n'
           << BackendExecutionNote(options.backend) << '\n';
  }
  terminal.PrintMessage(
      "mini-llama.cpp chat\nType /help for commands, /exit to quit.");

  bool pending_prompt = !options.prompt.empty();
  bool should_exit = false;
  while (!should_exit) {
    std::string line;
    if (pending_prompt) {
      // --prompt 只消费一次，作为首轮输入；此后恢复正常终端读取
      line = options.prompt;
      pending_prompt = false;
    } else {
      terminal.PrintUserPrompt();
      if (!terminal.ReadLine(&line)) {
        // EOF 正常结束交互，不将空输入当作一轮请求
        terminal.NewLine();
        break;
      }
    }

    // 空行和斜杠命令不创建请求，也不改变 token 生成统计
    if (line.empty() ||
        HandleCommand(line, session, context, model, terminal, should_exit)) {
      continue;
    }

    // 读取用户输入后才开始计时，将等待输入的时间排除在请求耗时之外
    RequestContext request =
        StartRequest("run", BackendKindName(options.backend.kind), model_path);
    try {
      RunChatTurn(line, model, tokenizer, options, builder, terminal, session,
                  context, sampler, request);
    } catch (const std::exception& exception) {
      terminal.NewLine();
      request.SetError(exception.what());
      terminal.PrintMessage("Error: " + request.error);

      // 消息仅在成功后提交；失败时清除物理缓存，避免半轮数据污染下一轮
      context = MiniLlamaContext(&model);
      session.SetTokenHistory({});
    }

    // 成功和失败的轮次都结束追踪；单轮失败不会退出整个聊天进程
    request.Finish();
    if (trace_turns) {
      PrintRequestTrace(request, output);
      if (model.cuda_weights) {
        PrintCudaRuntimeSummary(model, "", output);
      }
    }
  }
  return 0;
}

// ----------------------------------------------
// inspect / inspect-gguf：只读取元数据，不创建分词器或推理上下文
// ----------------------------------------------

// 检查 GGUF 文件头、元数据和张量目录，不装配模型权重或执行推理
static int RunInspectGguf(const std::filesystem::path& path, std::ostream& output) {
  GgufReader reader;
  if (!reader.Load(path.string())) {
    throw std::runtime_error("failed to inspect GGUF: " + reader.load_error);
  }
  // 使用注入输出流，避免检查命令绕过测试和嵌入调用的 I/O 边界
  output << "GGUF version=" << reader.version << " tensors=" << reader.n_tensors
         << " metadata=" << reader.n_metadata
         << " data_offset=" << reader.data_offset << '\n';

  // 数组元数据由 reader 提供摘要，避免终端输出完整的大型词表
  for (const auto& item : reader.metadata) {
    output << "metadata " << item.key << " (" << GgufValueTypeName(item.type)
           << "): " << item.value_str << '\n';
  }

  for (const auto& tensor : reader.tensors) {
    output << "tensor " << tensor.name << " shape=[";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
      output << (i == 0 ? "" : ",") << tensor.shape[i];
    }
    // 按 GGML 类型计算存储字节数；无法计算时以 0 展示而不读取张量数据
    uint64_t bytes = 0;
    output << "] type=" << GgmlTypeName(tensor.type) << " offset=" << tensor.offset
           << " bytes=" << (GgufTensorDataSize(tensor, bytes) ? bytes : 0) << '\n';
  }
  return 0;
}

// 统一模型检查入口，GGUF 复用文件检查，JSON 只解析和校验清单
static int RunInspect(const ModelPaths& paths, std::ostream& output) {
  if (paths.gguf) {
    return RunInspectGguf(paths.model, output);
  }

  // 无需打开 model.bin 或 vocab.json，因此可用于检查尚未备齐资源的目录
  const ModelManifest manifest = ParseManifest(paths.config.string());
  const auto& c = manifest.config;
  output << "=== Model Config ===\n"
         << "  vocab_size: " << c.vocab_size << "\n  dim: " << c.dim
         << "\n  hidden_dim: " << c.hidden_dim << "\n  n_layers: " << c.n_layers
         << "\n  n_heads: " << c.n_heads << "\n  n_kv_heads: " << c.n_kv_heads
         << "\n  head_dim: " << c.head_dim << "\n  max_seq_len: " << c.max_seq_len
         << "\n  rope_theta: " << c.rope_theta
         << "\n  rms_norm_eps: " << c.rms_norm_eps << "\n=== Tensors ("
         << manifest.tensors.size() << ") ===\n";

  // 展示张量的逻辑形状与二进制布局，便于核对清单与权重文件的约定
  for (const auto& tensor : manifest.tensors) {
    output << "  " << tensor.name << " shape=";
    PrintTokens(tensor.shape, output);
    output << " dtype=" << tensor.dtype << " offset=" << tensor.offset
           << " bytes=" << tensor.byte_size << '\n';
  }
  output << "=== Tokenizer ===\n  type: " << manifest.tokenizer.type
         << "\n  path: " << manifest.tokenizer.path << '\n';
  return 0;
}

// ----------------------------------------------
// 兼容现有库调用的可注入入口
// ----------------------------------------------

// 既有库接口：调用方提供已加载资源，返回过滤 EOS 的文本，不输出追踪日志
std::string Generate(const MiniLlamaModel& model, const ITokenizer& tokenizer,
                     const std::string& prompt, int max_tokens,
                     const SamplingParams& sampling) {
  CommandOptions options;
  options.prompt = prompt;
  options.n_predict = max_tokens;
  options.sampling = sampling;
  RequestContext request = StartRequest("generate", "cpu", "");
  return DecodeGeneratedTokens(tokenizer,
                               GenerateTokens(model, tokenizer, options, request));
}

// 可注入 I/O 的库入口，只负责运行选项与推理，不重复解析资源或加载模型
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::istream& input, std::ostream& output,
        std::ostream& error) {
  try {
    // false 禁止模型路径选项，模型与分词器已经由调用方传入
    const CommandOptions options = ParseOptions(argc, argv, false);
    if (options.help) {
      PrintCommandUsage(argv[0], options.command, output);
      return 0;
    }
    if (options.command != "generate" && options.command != "run") {
      throw std::invalid_argument("Run requires generate or run mode");
    }
    if (!ValidateBackendOrPrint(options.backend, error)) {
      return 1;
    }

    if (options.n_threads >= 0) {
      SetThreadCount(options.n_threads);
    }

    std::unique_ptr<MiniLlamaModel> prepared;
    const MiniLlamaModel* inference_model = &model;

    if ((options.backend.kind == BackendKind::kCuda &&
         (!model.cuda_weights ||
          model.cuda_weights->device_id != options.backend.device_id)) ||
        (options.backend.kind == BackendKind::kCpu && model.cuda_weights) ||
        !options.quant_type.empty()) {
      prepared = std::make_unique<MiniLlamaModel>(model);
      ApplyQuantOverride(*prepared, options.quant_type);
      if (options.backend.kind == BackendKind::kCuda) {
        UploadModelWeightsToCuda(*prepared, options.backend.device_id);
      } else {
        ClearModelCudaWeights(*prepared);
      }
      inference_model = prepared.get();
    }

    if (options.command == "run") {
      // 此接口没有模板参数，沿用纯文本聊天，并关闭逐轮追踪输出
      return RunChat(*inference_model, tokenizer, "", options, input, output, "",
                     false);
    }

    // 单轮库入口仅输出生成正文，保持既有调用和单元测试的输出约定
    RequestContext request =
        StartRequest("generate", BackendKindName(options.backend.kind), "");
    output << DecodeGeneratedTokens(
                  tokenizer,
                  GenerateTokens(*inference_model, tokenizer, options, request))
           << '\n';
    return 0;
  } catch (const std::exception& exception) {
    error << exception.what() << '\n';
    return 1;
  }
}

// 简化重载绑定标准输入，其余输出流继续由调用方指定
int Run(int argc, const char* const argv[], const MiniLlamaModel& model,
        const ITokenizer& tokenizer, std::ostream& output, std::ostream& error) {
  return Run(argc, argv, model, tokenizer, std::cin, output, error);
}

// ----------------------------------------------
// 进程入口：先校验命令，再加载资源，最后分发到对应模式
// ----------------------------------------------

// 进程装配入口：处理帮助与检查命令，加载资源，并在错误时统一收尾请求
int RunMain(int argc, const char* const argv[], std::istream& input,
            std::ostream& output, std::ostream& error) {
  RequestContext request;

  // 参数解析失败时尚未开始请求，避免输出带有未初始化身份信息的摘要
  bool trace_started = false;
  try {
    // 全局帮助和无参数用法在解析资源之前处理，避免不必要的文件访问
    if (argc < 2) {
      PrintGlobalUsage(argc > 0 ? argv[0] : "mini-llama", output);
      return 1;
    }
    if (argc == 2 &&
        (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      PrintGlobalUsage(argv[0], output);
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--cuda-info") {
      return RunCudaInfo(output, error);
    }

    const CommandOptions options = ParseOptions(argc, argv, true);
    if (options.help) {
      // 命令帮助同样先于路径检查，缺失模型时也可以查看有效参数
      PrintCommandUsage(argv[0], options.command, output);
      return 0;
    }

    if (IsInferenceCommand(options.command)) {
      if (!ValidateBackendOrPrint(options.backend, error)) {
        return 1;
      }
    }

    // 检查命令独立分发，不创建分词器、KV Cache 或生成请求
    if (options.command == "inspect-gguf") {
      if (options.model_path.empty()) {
        throw std::invalid_argument("inspect-gguf requires one GGUF file path");
      }
      return RunInspectGguf(options.model_path, output);
    }
    if (options.command == "inspect") {
      const int status = RunInspect(ResolveModelPaths(options), output);
      output << "backend: " << BackendKindName(options.backend.kind)
             << "\ncompute: none (metadata only; backend/device do not initialize "
                "CUDA)\n";
      return status;
    }
    if (options.n_threads >= 0) {
      SetThreadCount(options.n_threads);
    }

    std::string model_hint =
        options.model_path.empty() ? options.weights_path : options.model_path;
    if (model_hint.empty()) {
      model_hint = options.config_path.empty()
                       ? "models/tiny/model.bin"
                       : (std::filesystem::path(options.config_path).parent_path() /
                          "model.bin")
                             .string();
    }

    // 先用用户路径启动追踪，确保路径解析或模型加载失败也能报告请求状态
    request = StartRequest(options.command, BackendKindName(options.backend.kind),
                           model_hint);
    trace_started = true;
    const auto stage_start = RequestClock::now();

    // 解析实际模型文件后一起加载模型、分词器与模板，并检查词表兼容性
    const ModelPaths paths = ResolveModelPaths(options);
    LoadedModel loaded = LoadModelAndTokenizer(paths, options);

    // 基准量化误差对照保留原模型；量化与上传都在推理前完成。
    std::unique_ptr<MiniLlamaModel> baseline_model;
    if (options.command == "bench" && !options.quant_type.empty()) {
      baseline_model = std::make_unique<MiniLlamaModel>(loaded.model);
    }
    ApplyQuantOverride(loaded.model, options.quant_type);
    if (!PrepareCudaWeightsOrPrint(loaded.model, options.backend, error)) {
      throw std::runtime_error("CUDA weight preparation failed");
    }
    ResetModelCudaRuntimeStats(loaded.model);
    ValidateGeneration(loaded.model, *loaded.tokenizer, options.n_predict,
                       options.sampling);
    request.model_load_ms = ElapsedMs(stage_start);
    request.RecordEvent("model_load", request.model_load_ms, 0,
                        paths.model.string());

    if (options.command == "run") {
      // 模型加载和每轮用户请求独立计时，避免把等待用户输入算作推理耗时
      request.Finish();
      PrintRequestTrace(request, error);
      trace_started = false;
      return RunChat(loaded.model, *loaded.tokenizer, loaded.chat_template, options,
                     input, output, paths.model.string(), true);
    }

    // 单轮生成与基准测试都在完成推理后统一结束并输出请求追踪
    const int status =
        options.command == "bench"
            ? RunBench(loaded, options, request, output, baseline_model.get())
            : RunGenerate(loaded, options, request, output);
    request.Finish();
    PrintRequestTrace(request, error);
    return status;
  } catch (const std::exception& exception) {
    // 捕获资源与推理异常；只有已启动的请求才补充失败摘要
    error << exception.what() << '\n';
    if (trace_started) {
      request.SetError(exception.what());
      request.Finish();
      PrintRequestTrace(request, error);
    }
    return 1;
  }
}

// 标准进程调用绑定 stdin/stdout/stderr，测试可使用上面的流注入重载
int RunMain(int argc, const char* const argv[]) {
  return RunMain(argc, argv, std::cin, std::cout, std::cerr);
}

}  // namespace mini_llama

#ifdef MINI_LLAMA_BUILD_EXECUTABLE
// 核心库和可执行目标共享本文件，仅可执行目标启用全局 main
int main(int argc, char* argv[]) { return mini_llama::RunMain(argc, argv); }
#endif
