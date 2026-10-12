// 测试目的：验证 CLI 命令、参数校验、模型和分词器装配、输出及请求追踪。
#include "main_test.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "mini_llama/loader.h"
#include "mini_llama/terminal.h"
#include "mini_llama/tokenizer.h"

namespace {

const std::string kModelDir = MINI_LLAMA_TEST_MODEL_DIR;

using mini_llama::test::Invoke;

class MainTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    directory = std::filesystem::temp_directory_path() /
                ("mini_llama_main_test_" + std::to_string(id));
    std::filesystem::create_directory(directory);
  }
  void TearDown() override { std::filesystem::remove_all(directory); }

  // 保留真实教学模型的权重布局，只替换清单中的分词器描述。
  std::filesystem::path WriteManifest(const std::string& type, int bos = 1,
                                      int eos = 2, int unk = 0) {
    std::ifstream source(kModelDir + "/model.json");
    std::string json(std::istreambuf_iterator<char>(source), {});
    const size_t begin = json.find('{', json.find("\"tokenizer\""));
    const size_t end = json.find('}', begin);
    std::ostringstream description;
    description << "{\"type\":\"" << type << "\",\"path\":\"vocab.json\","
                << "\"bos_id\":" << bos << ",\"eos_id\":" << eos
                << ",\"unk_id\":" << unk << '}';
    json.replace(begin, end - begin + 1, description.str());
    const auto config = directory / "model.json";
    std::ofstream(config) << json;
    std::filesystem::copy_file(kModelDir + "/vocab.json", directory / "vocab.json",
                               std::filesystem::copy_options::overwrite_existing);
    return config;
  }
  std::filesystem::path directory;
};

std::string GeneratedLine(const std::string& output) {
  const size_t begin = output.find("generated tokens: ");
  if (begin == std::string::npos) {
    return {};
  }
  return output.substr(begin, output.find('\n', begin) - begin);
}

// 测试目的：帮助不访问模型；未实现命令必须报错，不能悄悄落入 generate。
TEST_F(MainTest, HelpAndUnknownCommandsDoNotLoadModels) {
  const auto help = Invoke({"mini-llama", "run", "missing-model", "--help"});
  EXPECT_EQ(help.status, 0);
  EXPECT_NE(help.output.find("--n-predict"), std::string::npos);
  EXPECT_TRUE(help.error.empty());
  const auto bench_help = Invoke({"mini-llama", "bench", "missing-model", "--help"});
  EXPECT_EQ(bench_help.status, 0);
  EXPECT_NE(bench_help.output.find("--verbose"), std::string::npos);
  EXPECT_TRUE(bench_help.error.empty());

  const auto unknown = Invoke({"mini-llama", "unknown"});
  EXPECT_EQ(unknown.status, 1);
  EXPECT_NE(unknown.error.find("unknown command"), std::string::npos);
  EXPECT_EQ(unknown.error.find("trace summary"), std::string::npos);
}

// 测试目的：bench 命令应复用模型与分词器装配，输出预填充和解码统计；
// verbose 调试信息必须写入调用方注入的输出流，并同步到请求追踪。
TEST_F(MainTest, BenchmarkCommandPrintsStatisticsAndVerboseDetails) {
  const auto result = Invoke({"mini-llama", "bench", kModelDir, "-p", "hello", "-n",
                              "2", "--seed", "42", "--verbose"});

  ASSERT_EQ(result.status, 0) << result.error;
  EXPECT_NE(result.output.find("[verbose] prefill"), std::string::npos);
  EXPECT_NE(result.output.find("tensor: logits"), std::string::npos);
  EXPECT_NE(result.output.find("KV cache: layers="), std::string::npos);
  EXPECT_NE(result.output.find("mini-llama benchmark"), std::string::npos);
  EXPECT_NE(result.output.find("generated tokens: 2"), std::string::npos);
  EXPECT_NE(result.output.find("decode tokens: 1"), std::string::npos);
  EXPECT_NE(result.output.find("decode throughput:"), std::string::npos);
  EXPECT_NE(result.output.find("total:"), std::string::npos);
  EXPECT_NE(result.error.find("mode=bench status=ok"), std::string::npos);
  EXPECT_NE(result.error.find("decode_tokens=1 generated_tokens=2"),
            std::string::npos);
}

TEST_F(MainTest, BenchmarkAcceptsWarmupAndRepeatOnlyInBenchMode) {
  const auto result = Invoke({"mini-llama", "bench", kModelDir, "-n", "2",
                              "--warmup", "1", "--repeat", "2"});
  ASSERT_EQ(result.status, 0) << result.error;
  EXPECT_NE(result.output.find("warmup runs: 1"), std::string::npos);
  EXPECT_NE(result.output.find("measured runs: 2"), std::string::npos);
  EXPECT_NE(result.output.find("inference min/median/max:"), std::string::npos);
  EXPECT_EQ(result.output.find("sample:"), std::string::npos);
  EXPECT_EQ(Invoke({"mini-llama", "bench", kModelDir, "--repeat", "0"}).status, 1);
  EXPECT_EQ(Invoke({"mini-llama", "generate", kModelDir, "--warmup", "1"}).status,
            1);
}

// 选项值中的标志字样是文本，不能被帮助扫描或资源路径扫描吞掉。
TEST_F(MainTest, PromptValuesAreNotReinterpretedAsOptions) {
  for (const auto* prompt : {"--help", "--model"}) {
    const auto result = Invoke({"mini-llama", "generate", "--model", kModelDir,
                                "--prompt", prompt, "-n", "0"});
    EXPECT_EQ(result.status, 0) << result.error;
    EXPECT_NE(result.output.find(std::string("prompt: ") + prompt),
              std::string::npos);
    EXPECT_NE(result.error.find("generated_tokens=0"), std::string::npos);
  }
}

// 路径写法只影响装配，不应改变同一模型、prompt 和 seed 的生成结果。
TEST_F(MainTest, JsonDirectoryBinaryAndLegacyArgumentsAgree) {
  const auto direct =
      Invoke({"mini-llama", "generate", "--model", kModelDir + "/model.bin", "-p",
              "hello", "-n", "3", "--seed", "42"});
  const auto directory_model = Invoke({"mini-llama", "generate", kModelDir, "-p",
                                       "hello", "-n", "3", "--seed", "42"});
  const auto legacy =
      Invoke({"mini-llama", "--config", kModelDir + "/model.json", "--weights",
              kModelDir + "/model.bin", "-p", "hello", "-n", "3", "--seed", "42"});
  ASSERT_EQ(direct.status, 0) << direct.error;
  ASSERT_EQ(directory_model.status, 0) << directory_model.error;
  ASSERT_EQ(legacy.status, 0) << legacy.error;
  EXPECT_EQ(GeneratedLine(direct.output), GeneratedLine(directory_model.output));
  EXPECT_EQ(GeneratedLine(direct.output), GeneratedLine(legacy.output));
}

// 现有 GGUF 无内嵌词表时应使用同目录词表；显式错误路径不能回退到 ASCII。
TEST_F(MainTest, GgufFallbackAndExplicitTokenizerErrors) {
  const auto fallback = Invoke(
      {"mini-llama", "generate", "--model", kModelDir + "/model.gguf", "-n", "0"});
  EXPECT_EQ(fallback.status, 0) << fallback.error;
  const auto invalid =
      Invoke({"mini-llama", "generate", "--model", kModelDir, "--tokenizer",
              (directory / "missing.json").string()});
  EXPECT_EQ(invalid.status, 1);
  EXPECT_NE(invalid.error.find("tokenizer vocabulary does not exist"),
            std::string::npos);
  EXPECT_NE(invalid.error.find("status=error"), std::string::npos);
}

// 自动加载、--tokenizer 和旧版 --vocab 覆盖都必须检查 BOS/EOS/UNK。
TEST_F(MainTest, JsonSpecialIdMismatchIsRejectedForEveryTokenizerSource) {
  const char* names[] = {"bos_id", "eos_id", "unk_id"};
  for (int field = 0; field < 3; ++field) {
    const auto config = WriteManifest("json_vocab", field == 0 ? 10 : 1,
                                      field == 1 ? 11 : 2, field == 2 ? 12 : 0);
    for (const auto* override_option : {"", "--tokenizer", "--vocab"}) {
      std::vector<std::string> args = {"mini-llama", "generate",
                                       "--config",   config.string(),
                                       "--weights",  kModelDir + "/model.bin",
                                       "-n",         "0"};
      if (override_option[0] != '\0') {
        args.insert(args.end(), {override_option, kModelDir + "/vocab.json"});
      }
      const auto result = Invoke(args);
      EXPECT_EQ(result.status, 1);
      EXPECT_NE(
          result.error.find(std::string("tokenizer ") + names[field] + " mismatch"),
          std::string::npos)
          << result.error;
      EXPECT_NE(result.error.find("manifest="), std::string::npos);
      EXPECT_NE(result.error.find("actual="), std::string::npos);
      EXPECT_NE(result.error.find("status=error"), std::string::npos);
      EXPECT_TRUE(result.output.empty());
    }
  }
}

// ASCII 分词器使用固定特殊 ID，清单与其约定冲突时必须在推理前报错。
TEST_F(MainTest, AsciiSpecialIdMismatchIsRejected) {
  for (int field = 0; field < 3; ++field) {
    const auto config = WriteManifest("ascii", field == 0 ? 10 : 1,
                                      field == 1 ? 11 : 2, field == 2 ? 12 : 0);
    const auto result = Invoke({"mini-llama", "run", "--config", config.string(),
                                "--weights", kModelDir + "/model.bin", "-n", "1"});
    EXPECT_EQ(result.status, 1);
    EXPECT_NE(result.error.find("mismatch"), std::string::npos);
    EXPECT_NE(result.error.find("mode=run status=error"), std::string::npos);
    EXPECT_TRUE(result.output.empty());
  }
}

// 校验要求清单与实际词表一致，不将特殊 ID 写死为教学模型的 1/2/0。
TEST_F(MainTest, MatchingNonDefaultSpecialIdsAreAccepted) {
  const auto config = WriteManifest("json_vocab", 3, 4, 5);
  std::ofstream(directory / "vocab.json")
      << R"([{"id":3,"content":"<bos>","special":true},
             {"id":4,"content":"<eos>","special":true},
             {"id":5,"content":"<unk>","special":true},
             {"id":6,"content":"hello","special":false}])";
  const auto result =
      Invoke({"mini-llama", "generate", "--config", config.string(), "--weights",
              kModelDir + "/model.bin", "-p", "hello", "-n", "0"});
  EXPECT_EQ(result.status, 0) << result.error;
  EXPECT_NE(result.error.find("prompt_tokens=2"), std::string::npos);
  EXPECT_NE(result.error.find("status=ok"), std::string::npos);
}

// Terminal 直接调用和 run 的 /help 应展示同一份完整说明，/params 只出现一次。
TEST_F(MainTest, TerminalAndChatShareCompleteCommandHelp) {
  std::istringstream input;
  std::ostringstream help;
  mini_llama::Terminal terminal(input, help);
  terminal.PrintHelp();
  EXPECT_NE(help.str().find("/params"), std::string::npos);
  const auto result =
      Invoke({"mini-llama", "run", kModelDir, "-n", "1"}, "/help\n/exit\n");
  EXPECT_EQ(result.status, 0) << result.error;
  EXPECT_NE(result.output.find(help.str()), std::string::npos);
  const size_t first = result.output.find("/params");
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(result.output.find("/params", first + 1), std::string::npos);
}

// 参数错误在模型读取前拒绝，整数溢出、负 seed 和非有限温度均不能进入采样。
TEST_F(MainTest, InvalidSamplingArgumentsAreRejectedBeforeLoading) {
  const std::vector<std::vector<std::string>> bad_options = {
      {"--seed", "-1"},
      {"--seed", "4294967296"},
      {"-n", "2147483648"},
      {"--top-k", "-1"},
      {"--temperature", "nan"},
      {"--temperature", "inf"},
      {"--temperature", "1e999"},
      {"--temperature", "1.0x"},
      {"--prompt"}};
  for (const auto& options : bad_options) {
    std::vector<std::string> args = {"mini-llama", "generate", "--model",
                                     "missing-model"};
    args.insert(args.end(), options.begin(), options.end());
    const auto result = Invoke(args);
    EXPECT_EQ(result.status, 1);
    EXPECT_EQ(result.error.find("trace summary"), std::string::npos) << result.error;
  }
}

// 开启逐步导出后生成结果应一致，每个 logits 文件应包含完整的 F32 词表。
TEST_F(MainTest, LogitsDumpMatchesNormalGenerationAndSupportsPrefillOnly) {
  const auto normal =
      Invoke({"mini-llama", "generate", "--model", kModelDir, "-n", "3"});
  const auto dump = Invoke({"mini-llama", "generate", "--model", kModelDir, "-n",
                            "3", "--dump-logits", directory.string()});
  ASSERT_EQ(normal.status, 0) << normal.error;
  ASSERT_EQ(dump.status, 0) << dump.error;
  EXPECT_EQ(GeneratedLine(normal.output), GeneratedLine(dump.output));
  EXPECT_TRUE(std::filesystem::is_regular_file(directory / "generation_tokens.txt"));
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() == ".bin") {
      EXPECT_EQ(entry.file_size(), 128u * sizeof(float));
    }
  }
  const auto prefill_dir = directory / "prefill";
  const auto prefill = Invoke({"mini-llama", "generate", "--model", kModelDir, "-n",
                               "0", "--dump-logits", prefill_dir.string()});
  ASSERT_EQ(prefill.status, 0) << prefill.error;
  EXPECT_NE(prefill.output.find("generated tokens: []"), std::string::npos);
  EXPECT_NE(prefill.error.find("decode_tokens=0 generated_tokens=0"),
            std::string::npos);
  std::ifstream file(prefill_dir / "generation_tokens.txt");
  EXPECT_EQ(std::string(std::istreambuf_iterator<char>(file), {}), "\n");
}

// inspect 仅解析清单，即使没有权重和词表文件也应成功，并写入注入的流。
TEST_F(MainTest, InspectNeedsMetadataOnly) {
  std::filesystem::copy_file(kModelDir + "/model.json", directory / "model.json");
  const auto result = Invoke({"mini-llama", "inspect", directory.string()});
  EXPECT_EQ(result.status, 0) << result.error;
  EXPECT_NE(result.output.find("vocab_size: 128"), std::string::npos);
  EXPECT_NE(result.output.find("lm_head"), std::string::npos);
  const auto gguf =
      Invoke({"mini-llama", "inspect-gguf", kModelDir + "/model.gguf"});
  EXPECT_EQ(gguf.status, 0) << gguf.error;
  EXPECT_NE(gguf.output.find("GGUF version="), std::string::npos);
}

// generate 严格检查预算；run 拒绝零响应预算，同时追踪每轮成功与失败。
TEST_F(MainTest, ContextBudgetAndPerTurnTracing) {
  const auto overflow =
      Invoke({"mini-llama", "generate", "--model", kModelDir, "-n", "128"});
  EXPECT_EQ(overflow.status, 1);
  EXPECT_NE(overflow.error.find("requested tokens exceed context window"),
            std::string::npos);
  const auto zero = Invoke({"mini-llama", "run", kModelDir, "-n", "0"});
  EXPECT_EQ(zero.status, 1);
  const auto chat =
      Invoke({"mini-llama", "run", kModelDir, "-n", "1"},
             std::string(200, 'x') + "\nhello\n/params\n/stats\n/clear\n/exit\n");
  EXPECT_EQ(chat.status, 0) << chat.error;
  EXPECT_NE(chat.output.find("mode=run status=error"), std::string::npos);
  EXPECT_NE(chat.output.find("mode=run status=ok"), std::string::npos);
  EXPECT_NE(chat.output.find("Messages: 2"), std::string::npos);
  EXPECT_NE(chat.output.find("temperature=0"), std::string::npos);
}

class EosTokenizer : public mini_llama::AsciiTokenizer {
 public:
  int eos_id() const override { return 0; }
  std::string Decode(const std::vector<int>& tokens) const override {
    return tokens.empty() ? "" : "unexpected EOS text";
  }
};

// 首次采样命中 EOS 时，库入口返回空文本且不执行后续 decode。
TEST_F(MainTest, FirstGeneratedEosIsNotDisplayed) {
  auto model =
      mini_llama::LoadModel(kModelDir + "/model.json", kModelDir + "/model.bin");
  ASSERT_TRUE(model.loaded);
  std::fill(model.lm_head.f32_data.begin(), model.lm_head.f32_data.end(), 0.0f);
  EosTokenizer tokenizer;
  std::istringstream input;
  std::ostringstream output, error;
  const char* argv[] = {"mini-llama", "generate", "-p", "hello", "-n", "3"};
  EXPECT_EQ(mini_llama::Run(6, argv, model, tokenizer, input, output, error), 0);
  EXPECT_EQ(output.str(), "\n");
  EXPECT_TRUE(error.str().empty());
}

}  // namespace
