#include "mini_llama/loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mini_llama::LoadModel;
using mini_llama::ModelManifest;
using mini_llama::ParseManifest;

const std::filesystem::path kModelDirectory = MINI_LLAMA_TEST_MODEL_DIR;

// 临时文件在测试结束时自动清理，避免失败分支污染源码目录。
class TemporaryFile {
 public:
  explicit TemporaryFile(const std::string& filename) {
    path_ = std::filesystem::temp_directory_path() /
            ("mini_llama_loader_test_" + filename);
  }

  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("无法读取测试文件: " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void WriteText(const std::filesystem::path& path, const std::string& content) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output.is_open());
  output.write(content.data(), static_cast<std::streamsize>(content.size()));
  ASSERT_TRUE(output.good());
}

void ReplaceOnce(std::string* content, const std::string& from,
                 const std::string& to) {
  const size_t position = content->find(from);
  ASSERT_NE(position, std::string::npos);
  content->replace(position, from.size(), to);
}

// 验证样例清单能够生成完整配置、分词器信息和所有张量元数据。
TEST(LoaderTest, ParsesTinyManifest) {
  const ModelManifest manifest =
      ParseManifest((kModelDirectory / "model.json").string());

  EXPECT_EQ(manifest.config.vocab_size, 128);
  EXPECT_EQ(manifest.config.dim, 32);
  EXPECT_EQ(manifest.config.head_dim, 8);
  EXPECT_EQ(manifest.config.n_layers, 2);
  EXPECT_EQ(manifest.tokenizer.type, "json_vocab");
  EXPECT_EQ(manifest.tokenizer.path, "vocab.json");
  EXPECT_EQ(manifest.tensors.size(), 21u);
  EXPECT_EQ(manifest.tensors.front().name, "token_embedding");
  EXPECT_EQ(manifest.tensors.back().name, "lm_head");
}

// 验证真实 tiny 权重按名称和偏移加载，且各层形状可直接供前向模块使用。
TEST(LoaderTest, LoadsTinyModel) {
  const mini_llama::MiniLlamaModel model =
      LoadModel((kModelDirectory / "model.json").string(),
                (kModelDirectory / "model.bin").string());

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());
  ASSERT_EQ(model.layers.size(), 2u);
  EXPECT_EQ(model.token_embedding.shape, std::vector<int>({128, 32}));
  EXPECT_EQ(model.layers[0].w_gate.shape, std::vector<int>({86, 32}));
  EXPECT_EQ(model.layers[1].w_down.shape, std::vector<int>({32, 86}));
  EXPECT_EQ(model.lm_head.shape, std::vector<int>({128, 32}));
  EXPECT_TRUE(model.layers[0].bq.data.empty());
}

// 重复 JSON 键具有歧义，解析器必须拒绝而不是取第一个或最后一个值。
TEST(LoaderTest, RejectsDuplicateJsonKey) {
  std::string manifest = ReadText(kModelDirectory / "model.json");
  ReplaceOnce(&manifest, "\"dim\": 32,", "\"dim\": 32,\n    \"dim\": 32,");
  TemporaryFile file("duplicate_key.json");
  WriteText(file.path(), manifest);

  EXPECT_THROW(ParseManifest(file.path().string()), std::runtime_error);
}

// 无法整除注意力头数的隐藏维度会使 head_dim 非法，应在分配权重前失败。
TEST(LoaderTest, RejectsInvalidAttentionDimensions) {
  std::string manifest = ReadText(kModelDirectory / "model.json");
  ReplaceOnce(&manifest, "\"n_heads\": 4", "\"n_heads\": 3");
  TemporaryFile file("invalid_attention.json");
  WriteText(file.path(), manifest);

  EXPECT_THROW(ParseManifest(file.path().string()), std::runtime_error);
}

// 即使 tokenizer 整段缺省，默认 special ID 也必须落在当前词表范围内。
TEST(LoaderTest, ValidatesDefaultTokenizerIds) {
  const std::string manifest = R"({
    "config": {
      "vocab_size": 2,
      "dim": 2,
      "hidden_dim": 2,
      "n_layers": 1,
      "n_heads": 1,
      "n_kv_heads": 1,
      "max_seq_len": 2,
      "rope_theta": 10000.0,
      "rms_norm_eps": 0.00001
    },
    "tensors": []
  })";
  TemporaryFile file("default_tokenizer.json");
  WriteText(file.path(), manifest);

  EXPECT_THROW(ParseManifest(file.path().string()), std::runtime_error);
}

// 文件末尾的未知数据通常表示清单与权重版本不匹配，加载器必须明确失败。
TEST(LoaderTest, RejectsTrailingWeightBytes) {
  TemporaryFile weights_file("trailing_bytes.bin");
  std::ifstream source(kModelDirectory / "model.bin", std::ios::binary);
  ASSERT_TRUE(source.is_open());
  std::ofstream destination(weights_file.path(), std::ios::binary);
  ASSERT_TRUE(destination.is_open());
  destination << source.rdbuf();
  const char trailing = 0;
  destination.write(&trailing, 1);
  destination.close();

  const mini_llama::MiniLlamaModel model = LoadModel(
      (kModelDirectory / "model.json").string(), weights_file.path().string());
  EXPECT_FALSE(model.loaded);
  EXPECT_NE(model.load_error.find("trailing bytes"), std::string::npos);
}

}  // namespace
