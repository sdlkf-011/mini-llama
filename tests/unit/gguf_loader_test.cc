#include "mini_llama/gguf_loader.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/gguf.h"
#include "mini_llama/quantized_tensor.h"

namespace {

using mini_llama::LoadGgufModel;
using mini_llama::QuantType;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            ("mini_llama_gguf_loader_test_" + std::to_string(unique));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  std::filesystem::path File(const std::string& name) const { return path_ / name; }

 private:
  std::filesystem::path path_;
};

template <typename Value>
void AppendScalar(std::vector<uint8_t>* bytes, Value value) {
  const auto* raw = reinterpret_cast<const uint8_t*>(&value);
  bytes->insert(bytes->end(), raw, raw + sizeof(Value));
}

void AppendString(std::vector<uint8_t>* bytes, const std::string& value) {
  AppendScalar<uint64_t>(bytes, value.size());
  bytes->insert(bytes->end(), value.begin(), value.end());
}

struct FixtureTensor {
  std::string name;
  std::vector<uint64_t> shape;  // GGUF 维度顺序，最快变化维在前
  uint32_t type;
  std::vector<uint8_t> data;
};

// 构造数值均为 1 的权重，量化块使用可手算的 scale 和载荷。
FixtureTensor MakeTensor(const std::string& name, const std::vector<uint64_t>& shape,
                         uint32_t type) {
  FixtureTensor tensor{name, shape, type, {}};
  size_t count = 1;
  for (uint64_t dimension : shape) {
    count *= dimension;
  }
  if (type == mini_llama::kGgmlTypeF32) {
    for (size_t index = 0; index < count; ++index) {
      AppendScalar<float>(&tensor.data, 1.0f);
    }
  } else if (type == mini_llama::kGgmlTypeF16) {
    for (size_t index = 0; index < count; ++index) {
      AppendScalar<uint16_t>(&tensor.data, 0x3c00);
    }
  } else {
    for (size_t block = 0; block < count / 32; ++block) {
      AppendScalar<uint16_t>(&tensor.data, 0x3800);  // FP16 scale=0.5
      if (type == mini_llama::kGgmlTypeQ80) {
        tensor.data.insert(tensor.data.end(), 32, 2);
      } else if (type == mini_llama::kGgmlTypeQ40) {
        tensor.data.insert(tensor.data.end(), 16, 0xaa);
      } else {
        AppendScalar<uint16_t>(&tensor.data, 0x3800);  // FP16 minimum=0.5
        tensor.data.insert(tensor.data.end(), 16, 0x11);
      }
    }
  }
  return tensor;
}

// 一层 GQA 模型使用不同的词表、隐藏和 FFN 维度，便于发现维度反转错误。
std::vector<FixtureTensor> ModelTensors(
    uint32_t linear_type = mini_llama::kGgmlTypeF32,
    uint32_t dense_type = mini_llama::kGgmlTypeF32) {
  return {
      MakeTensor("token_embd.weight", {32, 3}, dense_type),
      MakeTensor("output_norm.weight", {32}, dense_type),
      MakeTensor("blk.0.attn_norm.weight", {32}, dense_type),
      MakeTensor("blk.0.attn_q.weight", {32, 32}, linear_type),
      MakeTensor("blk.0.attn_k.weight", {32, 16}, linear_type),
      MakeTensor("blk.0.attn_v.weight", {32, 16}, linear_type),
      MakeTensor("blk.0.attn_output.weight", {32, 32}, linear_type),
      MakeTensor("blk.0.ffn_norm.weight", {32}, dense_type),
      MakeTensor("blk.0.ffn_gate.weight", {32, 64}, linear_type),
      MakeTensor("blk.0.ffn_up.weight", {32, 64}, linear_type),
      MakeTensor("blk.0.ffn_down.weight", {64, 32}, linear_type),
      MakeTensor("output.weight", {32, 3}, linear_type),
  };
}

// 临时 GGUF 包含完整模型目录和对齐后的真实数据，不依赖外部模型下载。
void WriteModel(const std::filesystem::path& path,
                const std::vector<FixtureTensor>& tensors,
                const std::string& architecture = "llama", int heads = 4) {
  std::vector<uint8_t> bytes = {'G', 'G', 'U', 'F'};
  AppendScalar<uint32_t>(&bytes, 3);
  AppendScalar<uint64_t>(&bytes, tensors.size());
  AppendScalar<uint64_t>(&bytes, 7);
  AppendString(&bytes, "general.architecture");
  AppendScalar<uint32_t>(&bytes, mini_llama::kGgufTypeString);
  AppendString(&bytes, architecture);

  const auto metadata = [&](const std::string& key, uint32_t value) {
    AppendString(&bytes, architecture + "." + key);
    AppendScalar<uint32_t>(&bytes, mini_llama::kGgufTypeUint32);
    AppendScalar<uint32_t>(&bytes, value);
  };
  metadata("embedding_length", 32);
  metadata("feed_forward_length", 64);
  metadata("block_count", 1);
  metadata("attention.head_count", heads);
  metadata("attention.head_count_kv", 2);
  metadata("context_length", 8);

  std::vector<uint8_t> data;
  for (const auto& tensor : tensors) {
    data.resize((data.size() + 31) / 32 * 32, 0);
    AppendString(&bytes, tensor.name);
    AppendScalar<uint32_t>(&bytes, tensor.shape.size());
    for (uint64_t dimension : tensor.shape) {
      AppendScalar<uint64_t>(&bytes, dimension);
    }
    AppendScalar<uint32_t>(&bytes, tensor.type);
    AppendScalar<uint64_t>(&bytes, data.size());
    data.insert(data.end(), tensor.data.begin(), tensor.data.end());
  }
  bytes.resize((bytes.size() + 31) / 32 * 32, 0);
  bytes.insert(bytes.end(), data.begin(), data.end());
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  ASSERT_TRUE(output);
}

// 测试 F32 模型的配置推导、GGUF 维度反转和层权重映射，并验证可执行前向。
TEST(GgufLoaderTest, LoadsF32ModelAndMapsShapes) {
  TemporaryDirectory directory;
  const auto path = directory.File("f32.gguf");
  WriteModel(path, ModelTensors());

  auto model = LoadGgufModel(path.string());

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_TRUE(model.load_error.empty());
  EXPECT_EQ(model.config.vocab_size, 3);
  EXPECT_EQ(model.config.dim, 32);
  EXPECT_EQ(model.config.hidden_dim, 64);
  EXPECT_EQ(model.config.n_heads, 4);
  EXPECT_EQ(model.config.n_kv_heads, 2);
  EXPECT_EQ(model.config.head_dim, 8);
  EXPECT_EQ(model.config.max_seq_len, 8);
  EXPECT_EQ(model.config.rope_type, mini_llama::RopeType::kNormal);
  EXPECT_EQ(model.token_embedding.shape, std::vector<int>({3, 32}));
  EXPECT_FLOAT_EQ(model.token_embedding.data[0], 1.0f);
  ASSERT_EQ(model.layers.size(), 1u);
  EXPECT_EQ(model.layers[0].wk.shape, std::vector<int>({16, 32}));
  EXPECT_EQ(model.layers[0].w_gate.shape, std::vector<int>({64, 32}));
  EXPECT_EQ(model.layers[0].w_down.shape, std::vector<int>({32, 64}));
  EXPECT_TRUE(model.layers[0].bq.data.empty());
  EXPECT_EQ(model.lm_head.type, QuantType::kF32);
  ASSERT_EQ(model.lm_head.f32_data.size(), 96u);
  EXPECT_FLOAT_EQ(model.lm_head.f32_data[0], 1.0f);

  mini_llama::MiniLlamaContext context(&model);
  const auto logits = mini_llama::ForwardToken(context, model, 0);
  ASSERT_EQ(logits.shape, std::vector<int>({3}));
  for (float value : logits.data) {
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_GT(value, 0.0f);
  }
}

// 测试 F16 的正负数被转换为 F32，线性权重也使用统一的 F32 存储。
TEST(GgufLoaderTest, ConvertsF16DenseAndLinearWeights) {
  TemporaryDirectory directory;
  const auto path = directory.File("f16.gguf");
  auto tensors = ModelTensors(mini_llama::kGgmlTypeF16, mini_llama::kGgmlTypeF16);
  tensors[0].data[2] = 0x00;
  tensors[0].data[3] = 0xc0;  // FP16 -2
  WriteModel(path, tensors);

  const auto model = LoadGgufModel(path.string());

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_FLOAT_EQ(model.token_embedding.data[0], 1.0f);
  EXPECT_FLOAT_EQ(model.token_embedding.data[1], -2.0f);
  EXPECT_EQ(model.layers[0].wq.type, QuantType::kF32);
  EXPECT_FLOAT_EQ(model.layers[0].wq.f32_data[0], 1.0f);
}

// 测试 Q8_0、Q4_0 和 Q4_1 保留线性权重块，稠密 embedding 则正确反量化。
TEST(GgufLoaderTest, KeepsQuantizedBlocksAndDecodesDenseWeights) {
  TemporaryDirectory directory;
  const auto path = directory.File("quantized.gguf");
  for (uint32_t type : {mini_llama::kGgmlTypeQ80, mini_llama::kGgmlTypeQ40,
                        mini_llama::kGgmlTypeQ41}) {
    SCOPED_TRACE(type);
    WriteModel(path, ModelTensors(type, type));
    const auto model = LoadGgufModel(path.string());

    ASSERT_TRUE(model.loaded) << model.load_error;
    ASSERT_EQ(model.token_embedding.size(), 96u);
    for (float value : model.token_embedding.data) {
      EXPECT_FLOAT_EQ(value, 1.0f);
    }
    const auto& weight = model.layers[0].wq;
    EXPECT_TRUE(weight.f32_data.empty());
    if (type == mini_llama::kGgmlTypeQ80) {
      EXPECT_EQ(weight.type, QuantType::kQ80);
      ASSERT_EQ(weight.q8_0_data.size(), 32u);
      EXPECT_EQ(weight.q8_0_data[0].d, 0x3800);
      EXPECT_EQ(weight.q8_0_data[0].qs[0], 2);
    } else if (type == mini_llama::kGgmlTypeQ40) {
      EXPECT_EQ(weight.type, QuantType::kQ40);
      ASSERT_EQ(weight.q4_0_data.size(), 32u);
      EXPECT_EQ(weight.q4_0_data[0].d, 0x3800);
      EXPECT_EQ(weight.q4_0_data[0].qs[0], 0xaa);
    } else {
      EXPECT_EQ(weight.type, QuantType::kQ41);
      ASSERT_EQ(weight.q4_1_data.size(), 32u);
      EXPECT_EQ(weight.q4_1_data[0].d, 0x3800);
      EXPECT_EQ(weight.q4_1_data[0].m, 0x3800);
      EXPECT_EQ(weight.q4_1_data[0].qs[0], 0x11);
    }
  }
}

// 测试省略 output.weight 时复用 embedding 的数值，并保留 Qwen2 RoPE 配对方式。
TEST(GgufLoaderTest, UsesTiedEmbeddingAndQwen2Rope) {
  TemporaryDirectory directory;
  const auto path = directory.File("tied.gguf");
  auto tensors = ModelTensors();
  tensors.pop_back();
  WriteModel(path, tensors, "qwen2");

  const auto model = LoadGgufModel(path.string());

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_EQ(model.config.rope_type, mini_llama::RopeType::kNeoX);
  EXPECT_EQ(model.lm_head.shape, model.token_embedding.shape);
  EXPECT_EQ(model.lm_head.f32_data, model.token_embedding.data);
}

// 测试可选 Q/K/V 偏置映射到正确字段，GQA 的 K/V 偏置使用较小的头维度。
TEST(GgufLoaderTest, LoadsOptionalAttentionBiases) {
  TemporaryDirectory directory;
  const auto path = directory.File("bias.gguf");
  auto tensors = ModelTensors();
  tensors.push_back(MakeTensor("blk.0.attn_q.bias", {32}, mini_llama::kGgmlTypeF32));
  tensors.push_back(MakeTensor("blk.0.attn_k.bias", {16}, mini_llama::kGgmlTypeF32));
  tensors.push_back(MakeTensor("blk.0.attn_v.bias", {16}, mini_llama::kGgmlTypeF32));
  WriteModel(path, tensors);

  const auto model = LoadGgufModel(path.string());

  ASSERT_TRUE(model.loaded) << model.load_error;
  EXPECT_EQ(model.layers[0].bq.shape, std::vector<int>({32}));
  EXPECT_EQ(model.layers[0].bk.shape, std::vector<int>({16}));
  EXPECT_EQ(model.layers[0].bv.shape, std::vector<int>({16}));
  EXPECT_FLOAT_EQ(model.layers[0].bk.data[0], 1.0f);
}

// 测试缺失文件和损坏头返回可检查的失败状态，不向调用方抛出异常。
TEST(GgufLoaderTest, RejectsMissingAndMalformedFiles) {
  TemporaryDirectory directory;
  const auto missing = LoadGgufModel(directory.File("missing.gguf").string());
  EXPECT_FALSE(missing.loaded);
  EXPECT_FALSE(missing.load_error.empty());

  const auto path = directory.File("malformed.gguf");
  {
    std::ofstream output(path, std::ios::binary);
    output << "invalid";
  }
  const auto malformed = LoadGgufModel(path.string());
  EXPECT_FALSE(malformed.loaded);
  EXPECT_FALSE(malformed.load_error.empty());
}

// 测试必需权重缺失和重复映射均拒绝发布部分初始化的模型。
TEST(GgufLoaderTest, RejectsMissingAndDuplicateWeights) {
  TemporaryDirectory directory;
  const auto path = directory.File("invalid_weights.gguf");
  auto tensors = ModelTensors();
  tensors.erase(tensors.begin() + 3);
  WriteModel(path, tensors);
  const auto missing = LoadGgufModel(path.string());
  EXPECT_FALSE(missing.loaded);
  EXPECT_NE(missing.load_error.find("missing required"), std::string::npos);

  tensors = ModelTensors();
  tensors.push_back(tensors[3]);
  WriteModel(path, tensors);
  const auto duplicate = LoadGgufModel(path.string());
  EXPECT_FALSE(duplicate.loaded);
  EXPECT_FALSE(duplicate.load_error.empty());
}

// 测试注意力头数不兼容或线性权重形状错误时，在进入前向前明确失败。
TEST(GgufLoaderTest, RejectsInvalidAttentionDimensionsAndWeightShapes) {
  TemporaryDirectory directory;
  const auto path = directory.File("invalid_shapes.gguf");
  WriteModel(path, ModelTensors(), "llama", 7);
  const auto attention = LoadGgufModel(path.string());
  EXPECT_FALSE(attention.loaded);
  EXPECT_NE(attention.load_error.find("attention dimensions"), std::string::npos);

  auto tensors = ModelTensors();
  tensors[3] = MakeTensor("blk.0.attn_q.weight", {32, 31}, mini_llama::kGgmlTypeF32);
  WriteModel(path, tensors);
  const auto shape = LoadGgufModel(path.string());
  EXPECT_FALSE(shape.loaded);
  EXPECT_FALSE(shape.load_error.empty());
}

// 测试权重数据短读会使加载失败，避免将截断文件作为完整模型使用。
TEST(GgufLoaderTest, RejectsTruncatedTensorData) {
  TemporaryDirectory directory;
  const auto path = directory.File("truncated.gguf");
  WriteModel(path, ModelTensors());
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);

  const auto model = LoadGgufModel(path.string());

  EXPECT_FALSE(model.loaded);
  EXPECT_FALSE(model.load_error.empty());
}

}  // namespace
