// 测试目的：构造独立临时 GGUF 模型，覆盖维度映射、F16 转换、量化块和损坏权重。
#ifndef MINI_LLAMA_TESTS_GGUF_LOADER_FIXTURE_H_
#define MINI_LLAMA_TESTS_GGUF_LOADER_FIXTURE_H_

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "mini_llama/gguf.h"

namespace mini_llama::test::gguf_loader {

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
inline void AppendScalar(std::vector<uint8_t>* bytes, Value value) {
  const auto* raw = reinterpret_cast<const uint8_t*>(&value);
  bytes->insert(bytes->end(), raw, raw + sizeof(Value));
}

inline void AppendString(std::vector<uint8_t>* bytes, const std::string& value) {
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
inline FixtureTensor MakeTensor(const std::string& name,
                                const std::vector<uint64_t>& shape, uint32_t type) {
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
inline std::vector<FixtureTensor> ModelTensors(
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
inline void WriteModel(const std::filesystem::path& path,
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

}  // namespace mini_llama::test::gguf_loader

#endif  // MINI_LLAMA_TESTS_GGUF_LOADER_FIXTURE_H_
