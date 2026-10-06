#include "mini_llama/gguf.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

template <typename T>
void Append(std::vector<uint8_t>* bytes, T value) {
  const auto* raw = reinterpret_cast<const uint8_t*>(&value);
  bytes->insert(bytes->end(), raw, raw + sizeof(T));
}

void AppendString(std::vector<uint8_t>* bytes, const std::string& value) {
  Append<uint64_t>(bytes, value.size());
  bytes->insert(bytes->end(), value.begin(), value.end());
}

void AppendMetadataString(std::vector<uint8_t>* bytes, const std::string& key,
                          const std::string& value) {
  AppendString(bytes, key);
  Append<uint32_t>(bytes, mini_llama::kGgufTypeString);
  AppendString(bytes, value);
}

class TemporaryGguf {
 public:
  TemporaryGguf() {
    path_ = std::filesystem::temp_directory_path() / "mini_llama_gguf_test.gguf";
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  ~TemporaryGguf() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void WriteFixture(const std::filesystem::path& path) {
  std::vector<uint8_t> bytes = {'G', 'G', 'U', 'F'};
  Append<uint32_t>(&bytes, 3);
  Append<uint64_t>(&bytes, 1);
  Append<uint64_t>(&bytes, 6);
  AppendMetadataString(&bytes, "general.name", "unit");
  AppendString(&bytes, "general.alignment");
  Append<uint32_t>(&bytes, mini_llama::kGgufTypeUint32);
  Append<uint32_t>(&bytes, 32);
  AppendString(&bytes, "test.int");
  Append<uint32_t>(&bytes, mini_llama::kGgufTypeInt64);
  Append<int64_t>(&bytes, -7);
  AppendString(&bytes, "test.float");
  Append<uint32_t>(&bytes, mini_llama::kGgufTypeFloat64);
  Append<double>(&bytes, 1.5);
  AppendString(&bytes, "test.strings");
  Append<uint32_t>(&bytes, mini_llama::kGgufTypeArray);
  Append<uint32_t>(&bytes, mini_llama::kGgufTypeString);
  Append<uint64_t>(&bytes, 2);
  AppendString(&bytes, "one");
  AppendString(&bytes, "two");
  AppendString(&bytes, "test.ints");
  Append<uint32_t>(&bytes, mini_llama::kGgufTypeArray);
  Append<uint32_t>(&bytes, mini_llama::kGgufTypeInt32);
  Append<uint64_t>(&bytes, 3);
  Append<int32_t>(&bytes, 1);
  Append<int32_t>(&bytes, -2);
  Append<int32_t>(&bytes, 3);
  AppendString(&bytes, "weight");
  Append<uint32_t>(&bytes, 1);
  Append<uint64_t>(&bytes, 2);
  Append<uint32_t>(&bytes, mini_llama::kGgmlTypeF32);
  Append<uint64_t>(&bytes, 0);
  bytes.resize((bytes.size() + 31) / 32 * 32, 0);
  Append<float>(&bytes, 1.0f);
  Append<float>(&bytes, 2.0f);
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output.is_open());
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  ASSERT_TRUE(output.good());
}

// 测试目的：验证完整 GGUF 头、元数据、张量信息和数据偏移都能被加载。
TEST(GgufTest, LoadsMetadataAndTensorInfo) {
  TemporaryGguf fixture;
  WriteFixture(fixture.path());
  mini_llama::GgufReader reader;

  ASSERT_TRUE(reader.Load(fixture.path().string())) << reader.load_error;
  EXPECT_EQ(reader.version, 3u);
  EXPECT_EQ(reader.n_metadata, 6u);
  ASSERT_EQ(reader.tensors.size(), 1u);
  EXPECT_EQ(reader.tensors[0].name, "weight");
  EXPECT_EQ(reader.tensors[0].shape, std::vector<uint64_t>({2}));
  EXPECT_EQ(reader.data_offset % 32, 0u);
}

// 测试目的：验证五种类型化元数据读取接口的类型匹配、数组内容和缺失键行为。
TEST(GgufTest, ReadsTypedMetadataAndRawTensor) {
  TemporaryGguf fixture;
  WriteFixture(fixture.path());
  int64_t integer = 0;
  double floating = 0.0;
  std::string text;
  std::vector<std::string> strings;
  std::vector<int32_t> integers;
  ASSERT_TRUE(
      mini_llama::GgufGetMetadataInt(fixture.path().string(), "test.int", integer));
  ASSERT_TRUE(mini_llama::GgufGetMetadataFloat(fixture.path().string(), "test.float",
                                               floating));
  ASSERT_TRUE(mini_llama::GgufGetMetadataString(fixture.path().string(),
                                                "general.name", text));
  ASSERT_TRUE(mini_llama::GgufGetMetadataStringArray(fixture.path().string(),
                                                     "test.strings", strings));
  ASSERT_TRUE(mini_llama::GgufGetMetadataIntArray(fixture.path().string(),
                                                  "test.ints", integers));
  EXPECT_EQ(integer, -7);
  EXPECT_DOUBLE_EQ(floating, 1.5);
  EXPECT_EQ(text, "unit");
  EXPECT_EQ(strings, std::vector<std::string>({"one", "two"}));
  EXPECT_EQ(integers, std::vector<int32_t>({1, -2, 3}));
  EXPECT_FALSE(
      mini_llama::GgufGetMetadataInt(fixture.path().string(), "missing", integer));

  mini_llama::GgufReader reader;
  ASSERT_TRUE(reader.Load(fixture.path().string()));
  uint64_t size = 0;
  ASSERT_TRUE(mini_llama::GgufTensorDataSize(reader.tensors[0], size));
  EXPECT_EQ(size, 8u);
  EXPECT_EQ(mini_llama::ReadGgufTensorRaw(fixture.path().string(), reader.tensors[0],
                                          reader.data_offset),
            std::vector<uint8_t>({0, 0, 128, 63, 0, 0, 0, 64}));
}

// 测试目的：确认损坏头、越界张量和非法类型不会触发越界读取或伪成功。
TEST(GgufTest, RejectsMalformedInputAndInvalidSizes) {
  TemporaryGguf fixture;
  std::ofstream output(fixture.path(), std::ios::binary);
  output.write("bad", 3);
  output.close();
  mini_llama::GgufReader reader;
  EXPECT_FALSE(reader.Load(fixture.path().string()));
  EXPECT_FALSE(reader.load_error.empty());

  mini_llama::GgufTensorInfo invalid;
  invalid.shape = {3};
  invalid.type = mini_llama::kGgmlTypeQ40;
  uint64_t size = 0;
  EXPECT_FALSE(mini_llama::GgufTensorDataSize(invalid, size));
  EXPECT_TRUE(mini_llama::GgmlTypeName(999) == std::string("UNKNOWN"));
}

// 测试目的：验证仓库内的基础 GGUF fixture 可加载，并能读取其真实 F32 数据范围。
TEST(GgufTest, LoadsRepositoryFixture) {
  mini_llama::GgufReader reader;
  ASSERT_TRUE(reader.Load(std::string(MINI_LLAMA_TEST_MODEL_DIR) + "/test.gguf"))
      << reader.load_error;
  ASSERT_EQ(reader.tensors.size(), 1u);
  EXPECT_EQ(reader.tensors[0].name, "test.weight");
  EXPECT_EQ(reader.tensors[0].shape, std::vector<uint64_t>({2, 2}));
  uint64_t size = 0;
  ASSERT_TRUE(mini_llama::GgufTensorDataSize(reader.tensors[0], size));
  EXPECT_EQ(size, 4u * sizeof(float));
  EXPECT_EQ(mini_llama::ReadGgufTensorRaw(
                std::string(MINI_LLAMA_TEST_MODEL_DIR) + "/test.gguf",
                reader.tensors[0], reader.data_offset)
                .size(),
            size_t(size));
}

}  // namespace
