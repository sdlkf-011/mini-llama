#include "mini_llama/gguf.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

namespace {

// 测试 F32 和 F16 按元素数量计算物理字节数，不能混淆数据类型的宽度。
TEST(GgufUnitTest, ComputesDenseTensorByteSizes) {
  mini_llama::GgufTensorInfo info;
  info.shape = {3, 2};
  uint64_t size = 0;

  info.type = mini_llama::kGgmlTypeF32;
  ASSERT_TRUE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 24u);
  info.type = mini_llama::kGgmlTypeF16;
  ASSERT_TRUE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 12u);
}

// 测试量化张量按 32 个元素一块计算大小，并计入各格式的块头。
TEST(GgufUnitTest, ComputesQuantizedTensorByteSizes) {
  mini_llama::GgufTensorInfo info;
  info.shape = {32, 2};
  uint64_t size = 0;

  info.type = mini_llama::kGgmlTypeQ80;
  ASSERT_TRUE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 68u);
  info.type = mini_llama::kGgmlTypeQ40;
  ASSERT_TRUE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 36u);
  info.type = mini_llama::kGgmlTypeQ41;
  ASSERT_TRUE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 40u);
}

// 测试未知类型保留元数据读取能力，但不提供可读取的物理数据大小。
TEST(GgufUnitTest, UnknownTensorTypeHasNoDataLayout) {
  mini_llama::GgufTensorInfo info;
  info.shape = {32};
  info.type = 999;
  uint64_t size = 123;

  EXPECT_TRUE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 0u);
  EXPECT_EQ(mini_llama::GgmlTypeName(info.type), std::string("UNKNOWN"));
}

// 测试零维度和元素数溢出均返回失败，并清除旧的字节数结果。
TEST(GgufUnitTest, InvalidTensorSizesClearOutput) {
  mini_llama::GgufTensorInfo info;
  info.type = mini_llama::kGgmlTypeF32;
  info.shape = {0, 2};
  uint64_t size = 123;
  EXPECT_FALSE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 0u);
  info.shape = {std::numeric_limits<uint64_t>::max(), 2};
  size = 123;
  EXPECT_FALSE(mini_llama::GgufTensorDataSize(info, size));
  EXPECT_EQ(size, 0u);
}

// 测试读取原始数据时先检查偏移溢出，避免对文件执行越界 seek 或分配。
TEST(GgufUnitTest, RejectsRawTensorOffsetOverflow) {
  mini_llama::GgufTensorInfo info;
  info.shape = {1};
  info.type = mini_llama::kGgmlTypeF32;
  info.offset = std::numeric_limits<uint64_t>::max();

  EXPECT_TRUE(mini_llama::ReadGgufTensorRaw("missing.gguf", info, 1).empty());
}

}  // namespace
