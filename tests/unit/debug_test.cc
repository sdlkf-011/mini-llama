#include "mini_llama/debug.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>

#include "mini_llama/kv_cache.h"
#include "mini_llama/tensor.h"

namespace {

// 测试非正数或超过词表大小的 k 会展示完整词表，并按 logits 降序输出。
TEST(DebugUnitTest, InvalidTopKShowsCompleteVocabulary) {
  mini_llama::Tensor logits({3}, 0.0f);
  logits.data = {1.0f, 3.0f, 2.0f};

  for (int k : {-1, 0, 4}) {
    SCOPED_TRACE(k);
    testing::internal::CaptureStdout();
    mini_llama::DumpLogitsTopK(logits, k);
    const std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("logits top-3"), std::string::npos);
    const size_t first = output.find("id=   1");
    const size_t second = output.find("id=   2");
    const size_t third = output.find("id=   0");
    ASSERT_NE(first, std::string::npos);
    ASSERT_NE(second, std::string::npos);
    ASSERT_NE(third, std::string::npos);
    EXPECT_LT(first, second);
    EXPECT_LT(second, third);
  }
}

// 测试 Top-k 临时设置的小数格式不会改变调用方输出流的 flags 和 precision。
TEST(DebugUnitTest, TopKPreservesOutputStreamFormatting) {
  const mini_llama::Tensor logits({1}, 1.0f);
  const auto flags = std::cout.flags();
  const auto precision = std::cout.precision();

  testing::internal::CaptureStdout();
  mini_llama::DumpLogitsTopK(logits, 1);
  testing::internal::GetCapturedStdout();

  EXPECT_EQ(std::cout.flags(), flags);
  EXPECT_EQ(std::cout.precision(), precision);
}

// 测试未提供当前位置时省略位置行，但仍保留 KV 缓存的结构信息。
TEST(DebugUnitTest, KvCacheOmitsUnspecifiedPosition) {
  const mini_llama::KvCache cache(1, 2, 1, 2);

  testing::internal::CaptureStdout();
  mini_llama::DumpKvCacheInfo(cache, -1);
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_EQ(output.find("current_pos:"), std::string::npos);
  EXPECT_NE(output.find("keys shape: [1, 2, 1, 2]"), std::string::npos);
  EXPECT_NE(output.find("values shape: [1, 2, 1, 2]"), std::string::npos);
}

}  // namespace
