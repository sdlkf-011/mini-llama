// 测试目的：验证 token 批次、位置编号及空输入行为。
#include "mini_llama/batch.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

using mini_llama::MiniBatch;

// 测试默认批次为空，token 与位置列表保持一致。
TEST(BatchTest, DefaultBatchIsEmpty) {
  const MiniBatch batch;

  EXPECT_EQ(batch.num_tokens(), 0);
  EXPECT_TRUE(batch.tokens.empty());
  EXPECT_TRUE(batch.positions.empty());
}

// 测试解码批次保留调用方指定的 token ID 和位置。
TEST(BatchTest, SingleKeepsTokenAndPosition) {
  const MiniBatch batch = MiniBatch::Single(127, 5);

  EXPECT_EQ(batch.num_tokens(), 1);
  EXPECT_EQ(batch.tokens, std::vector<int>({127}));
  EXPECT_EQ(batch.positions, std::vector<int>({5}));
}

// 测试预填充保留 token 顺序和重复项，并从默认或指定位置连续编号。
TEST(BatchTest, FromTokensAssignsConsecutivePositions) {
  const std::vector<int> tokens = {4, 127, 4};
  const MiniBatch initial = MiniBatch::FromTokens(tokens);
  const MiniBatch continued = MiniBatch::FromTokens(tokens, 8);

  EXPECT_EQ(initial.num_tokens(), 3);
  EXPECT_EQ(initial.tokens, tokens);
  EXPECT_EQ(initial.positions, std::vector<int>({0, 1, 2}));
  EXPECT_EQ(continued.tokens, tokens);
  EXPECT_EQ(continued.positions, std::vector<int>({8, 9, 10}));
}

// 测试空输入不会产生虚构 token 或位置，起始位置不影响空批次。
TEST(BatchTest, FromEmptyTokensProducesEmptyBatch) {
  const MiniBatch batch = MiniBatch::FromTokens({}, 8);

  EXPECT_EQ(batch.num_tokens(), 0);
  EXPECT_TRUE(batch.tokens.empty());
  EXPECT_TRUE(batch.positions.empty());
}

}  // namespace
