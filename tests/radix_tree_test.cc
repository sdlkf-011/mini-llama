// 测试目的：验证前缀树最长匹配、重复插入、清空和移动后的序列计数。
#include "mini_llama/radix_tree.h"

#include <gtest/gtest.h>

#include <utility>

using mini_llama::RadixTree;

// 测试压缩前缀树对共享前缀、完整序列和缺失分支返回最长匹配长度。
TEST(RadixTreeTest, LongestPrefixAndSize) {
  RadixTree tree;
  EXPECT_TRUE(tree.empty());
  tree.Insert({1, 2, 3});
  tree.Insert({1, 2, 4});
  tree.Insert({9});
  EXPECT_EQ(tree.size(), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 3, 8}), 3u);
  EXPECT_EQ(tree.LongestPrefix({1, 2, 7}), 2u);
  EXPECT_EQ(tree.LongestPrefix({8, 1}), 0u);
}

// 测试空序列、重复插入、清空以及移动构造不会破坏终止序列计数。
TEST(RadixTreeTest, EmptyDuplicateClearAndMove) {
  RadixTree tree;
  tree.Insert({});
  tree.Insert({});
  tree.Insert({4, 5});
  tree.Insert({4, 5});
  EXPECT_EQ(tree.size(), 2u);
  EXPECT_EQ(tree.LongestPrefix({}), 0u);
  RadixTree moved(std::move(tree));
  EXPECT_EQ(moved.size(), 2u);
  moved.Clear();
  EXPECT_TRUE(moved.empty());
  EXPECT_EQ(moved.LongestPrefix({4, 5}), 0u);
}
