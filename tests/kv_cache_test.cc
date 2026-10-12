// 测试目的：验证 CPU KV 缓存的层/位置/头寻址以及形状和越界检查。
#include "mini_llama/kv_cache.h"

#include <gtest/gtest.h>

using mini_llama::KvCache;
using mini_llama::MakeTensor2D;

// 测试 KV 写入按层、位置和头正确寻址，并可通过指针读取键和值。
TEST(KvCacheTest, WriteAndRead) {
  KvCache cache(2, 3, 2, 2);
  auto k = MakeTensor2D(2, 2);
  k.data = {1, 2, 3, 4};
  auto v = MakeTensor2D(2, 2);
  v.data = {5, 6, 7, 8};
  cache.Write(1, 2, k, v);
  EXPECT_EQ(cache.KeyPtr(1, 2, 0)[0], 1);
  EXPECT_EQ(cache.KeyPtr(1, 2, 1)[1], 4);
  EXPECT_EQ(cache.valuePtr(1, 2, 0)[1], 6);
  EXPECT_EQ(cache.valuePtr(1, 2, 1)[0], 7);
  EXPECT_EQ(cache.KeyPtr(0, 0, 0)[0], 0);
}

// 测试缓存写入和访问对错误秩、形状及越界参数进行拒绝。
TEST(KvCacheTest, RejectsInvalidInputs) {
  KvCache cache(1, 2, 1, 2);
  auto good = MakeTensor2D(1, 2);
  EXPECT_THROW(cache.Write(-1, 0, good, good), std::exception);
  EXPECT_THROW(cache.Write(0, 2, good, good), std::exception);
  EXPECT_THROW(cache.Write(0, 0, MakeTensor2D(2, 2), good), std::exception);
  EXPECT_THROW(cache.KeyPtr(0, 0, 1), std::exception);
  EXPECT_THROW(cache.valuePtr(0, -1, 0), std::exception);
}

// 空缓存、非法维度和截断存储都应报错，不能通过指针接口访问无效内存
TEST(KvCacheTest, RejectsInvalidCacheStorageBeforePointerAccess) {
  KvCache empty;
  EXPECT_THROW(empty.KeyPtr(0, 0, 0), std::runtime_error);
  EXPECT_THROW(empty.valuePtr(0, 0, 0), std::runtime_error);
  KvCache cache(1, 2, 1, 2);
  cache.keys.data.pop_back();
  cache.values.data.pop_back();
  EXPECT_THROW(cache.KeyPtr(0, 0, 0), std::runtime_error);
  EXPECT_THROW(cache.valuePtr(0, 0, 0), std::runtime_error);
  cache.values.shape = {1, 2};
  EXPECT_THROW(cache.valuePtr(0, 0, 0), std::runtime_error);
  cache.keys.shape[3] = -1;
  EXPECT_THROW(cache.KeyPtr(0, 0, 0), std::runtime_error);
}

// K/V 输入或缓存形状损坏时，应在复制前报错，避免键已写入而值写入失败
TEST(KvCacheTest, InvalidWritePreservesBothCaches) {
  KvCache cache(1, 2, 1, 2);
  auto good = MakeTensor2D(1, 2, 3.0f);
  auto truncated = good;
  truncated.data.pop_back();
  const auto keys = cache.keys.data;
  const auto values = cache.values.data;
  EXPECT_THROW(cache.Write(0, 0, good, truncated), std::runtime_error);
  EXPECT_THROW(cache.Write(0, 0, truncated, good), std::runtime_error);
  cache.values.shape = {1, 1, 2, 2};
  EXPECT_THROW(cache.Write(0, 0, good, good), std::runtime_error);
  EXPECT_EQ(cache.keys.data, keys);
  EXPECT_EQ(cache.values.data, values);
}
