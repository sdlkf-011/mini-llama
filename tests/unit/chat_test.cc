#include <gtest/gtest.h>

#include "mini_llama/context.h"

// 测试聊天推理上下文在无模型和有模型时的默认状态及 token 历史容器。
TEST(ChatTest, ContextLifecycle) {
  mini_llama::MiniLlamaContext empty;
  EXPECT_EQ(empty.model, nullptr);
  EXPECT_EQ(empty.pos, 0);
  EXPECT_TRUE(empty.token_history.empty());

  mini_llama::MiniLlamaModel model;
  model.config.n_layers = 2;
  model.config.max_seq_len = 4;
  model.config.n_kv_heads = 1;
  model.config.head_dim = 2;
  mini_llama::MiniLlamaContext context(&model);
  EXPECT_EQ(context.model, &model);
  EXPECT_EQ(context.kv_cache.keys.shape, std::vector<int>({2, 4, 1, 2}));
  context.token_history = {1, 2, 3};
  context.pos = 3;
  EXPECT_EQ(context.token_history.size(), 3u);
  EXPECT_EQ(context.pos, 3);
}
