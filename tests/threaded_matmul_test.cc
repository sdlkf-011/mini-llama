#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mini_llama/forward.h"
#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

namespace {

using mini_llama::GetThreadCount;
using mini_llama::MatmulMode;
using mini_llama::ParallelFor;
using mini_llama::SetThreadCount;
using mini_llama::Tensor;

class ThreadedMatmulTest : public ::testing::Test {
 protected:
  void SetUp() override { SetThreadCount(1); }
  void TearDown() override { SetThreadCount(0); }
};

Tensor MakeInput(const std::vector<int>& shape, int offset = 0) {
  Tensor tensor(shape);
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor.data[i] = std::sin(static_cast<float>(i + offset) * 0.17f) * 0.2f;
  }
  return tensor;
}

void ExpectTensorNear(const Tensor& actual, const Tensor& expected) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual.data[i], expected.data[i],
                1e-5f * (1.0f + std::abs(expected.data[i])))
        << "element " << i;
  }
}

TEST_F(ThreadedMatmulTest, ThreadCountSupportsExplicitAndAutomaticModes) {
  for (int threads : {1, 2, 4}) {
    SetThreadCount(threads);
    EXPECT_EQ(GetThreadCount(), threads);
  }
  SetThreadCount(0);
  const unsigned int hardware = std::thread::hardware_concurrency();
  EXPECT_EQ(GetThreadCount(), hardware > 0 ? static_cast<int>(hardware) : 4);
}

TEST_F(ThreadedMatmulTest, LinearMatchesNaiveAndSingleThreadWithSimdTails) {
  for (int in_features : {3, 7, 8, 9, 31, 33, 65}) {
    const Tensor weight = MakeInput({67, in_features}, 5);
    for (const auto& shape :
         {std::vector<int>{in_features}, std::vector<int>{1, in_features}}) {
      const Tensor input = MakeInput(shape, 1);
      SetThreadCount(1);
      const Tensor reference =
          mini_llama::LinearDispatch(input, weight, MatmulMode::kNaive);
      const Tensor single_thread = mini_llama::Linear(input, weight);
      for (int threads : {1, 2, 4}) {
        SCOPED_TRACE(::testing::Message()
                     << "threads=" << threads << " in_features=" << in_features);
        SetThreadCount(threads);
        ExpectTensorNear(mini_llama::Linear(input, weight), single_thread);
        for (MatmulMode mode :
             {MatmulMode::kThreaded, MatmulMode::kSimd, MatmulMode::kThreadedSimd}) {
          ExpectTensorNear(mini_llama::LinearDispatch(input, weight, mode),
                           reference);
        }
      }
    }
  }
}

TEST_F(ThreadedMatmulTest, MatmulMatchesNaiveAndSingleThreadWithUnevenRows) {
  const Tensor a = MakeInput({67, 13}, 2);
  const Tensor b = MakeInput({13, 9}, 7);
  const Tensor reference = mini_llama::MatmulDispatch(a, b, MatmulMode::kNaive);
  const Tensor single_thread = mini_llama::Matmul(a, b);
  for (int threads : {1, 2, 4}) {
    SetThreadCount(threads);
    ExpectTensorNear(mini_llama::Matmul(a, b), single_thread);
    for (MatmulMode mode :
         {MatmulMode::kThreaded, MatmulMode::kSimd, MatmulMode::kThreadedSimd}) {
      ExpectTensorNear(mini_llama::MatmulDispatch(a, b, mode), reference);
    }
  }
}

TEST_F(ThreadedMatmulTest, SmallWorkRunsOnCallerAndVisitsEachTaskOnce) {
  SetThreadCount(3);
  const auto caller = std::this_thread::get_id();
  std::vector<int> visits(10, 0);
  int calls = 0;
  ParallelFor(10, [&](int begin, int end) {
    EXPECT_EQ(std::this_thread::get_id(), caller);
    EXPECT_EQ(begin, 0);
    EXPECT_EQ(end, 10);
    ++calls;
    for (int i = begin; i < end; ++i) {
      ++visits[i];
    }
  });
  EXPECT_EQ(calls, 1);
  for (int count : visits) {
    EXPECT_EQ(count, 1);
  }

  const Tensor input = MakeInput({7});
  const Tensor weight = MakeInput({10, 7}, 3);
  const Tensor a = MakeInput({10, 7}, 1);
  const Tensor b = MakeInput({7, 3}, 4);
  ExpectTensorNear(mini_llama::Linear(input, weight),
                   mini_llama::LinearDispatch(input, weight, MatmulMode::kNaive));
  ExpectTensorNear(mini_llama::Matmul(a, b),
                   mini_llama::MatmulDispatch(a, b, MatmulMode::kNaive));
}

TEST_F(ThreadedMatmulTest, ThresholdAndRemaindersProduceBalancedDisjointRanges) {
  for (int n : {48, 67}) {
    SetThreadCount(3);
    const auto caller = std::this_thread::get_id();
    std::vector<std::atomic<int>> visits(n);
    for (auto& count : visits) {
      count.store(0);
    }
    std::vector<int> range_ends(n, -1);
    std::atomic<int> calls{0};
    ParallelFor(n, [&](int begin, int end) {
      EXPECT_NE(std::this_thread::get_id(), caller);
      ++calls;
      range_ends[begin] = end;
      for (int i = begin; i < end; ++i) {
        ++visits[i];
      }
    });
    EXPECT_EQ(calls.load(), 3);
    int begin = 0;
    for (int t = 0; t < 3; ++t) {
      const int end = begin + n / 3 + (t < n % 3 ? 1 : 0);
      EXPECT_EQ(range_ends[begin], end);
      begin = end;
    }
    EXPECT_EQ(begin, n);
    for (const auto& count : visits) {
      EXPECT_EQ(count.load(), 1);
    }
  }
}

TEST_F(ThreadedMatmulTest, SingleThreadAndLargeThreadCountsAvoidExtraWorkers) {
  const auto caller = std::this_thread::get_id();
  for (int threads : {1, std::numeric_limits<int>::max()}) {
    SetThreadCount(threads);
    int calls = 0;
    ParallelFor(67, [&](int begin, int end) {
      EXPECT_EQ(std::this_thread::get_id(), caller);
      EXPECT_EQ(begin, 0);
      EXPECT_EQ(end, 67);
      ++calls;
    });
    EXPECT_EQ(calls, 1);
  }
}

TEST_F(ThreadedMatmulTest, EmptyWorkDoesNotInvokeCallback) {
  for (int n : {0, -1}) {
    ParallelFor(n, [](int, int) { FAIL() << "empty work invoked callback"; });
  }
}

TEST_F(ThreadedMatmulTest, CallbackExceptionsPropagateAfterAllWorkersFinish) {
  SetThreadCount(4);
  std::vector<std::atomic<int>> visits(67);
  for (auto& count : visits) {
    count.store(0);
  }
  std::atomic<int> completed{0};
  EXPECT_THROW(ParallelFor(67,
                           [&](int begin, int end) {
                             for (int i = begin; i < end; ++i) {
                               ++visits[i];
                             }
                             ++completed;
                             throw std::runtime_error("callback failed");
                           }),
               std::runtime_error);
  EXPECT_EQ(completed.load(), 4);
  for (const auto& count : visits) {
    EXPECT_EQ(count.load(), 1);
  }
  EXPECT_THROW(ParallelFor(1, [](int, int) { throw std::runtime_error("small"); }),
               std::runtime_error);
}

// 小 head 数覆盖退化路径；68 个 head 在 4 线程下真正并行，且共享 GQA KV 头。
mini_llama::MiniLlamaModel MakeAttentionModel(int n_heads) {
  mini_llama::MiniLlamaModel model;
  auto& config = model.config;
  config.n_heads = n_heads;
  config.n_kv_heads = 4;
  config.head_dim = 2;
  config.dim = n_heads * config.head_dim;
  config.vocab_size = 67;
  config.hidden_dim = 65;
  config.n_layers = 1;
  config.max_seq_len = 4;
  model.loaded = true;
  model.token_embedding = MakeInput({config.vocab_size, config.dim}, 1);
  model.final_norm = Tensor({config.dim}, 1.0f);
  model.lm_head =
      mini_llama::ToQuantizedTensor(MakeInput({config.vocab_size, config.dim}, 11));
  model.layers.resize(1);
  auto& layer = model.layers[0];
  const int kv_dim = config.n_kv_heads * config.head_dim;
  layer.attention_norm = Tensor({config.dim}, 1.0f);
  layer.wq = mini_llama::ToQuantizedTensor(MakeInput({config.dim, config.dim}, 2));
  layer.wk = mini_llama::ToQuantizedTensor(MakeInput({kv_dim, config.dim}, 3));
  layer.wv = mini_llama::ToQuantizedTensor(MakeInput({kv_dim, config.dim}, 4));
  layer.wo = mini_llama::ToQuantizedTensor(MakeInput({config.dim, config.dim}, 5));
  layer.bq = MakeInput({config.dim}, 6);
  layer.bk = MakeInput({kv_dim}, 7);
  layer.bv = MakeInput({kv_dim}, 8);
  layer.ffn_norm = Tensor({config.dim}, 1.0f);
  layer.w_gate =
      mini_llama::ToQuantizedTensor(MakeInput({config.hidden_dim, config.dim}, 9));
  layer.w_up =
      mini_llama::ToQuantizedTensor(MakeInput({config.hidden_dim, config.dim}, 10));
  layer.w_down =
      mini_llama::ToQuantizedTensor(MakeInput({config.dim, config.hidden_dim}, 12));
  return model;
}

TEST_F(ThreadedMatmulTest, AttentionWithBiasAndGqaMatchesSingleThreadAndKvCache) {
  for (int n_heads : {4, 68}) {
    const auto model = MakeAttentionModel(n_heads);
    SetThreadCount(1);
    mini_llama::MiniLlamaContext reference_context(&model);
    std::vector<Tensor> expected;
    for (int pos = 0; pos < 3; ++pos) {
      expected.push_back(mini_llama::ForwardBatch(
          reference_context, model, mini_llama::MiniBatch::Single(pos, pos)));
    }
    for (int threads : {2, 4}) {
      SCOPED_TRACE(::testing::Message()
                   << "threads=" << threads << " n_heads=" << n_heads);
      SetThreadCount(threads);
      mini_llama::MiniLlamaContext context(&model);
      for (int pos = 0; pos < 3; ++pos) {
        ExpectTensorNear(
            mini_llama::ForwardBatch(context, model,
                                     mini_llama::MiniBatch::Single(pos, pos)),
            expected[pos]);
      }
      ExpectTensorNear(context.kv_cache.keys, reference_context.kv_cache.keys);
      ExpectTensorNear(context.kv_cache.values, reference_context.kv_cache.values);
      EXPECT_EQ(context.token_history, reference_context.token_history);
      EXPECT_EQ(context.pos, reference_context.pos);
    }
  }
}

}  // namespace
