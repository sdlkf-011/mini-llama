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
#include "mini_llama/quant.h"
#include "mini_llama/thread_pool.h"

namespace {

using mini_llama::GetThreadCount;
using mini_llama::MatmulMode;
using mini_llama::ParallelFor;
using mini_llama::SetThreadCount;
using mini_llama::Tensor;

// 隔离各用例的线程数配置，结束时恢复自动模式。
class ThreadedMatmulTest : public ::testing::Test {
 protected:
  void SetUp() override { SetThreadCount(1); }
  void TearDown() override { SetThreadCount(0); }
};

// 生成确定性的小幅正负值，使各后端使用相同输入比较计算结果。
Tensor MakeInput(const std::vector<int>& shape, int offset = 0) {
  Tensor tensor(shape);
  for (size_t i = 0; i < tensor.size(); ++i) {
    tensor.data[i] = std::sin(static_cast<float>(i + offset) * 0.17f) * 0.2f;
  }
  return tensor;
}

// 同时检查输出形状和数值，允许 SIMD 归约及 FMA 带来的正常浮点误差。
void ExpectTensorNear(const Tensor& actual, const Tensor& expected) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual.data[i], expected.data[i],
                1e-5f * (1.0f + std::abs(expected.data[i])))
        << "element " << i;
  }
}

// 验证显式线程数生效，设为 0 时按硬件线程数或默认值选择自动配置。
TEST_F(ThreadedMatmulTest, ThreadCountSupportsExplicitAndAutomaticModes) {
  for (int threads : {1, 2, 4}) {
    SetThreadCount(threads);
    EXPECT_EQ(GetThreadCount(), threads);
  }
  SetThreadCount(0);
  const unsigned int hardware = std::thread::hardware_concurrency();
  EXPECT_EQ(GetThreadCount(), hardware > 0 ? static_cast<int>(hardware) : 4);
}

// 对比 naive、threaded、SIMD 和 threaded-SIMD，验证两种输入秩的结果一致。
// 长度覆盖 SIMD 整块、不足一块和 NEON 剩余 1～3 个元素的尾部。
TEST_F(ThreadedMatmulTest, LinearMatchesNaiveAndSingleThreadWithSimdTails) {
  for (int in_features : {3, 7, 8, 9, 10, 31, 33, 65}) {
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

// 用不能被线程数均分的 67 行验证通用 Matmul 的分片和各模式回退结果。
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

// 验证 Q8_0/Q4_0/Q4_1 的批量、多线程结果与逐行计算及反量化后的 naive Linear 一致。
// 覆盖小输出量、并行分片、两种输入秩、多行 batch，以及半块和跨块尾部。
TEST_F(ThreadedMatmulTest, QuantizedLinearMatchesSingleThreadWithBlockTails) {
  for (int out_features : {10, 48, 67}) {
    for (int in_features : {7, 8, 16, 17, 24, 31, 32, 33, 65}) {
      const Tensor weight = MakeInput({out_features, in_features}, 5);
      const auto q80 = mini_llama::QuantizeToQ80(weight);
      const auto q40 = mini_llama::QuantizeToQ40(weight);
      std::vector<mini_llama::BlockQ41> q41(q40.size());
      for (size_t i = 0; i < q41.size(); ++i) {
        q41[i].d = 0x3000;  // FP16 0.125
        q41[i].m = 0xb400;  // FP16 -0.25
        for (int k = 0; k < mini_llama::kQ41BlockSize / 2; ++k) {
          q41[i].qs[k] =
              static_cast<uint8_t>(((i + k) % 16) | (((i + 3 * k + 5) % 16) << 4));
        }
      }

      for (const auto& shape :
           {std::vector<int>{in_features}, std::vector<int>{1, in_features},
            std::vector<int>{3, in_features}, std::vector<int>{5, in_features}}) {
        const Tensor input = MakeInput(shape, 1);
        const int rows = input.num_dims() == 1 ? 1 : input.shape[0];
        const auto check = [&](const char* format, const auto& blocks,
                               const Tensor& decoded_weight, auto linear) {
          SCOPED_TRACE(::testing::Message()
                       << "format=" << format << " in_features=" << in_features
                       << " out_features=" << out_features << " rows=" << rows
                       << " input_rank=" << input.num_dims());
          SetThreadCount(1);
          const Tensor single_thread = linear(input, blocks, weight.shape);
          Tensor expected(input.num_dims() == 1
                              ? std::vector<int>{out_features}
                              : std::vector<int>{rows, out_features});
          for (int row = 0; row < rows; ++row) {
            Tensor row_input({in_features});
            for (int k = 0; k < in_features; ++k) {
              row_input.data[k] =
                  input.data[static_cast<size_t>(row) * in_features + k];
            }
            const Tensor row_result = linear(row_input, blocks, weight.shape);
            ExpectTensorNear(row_result,
                             mini_llama::LinearDispatch(row_input, decoded_weight,
                                                        MatmulMode::kNaive));
            for (int j = 0; j < out_features; ++j) {
              expected.data[static_cast<size_t>(row) * out_features + j] =
                  row_result.data[j];
            }
          }
          ExpectTensorNear(single_thread, expected);
          for (int threads : {1, 2, 3, 4}) {
            SCOPED_TRACE(::testing::Message() << "threads=" << threads);
            SetThreadCount(threads);
            ExpectTensorNear(linear(input, blocks, weight.shape), single_thread);
          }
        };
        check("Q8_0", q80, mini_llama::DequantizeFromQ80(q80, weight.shape),
              mini_llama::LinearQ80);
        check("Q4_0", q40, mini_llama::DequantizeFromQ40(q40, weight.shape),
              mini_llama::LinearQ40);
        check("Q4_1", q41, mini_llama::DequantizeFromQ41(q41, weight.shape),
              mini_llama::LinearQ41);
      }
    }
  }
}

// 验证批量 Linear 拒绝错误秩、特征维度、块数，以及非正 batch 和任务数溢出。
TEST_F(ThreadedMatmulTest, QuantizedLinearRejectsInvalidBatches) {
  const Tensor weight = MakeInput({2, 33}, 5);
  const auto q80 = mini_llama::QuantizeToQ80(weight);
  const auto q40 = mini_llama::QuantizeToQ40(weight);
  const std::vector<mini_llama::BlockQ41> q41(q40.size());
  const auto check = [&](const auto& blocks, auto linear) {
    EXPECT_THROW(linear(Tensor({3, 32}), blocks, weight.shape), std::runtime_error);
    EXPECT_THROW(linear(Tensor({1, 2, 33}), blocks, weight.shape),
                 std::runtime_error);
    Tensor invalid_input({1, 33});
    for (int rows : {0, -1, std::numeric_limits<int>::max()}) {
      // 只修改 shape 元数据，避免为非法 batch 分配巨大输入数组。
      invalid_input.shape[0] = rows;
      EXPECT_THROW(linear(invalid_input, blocks, weight.shape), std::runtime_error);
    }
    auto missing_blocks = blocks;
    missing_blocks.pop_back();
    EXPECT_THROW(linear(Tensor({3, 33}), missing_blocks, weight.shape),
                 std::runtime_error);
  };
  check(q80, mini_llama::LinearQ80);
  check(q40, mini_llama::LinearQ40);
  check(q41, mini_llama::LinearQ41);
}

// 验证 Q4_0 在加 8 编码前对有符号值舍入，正负半整数均向远离零的方向取整。
// 使用 scale=1 检查半整数两侧、低/高 nibble 及解码结果，固定现有舍入规则。
TEST_F(ThreadedMatmulTest, Q40RoundsSignedValuesBeforePacking) {
  const std::vector<float> values = {0.49f, 0.5f,   0.51f, -0.49f,
                                     -0.5f, -0.51f, 1.5f,  -1.5f};
  const std::vector<int> expected = {0, 1, 1, 0, -1, -1, 2, -2};
  Tensor weight({mini_llama::kQ40BlockSize}, 0.0f);
  weight.data[0] = 7.0f;
  weight.data[mini_llama::kQ40BlockSize / 2] = -7.0f;
  for (size_t i = 0; i < values.size(); ++i) {
    weight.data[i + 1] = values[i];
    weight.data[i + 1 + mini_llama::kQ40BlockSize / 2] = values[i];
  }
  const auto blocks = mini_llama::QuantizeToQ40(weight);
  ASSERT_EQ(blocks.size(), 1u);
  ASSERT_EQ(blocks[0].d, 0x3c00);  // FP16 1.0
  const Tensor decoded = mini_llama::DequantizeFromQ40(blocks, weight.shape);
  for (size_t i = 0; i < values.size(); ++i) {
    SCOPED_TRACE(::testing::Message() << "value=" << values[i]);
    EXPECT_EQ(blocks[0].qs[i + 1] & 0x0f, expected[i] + 8);
    EXPECT_EQ(blocks[0].qs[i + 1] >> 4, expected[i] + 8);
    EXPECT_FLOAT_EQ(decoded.data[i + 1], static_cast<float>(expected[i]));
    EXPECT_FLOAT_EQ(decoded.data[i + 1 + mini_llama::kQ40BlockSize / 2],
                    static_cast<float>(expected[i]));
  }
}

// 验证 Q8_0/Q4_0 兼容 Matmul 在整块和尾块下等价于反量化后执行 naive Matmul。
// 同时检查单线程、多线程以及输入维度不匹配和缺少量化块时的异常。
TEST_F(ThreadedMatmulTest, QuantizedMatmulCompatibilityHandlesBlocksAndTails) {
  for (int in_features : {31, 32, 33, 65}) {
    const Tensor weight = MakeInput({67, in_features}, 5);
    const Tensor input = MakeInput({in_features, 5}, 1);
    auto q80 = mini_llama::QuantizeToQ80(weight);
    auto q40 = mini_llama::QuantizeToQ40(weight);
    const Tensor reference_q80 = mini_llama::MatmulDispatch(
        mini_llama::DequantizeFromQ80(q80, weight.shape), input, MatmulMode::kNaive);
    const Tensor reference_q40 = mini_llama::MatmulDispatch(
        mini_llama::DequantizeFromQ40(q40, weight.shape), input, MatmulMode::kNaive);
    for (int threads : {1, 3}) {
      SCOPED_TRACE(::testing::Message()
                   << "threads=" << threads << " in_features=" << in_features);
      SetThreadCount(threads);
      ExpectTensorNear(mini_llama::MatmulQ80(q80, input, weight.shape),
                       reference_q80);
      ExpectTensorNear(mini_llama::MatmulQ40(q40, input, weight.shape),
                       reference_q40);
    }
    const Tensor invalid_input({in_features - 1, 5});
    EXPECT_THROW(mini_llama::MatmulQ80(q80, invalid_input, weight.shape),
                 std::runtime_error);
    EXPECT_THROW(mini_llama::MatmulQ40(q40, invalid_input, weight.shape),
                 std::runtime_error);
    q80.pop_back();
    q40.pop_back();
    EXPECT_THROW(mini_llama::MatmulQ80(q80, input, weight.shape),
                 std::runtime_error);
    EXPECT_THROW(mini_llama::MatmulQ40(q40, input, weight.shape),
                 std::runtime_error);
  }
}

// 用可手算的误差验证比较函数返回最大绝对误差，全零权重的误差应为 0。
// 同时覆盖两种 Linear 输入秩和维度不匹配时的异常传播。
TEST_F(ThreadedMatmulTest, QuantizedErrorHelpersReturnKnownMaximumError) {
  Tensor weight_q80({2, 3});
  weight_q80.data = {127.0f, 0.25f, 0.0f, 0.0f, 127.0f, 0.75f};
  Tensor matrix({3, 2});
  matrix.data = {1.0f, 2.0f, 2.0f, 1.0f, 3.0f, 4.0f};
  Tensor weight_q40({2, 3});
  weight_q40.data = {7.0f, 0.25f, 0.0f, 0.0f, 7.0f, 0.75f};
  // 两种权重的块级 scale 均为 1，误差最大值可以精确计算为 1。
  for (int threads : {1, 3}) {
    SetThreadCount(threads);
    EXPECT_FLOAT_EQ(mini_llama::CompareMatmulError(weight_q80, matrix), 1.0f);
    EXPECT_FLOAT_EQ(mini_llama::CompareMatmulError(Tensor({2, 3}, 0.0f), matrix),
                    0.0f);
    for (const auto& shape : {std::vector<int>{3}, std::vector<int>{1, 3}}) {
      Tensor input(shape);
      input.data = {1.0f, 2.0f, 4.0f};
      EXPECT_FLOAT_EQ(mini_llama::CompareQ40Error(weight_q40, input), 1.0f);
      EXPECT_FLOAT_EQ(mini_llama::CompareQ40Error(Tensor({2, 3}, 0.0f), input),
                      0.0f);
    }
  }
  EXPECT_THROW(mini_llama::CompareMatmulError(weight_q80, Tensor({2, 2})),
               std::runtime_error);
  EXPECT_THROW(mini_llama::CompareQ40Error(weight_q40, Tensor({2})),
               std::runtime_error);
}

// 验证小任务在调用线程执行且每项恰好处理一次，Linear/Matmul 的退化路径仍正确。
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

// 在 3 线程下用 48 项覆盖并行阈值、67 项覆盖余数分配，检查分片均衡且无重复遗漏。
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

// 验证单线程及相对任务量过大的线程配置均退化为调用线程上的一次完整处理。
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

// 验证零任务量和负任务量不会调用回调。
TEST_F(ThreadedMatmulTest, EmptyWorkDoesNotInvokeCallback) {
  for (int n : {0, -1}) {
    ParallelFor(n, [](int, int) { FAIL() << "empty work invoked callback"; });
  }
}

// 验证工作线程抛异常后仍等待全部线程结束，再向调用方传播异常。
// 同时检查调用线程直接执行的小任务也能传播异常。
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

// 验证含 bias 和 GQA 的连续 token 推理在不同线程数下保持相同结果。
// 比较 logits、KV 缓存、token 历史和位置，覆盖小 head 数与实际并行路径。
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
