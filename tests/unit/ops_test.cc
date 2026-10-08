#include "mini_llama/ops.h"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "mini_llama/matmul_dispatch.h"

namespace {

using mini_llama::MatmulMode;
using mini_llama::Tensor;

void Check(bool condition, const std::string& message) {
  EXPECT_TRUE(condition) << message;
}

void CheckNear(float actual, float expected, float tolerance = 1e-5f) {
  EXPECT_NEAR(actual, expected, tolerance);
}

void CheckTensor(const Tensor& actual, const std::vector<int>& expected_shape,
                 const std::vector<float>& expected_data, float tolerance = 1e-5f) {
  Check(actual.shape == expected_shape, "tensor shape mismatch");
  Check(actual.data.size() == expected_data.size(), "tensor size mismatch");
  for (size_t i = 0; i < expected_data.size(); ++i) {
    CheckNear(actual.data[i], expected_data[i], tolerance);
  }
}

void ExpectThrows(const std::function<void()>& operation) {
  EXPECT_THROW(operation(), std::exception);
}

// 测试小矩阵乘法的数值结果，并覆盖当前已实现的朴素 dispatch 模式。
TEST(OpsTest, MatmulDispatch) {
  Tensor a({2, 3}, 0.0f);
  a.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor b({3, 2}, 0.0f);
  b.data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};
  CheckTensor(mini_llama::MatmulDispatch(a, b, MatmulMode::kNaive), {2, 2},
              expected);
  CheckTensor(mini_llama::Matmul(a, b), {2, 2}, expected);
  ExpectThrows([&] {
    mini_llama::MatmulDispatch(Tensor({2, 3}), Tensor({2, 2}), MatmulMode::kNaive);
  });
  ExpectThrows(
      [&] { mini_llama::MatmulDispatch(Tensor({3}), b, MatmulMode::kNaive); });
  // 未登记的枚举值必须被拒绝，不能静默回退到某个实现。
  ExpectThrows(
      [&] { mini_llama::MatmulDispatch(a, b, static_cast<MatmulMode>(-1)); });
}

// 测试 Linear 对一维输入和 [1, in_features] 输入的形状及转置权重计算。
TEST(OpsTest, Linear) {
  Tensor weight({2, 3}, 0.0f);
  weight.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor input({3}, 0.0f);
  input.data = {1.0f, 2.0f, 3.0f};
  CheckTensor(mini_llama::Linear(input, weight), {2}, {14.0f, 32.0f});

  Tensor batched({1, 3}, 0.0f);
  batched.data = input.data;
  CheckTensor(mini_llama::LinearDispatch(batched, weight, MatmulMode::kNaive),
              {1, 2}, {14.0f, 32.0f});
  ExpectThrows([&] {
    mini_llama::LinearDispatch(input, weight, static_cast<MatmulMode>(-1));
  });
  ExpectThrows([&] { mini_llama::Linear(input, Tensor({3})); });
  ExpectThrows([&] { mini_llama::Linear(Tensor({2, 3}), weight); });
}

// 测试 RMSNorm 的均方根归一化、权重缩放和维度校验。
TEST(OpsTest, RmsNorm) {
  Tensor input({2}, 0.0f);
  input.data = {3.0f, 4.0f};
  Tensor weight({2}, 1.0f);
  CheckTensor(mini_llama::RmsNorm(input, weight, 0.0f), {2},
              {0.84852814f, 1.1313709f}, 1e-5f);
  ExpectThrows([&] { mini_llama::RmsNorm(input, Tensor({3}, 1.0f), 1e-5f); });
  ExpectThrows([&] { mini_llama::RmsNorm(Tensor({1, 2}), weight, 1e-5f); });
}

// 测试带最大值平移的 Softmax，确认概率和为一且不会溢出。
TEST(OpsTest, Softmax) {
  Tensor input({3}, 0.0f);
  input.data = {1000.0f, 1001.0f, 999.0f};
  Tensor output = mini_llama::Softmax(input);
  const float denominator = 1.0f + std::exp(-1.0f) + std::exp(-2.0f);
  CheckNear(output.data[0], std::exp(-1.0f) / denominator, 1e-6f);
  CheckNear(output.data[1], 1.0f / denominator, 1e-6f);
  CheckNear(output.data[2], std::exp(-2.0f) / denominator, 1e-6f);
  CheckNear(output.data[0] + output.data[1] + output.data[2], 1.0f, 1e-6f);
  ExpectThrows([&] { mini_llama::Softmax(Tensor({1, 3})); });
  ExpectThrows([] { mini_llama::Softmax(Tensor()); });
}

// 测试 SiLU 的逐元素公式，并保留输入张量形状。
TEST(OpsTest, Silu) {
  Tensor input({2}, 0.0f);
  input.data = {-1.0f, 1.0f};
  CheckTensor(mini_llama::Silu(input), {2},
              {-1.0f / (1.0f + std::exp(1.0f)), 1.0f / (1.0f + std::exp(-1.0f))});
}

// 测试逐元素乘法的数值结果和形状不匹配错误。
TEST(OpsTest, ElementwiseMul) {
  Tensor a({2, 2}, 0.0f);
  a.data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor b({2, 2}, 2.0f);
  CheckTensor(mini_llama::ElementwiseMul(a, b), {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  ExpectThrows([&] { mini_llama::ElementwiseMul(a, Tensor({4}, 1.0f)); });
}

// 测试 SwiGLU 对每个 gate 元素分别执行 SiLU 后再与 up 相乘。
TEST(OpsTest, SwiGlu) {
  Tensor gate({2}, 0.0f);
  gate.data = {-1.0f, 2.0f};
  Tensor up({2}, 3.0f);
  CheckTensor(mini_llama::SwiGlu(gate, up), {2},
              {-3.0f / (1.0f + std::exp(1.0f)), 6.0f / (1.0f + std::exp(-2.0f))});
  ExpectThrows([&] { mini_llama::SwiGlu(gate, Tensor({1}, 1.0f)); });
}

// 测试 ArgMax 返回展平索引，并在并列最大值时保留首次出现的位置。
TEST(OpsTest, ArgMax) {
  Tensor tensor({2, 2}, 0.0f);
  tensor.data = {1.0f, 5.0f, 5.0f, 2.0f};
  Check(mini_llama::ArgMax(tensor) == 1, "ArgMax tie handling mismatch");
  ExpectThrows([] { mini_llama::ArgMax(Tensor()); });
}

// 测试位置零不改变 Q/K，且查询头与 KV 头数量不同的 GQA 输入可正常旋转。
TEST(OpsTest, RopeAtPositionZeroPreservesQueryAndKey) {
  for (auto type : {mini_llama::RopeType::kNormal, mini_llama::RopeType::kNeoX}) {
    Tensor query({2, 4}, 0.0f);
    query.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    Tensor key({1, 4}, 0.0f);
    key.data = {9.0f, 10.0f, 11.0f, 12.0f};
    const auto original_query = query.data;
    const auto original_key = key.data;

    mini_llama::Rope(query, key, 0, 4.0f, type);

    EXPECT_EQ(query.data, original_query);
    EXPECT_EQ(key.data, original_key);
  }
}

// 测试普通 RoPE 配对相邻维度，NeoX 配对前后半维，并使用各自的旋转频率。
TEST(OpsTest, RopeUsesNormalAndNeoXDimensionPairing) {
  const float cosine = std::cos(1.0f);
  const float sine = std::sin(1.0f);
  const float half_cosine = std::cos(0.5f);
  const float half_sine = std::sin(0.5f);
  Tensor query({1, 4}, 0.0f);
  query.data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor key = query;

  mini_llama::Rope(query, key, 1, 4.0f, mini_llama::RopeType::kNormal);
  CheckTensor(query, {1, 4},
              {cosine - 2.0f * sine, sine + 2.0f * cosine,
               3.0f * half_cosine - 4.0f * half_sine,
               3.0f * half_sine + 4.0f * half_cosine});
  EXPECT_EQ(query.data, key.data);

  query.data = {1.0f, 2.0f, 3.0f, 4.0f};
  key = query;
  mini_llama::Rope(query, key, 1, 4.0f, mini_llama::RopeType::kNeoX);
  CheckTensor(query, {1, 4},
              {cosine - 3.0f * sine, 2.0f * half_cosine - 4.0f * half_sine,
               sine + 3.0f * cosine, 2.0f * half_sine + 4.0f * half_cosine});
  EXPECT_EQ(query.data, key.data);
}

// 测试 RoPE 拒绝错误秩、奇数头维度、Q/K 维度不一致及非法位置和基频。
TEST(OpsTest, RopeRejectsInvalidShapesPositionAndTheta) {
  Tensor query({1, 4}, 1.0f);
  Tensor key({1, 4}, 1.0f);
  ExpectThrows([&] { mini_llama::Rope(query, key, -1, 4.0f); });
  ExpectThrows([&] { mini_llama::Rope(query, key, 0, 0.0f); });
  ExpectThrows([&] { mini_llama::Rope(query, key, 0, -1.0f); });
  Tensor flat({4}, 1.0f);
  ExpectThrows([&] { mini_llama::Rope(flat, key, 0, 4.0f); });
  Tensor odd({1, 3}, 1.0f);
  ExpectThrows([&] { mini_llama::Rope(odd, odd, 0, 4.0f); });
  Tensor mismatched({1, 2}, 1.0f);
  ExpectThrows([&] { mini_llama::Rope(query, mismatched, 0, 4.0f); });
}

}  // namespace
