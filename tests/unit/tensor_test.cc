#include "mini_llama/tensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/ops.h"

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

void ExpectThrowsContaining(const std::function<void()>& operation,
                            const std::string& expected_text) {
  EXPECT_THROW(
      {
        try {
          operation();
        } catch (const std::exception& error) {
          EXPECT_NE(std::string(error.what()).find(expected_text),
                    std::string::npos);
          throw;
        }
      },
      std::exception);
}

// 测试不同秩张量的构造、填充值、元素数量和便捷构造函数。
TEST(TensorTest, Construction) {
  Tensor tensor({2, 3, 4}, 7.0f);
  Check(tensor.shape == std::vector<int>({2, 3, 4}), "shape mismatch");
  Check(tensor.size() == 24 && tensor.num_elements() == 24,
        "element count mismatch");
  Check(tensor.num_dims() == 3, "rank mismatch");
  Check(tensor[0] == 7.0f && tensor[23] == 7.0f, "fill mismatch");
  CheckTensor(mini_llama::MakeTensor1D(3, 1.0f), {3}, {1.0f, 1.0f, 1.0f});
  CheckTensor(mini_llama::MakeTensor2D(2, 2, -1.0f), {2, 2},
              {-1.0f, -1.0f, -1.0f, -1.0f});
  Check(mini_llama::MakeTensor3D(1, 2, 3).num_dims() == 3,
        "3D helper rank mismatch");
  Check(mini_llama::MakeTensor4D(1, 1, 2, 2).num_dims() == 4,
        "4D helper rank mismatch");
  Tensor scalar({}, 3.0f);
  Check(scalar.size() == 1 && scalar.FlatIndex({}) == 0 && scalar.At({}) == 3.0f,
        "scalar tensor mismatch");
}

// 测试多维索引展开公式，以及 At/At1-At4 的读写一致性。
TEST(TensorTest, Indexing) {
  Tensor tensor({2, 3, 4}, 0.0f);
  Check(tensor.FlatIndex({1, 2, 3}) == 23, "flat index mismatch");
  tensor.At({1, 2, 3}) = 9.0f;
  Check(tensor.At3(1, 2, 3) == 9.0f, "At3 mismatch");

  Tensor one({3}, 0.0f);
  one.At1(1) = 2.0f;
  Check(one.At1(1) == 2.0f, "At1 mismatch");
  Tensor two({2, 3}, 0.0f);
  two.At2(1, 2) = 3.0f;
  Check(two.At2(1, 2) == 3.0f, "At2 mismatch");
  Tensor four({1, 1, 2, 2}, 0.0f);
  four.At4(0, 0, 1, 1) = 4.0f;
  Check(four.At4(0, 0, 1, 1) == 4.0f, "At4 mismatch");
}

// 测试二维行指针访问，并确认 const 访问不会改变数据。
TEST(TensorTest, RowPtr) {
  Tensor tensor({2, 3}, 0.0f);
  float* row = tensor.RowPtr(1);
  row[0] = 4.0f;
  row[2] = 6.0f;
  CheckTensor(tensor, {2, 3}, {0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 6.0f});
  const Tensor& const_tensor = tensor;
  Check(const_tensor.RowPtr(1)[2] == 6.0f, "const RowPtr mismatch");
}

// 测试非法维度、错误秩和越界索引均被明确拒绝。
TEST(TensorTest, Validation) {
  ExpectThrows([] { Tensor({0, 2}); });
  ExpectThrows([] { Tensor({-1}); });
  Tensor tensor({2, 3}, 0.0f);
  ExpectThrows([&] { tensor.FlatIndex({1}); });
  ExpectThrows([&] { tensor.FlatIndex({2, 0}); });
  ExpectThrows([&] { tensor.At1(0); });
  ExpectThrows([&] { tensor.RowPtr(2); });
  ExpectThrows([&] { tensor.At2(-1, 0); });
}

// 测试 shape 字符串、AssertShape 以及保持元素数量的 ReshapeChecked。
TEST(TensorTest, ShapeOperations) {
  Tensor tensor({2, 3}, 0.0f);
  tensor[5] = 5.0f;
  Check(tensor.ShapeStringShort() == "[2, 3]", "shape string mismatch");
  tensor.AssertShape({2, 3}, "ShapeTest");
  ExpectThrowsContaining([&] { tensor.AssertShape({3, 2}, "ShapeTest"); },
                         "ShapeTest");
  Tensor reshaped = tensor.ReshapeChecked({3, 2}, "ShapeTest");
  CheckTensor(reshaped, {3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f});
  Check(tensor.shape == std::vector<int>({2, 3}), "reshape changed source");
  ExpectThrows([&] { tensor.ReshapeChecked({4, 2}, "ShapeTest"); });
}

// 测试小矩阵乘法的数值结果，并覆盖四种 dispatch 模式。
TEST(TensorTest, MatmulDispatch) {
  Tensor a({2, 3}, 0.0f);
  a.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor b({3, 2}, 0.0f);
  b.data = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  const std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};
  for (MatmulMode mode : {MatmulMode::kNaive, MatmulMode::kThreaded,
                          MatmulMode::kSimd, MatmulMode::kThreadedSimd}) {
    CheckTensor(mini_llama::MatmulDispatch(a, b, mode), {2, 2}, expected);
  }
  CheckTensor(mini_llama::Matmul(a, b), {2, 2}, expected);
  ExpectThrows([&] {
    mini_llama::MatmulDispatch(Tensor({2, 3}), Tensor({2, 2}), MatmulMode::kNaive);
  });
  ExpectThrows(
      [&] { mini_llama::MatmulDispatch(Tensor({3}), b, MatmulMode::kNaive); });
}

// 测试 Linear 对一维输入和 [1, in_features] 输入的形状及转置权重计算。
TEST(TensorTest, Linear) {
  Tensor weight({2, 3}, 0.0f);
  weight.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Tensor input({3}, 0.0f);
  input.data = {1.0f, 2.0f, 3.0f};
  CheckTensor(mini_llama::Linear(input, weight), {2}, {14.0f, 32.0f});

  Tensor batched({1, 3}, 0.0f);
  batched.data = input.data;
  CheckTensor(mini_llama::LinearDispatch(batched, weight, MatmulMode::kNaive),
              {1, 2}, {14.0f, 32.0f});
  ExpectThrows([&] { mini_llama::Linear(input, Tensor({3})); });
  ExpectThrows([&] { mini_llama::Linear(Tensor({2, 3}), weight); });
}

// 测试 RMSNorm 的均方根归一化、权重缩放和维度校验。
TEST(TensorTest, RmsNorm) {
  Tensor input({2}, 0.0f);
  input.data = {3.0f, 4.0f};
  Tensor weight({2}, 1.0f);
  CheckTensor(mini_llama::RmsNorm(input, weight, 0.0f), {2},
              {0.84852814f, 1.1313709f}, 1e-5f);
  ExpectThrows([&] { mini_llama::RmsNorm(input, Tensor({3}, 1.0f), 1e-5f); });
  ExpectThrows([&] { mini_llama::RmsNorm(Tensor({1, 2}), weight, 1e-5f); });
}

// 测试带最大值平移的 Softmax，确认概率和为一且不会溢出。
TEST(TensorTest, Softmax) {
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
TEST(TensorTest, Silu) {
  Tensor input({2}, 0.0f);
  input.data = {-1.0f, 1.0f};
  CheckTensor(mini_llama::Silu(input), {2},
              {-1.0f / (1.0f + std::exp(1.0f)), 1.0f / (1.0f + std::exp(-1.0f))});
}

// 测试逐元素乘法的数值结果和形状不匹配错误。
TEST(TensorTest, ElementwiseMul) {
  Tensor a({2, 2}, 0.0f);
  a.data = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor b({2, 2}, 2.0f);
  CheckTensor(mini_llama::ElementwiseMul(a, b), {2, 2}, {2.0f, 4.0f, 6.0f, 8.0f});
  ExpectThrows([&] { mini_llama::ElementwiseMul(a, Tensor({4}, 1.0f)); });
}

// 测试 SwiGLU 对每个 gate 元素分别执行 SiLU 后再与 up 相乘。
TEST(TensorTest, SwiGlu) {
  Tensor gate({2}, 0.0f);
  gate.data = {-1.0f, 2.0f};
  Tensor up({2}, 3.0f);
  CheckTensor(mini_llama::SwiGlu(gate, up), {2},
              {-3.0f / (1.0f + std::exp(1.0f)), 6.0f / (1.0f + std::exp(-2.0f))});
  ExpectThrows([&] { mini_llama::SwiGlu(gate, Tensor({1}, 1.0f)); });
}

// 测试 ArgMax 返回展平索引，并在并列最大值时保留首次出现的位置。
TEST(TensorTest, ArgMax) {
  Tensor tensor({2, 2}, 0.0f);
  tensor.data = {1.0f, 5.0f, 5.0f, 2.0f};
  Check(mini_llama::ArgMax(tensor) == 1, "ArgMax tie handling mismatch");
  ExpectThrows([] { mini_llama::ArgMax(Tensor()); });
}

}  // namespace
