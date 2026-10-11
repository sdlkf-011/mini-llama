// 测试目的：验证 CPU Q8_0/Q4_0/Q4_1 量化、反量化及 Linear 的数值误差。
#include "mini_llama/quant.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "mini_llama/ops.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace {

using mini_llama::BlockQ41;
using mini_llama::DequantizeFromQ40;
using mini_llama::DequantizeFromQ80;
using mini_llama::Linear;
using mini_llama::LinearQ40;
using mini_llama::LinearQ41;
using mini_llama::LinearQ80;
using mini_llama::QuantizeToQ40;
using mini_llama::QuantizeToQ80;
using mini_llama::Tensor;

void ExpectNear(const Tensor& actual, const Tensor& expected, float tolerance) {
  ASSERT_EQ(actual.shape, expected.shape);
  ASSERT_EQ(actual.data.size(), expected.data.size());
  for (size_t index = 0; index < actual.data.size(); ++index) {
    EXPECT_NEAR(actual.data[index], expected.data[index], tolerance);
  }
}

Tensor MakeWeight() {
  Tensor weight({2, 32}, 0.0f);
  for (size_t index = 0; index < weight.size(); ++index) {
    weight.data[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 3.0f;
  }
  return weight;
}

// 测试目的：验证 Q8_0 往返误差受控，且直接量化线性层等价于反量化后计算。
TEST(QuantTest, Q80RoundTripAndLinearMatchDequantizedReference) {
  const Tensor weight = MakeWeight();
  const auto blocks = QuantizeToQ80(weight);
  const Tensor dequantized = DequantizeFromQ80(blocks, weight.shape);
  Tensor input({32}, 1.0f);

  ExpectNear(dequantized, weight, 0.02f);
  ExpectNear(LinearQ80(input, blocks, weight.shape), Linear(input, dequantized),
             1e-5f);
}

// 测试目的：验证 Q4_0 往返误差受控，并覆盖批大小为 1 的二维输入路径。
TEST(QuantTest, Q40RoundTripAndLinearMatchDequantizedReference) {
  const Tensor weight = MakeWeight();
  const auto blocks = QuantizeToQ40(weight);
  const Tensor dequantized = DequantizeFromQ40(blocks, weight.shape);
  Tensor input({1, 32}, 1.0f);

  ExpectNear(dequantized, weight, 0.3f);
  ExpectNear(LinearQ40(input, blocks, weight.shape), Linear(input, dequantized),
             1e-5f);
}

// 测试目的：验证 Q4_1 点积同时应用块级缩放因子和最小值偏移。
TEST(QuantTest, Q41LinearUsesScaleAndMinimum) {
  BlockQ41 block{};
  block.d = 0x3c00;
  block.m = 0x3800;
  for (uint8_t& packed : block.qs) {
    packed = 0x22;
  }
  Tensor input({32}, 1.0f);

  const Tensor output = LinearQ41(input, {block}, {1, 32});
  ASSERT_EQ(output.shape, std::vector<int>({1}));
  EXPECT_FLOAT_EQ(output.data[0], 80.0f);
}

// 测试目的：验证线性层和反量化会拒绝非法权重形状及不匹配的块数量。
TEST(QuantTest, RejectsInvalidShapesAndBlockCounts) {
  Tensor input({32}, 1.0f);
  EXPECT_THROW(LinearQ80(input, {}, {1, 32}), std::runtime_error);
  EXPECT_THROW(LinearQ40(input, {}, {32}), std::runtime_error);
  EXPECT_THROW(DequantizeFromQ80({}, {1, 32}), std::runtime_error);
}

}  // namespace
