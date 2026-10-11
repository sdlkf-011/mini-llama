#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <limits>

#include "cuda_forward_test_utils.h"

namespace {
using namespace mini_llama;
using mini_llama::test::RequireCloseTensor;

// 在 CPU 上验证比较器的两条接受分支及小值归一化语义。
TEST(CudaComparisonTest, AbsoluteOrNormalizedRelativeTolerance) {
  Tensor expected({3});
  expected.data = {0.001f, -100.0f, 0.0f};
  Tensor actual = expected;
  actual.data = {0.002f, -100.1f, 0.001f};
  RequireCloseTensor(actual, expected, 2e-3f, 2e-3f, "logits");
  RequireCloseTensor(Tensor({1}, 0.01f), Tensor({1}), 0.02f, 0.0f, "absolute");
  RequireCloseTensor(Tensor({1}, 100.1f), Tensor({1}, 100.0f), 0.0f, 2e-3f,
                     "relative");
  EXPECT_NONFATAL_FAILURE(RequireCloseTensor(Tensor({1}, 0.003f), Tensor({1}), 2e-3f,
                                             2e-3f, "near zero"),
                          "abs_error=");
}

TEST(CudaComparisonTest, FailureIdentifiesTensorPositionValuesAndThresholds) {
  Tensor actual({2, 2}), expected({2, 2});
  actual.data[3] = 1;
  EXPECT_NONFATAL_FAILURE(
      RequireCloseTensor(actual, expected, 2e-3f, 2e-3f, "logits"),
      "logits index=3 position=[1,1] CPU expected=0 CUDA actual=1 "
      "abs_error=1 relative_error=1 abs_tol=0.002 rel_tol=0.002");
}

TEST(CudaComparisonTest, RejectsNonFiniteCpuAndCudaValuesAndWrongShape) {
  EXPECT_FATAL_FAILURE(
      RequireCloseTensor(Tensor({1}, std::numeric_limits<float>::infinity()),
                         Tensor({1}), 2e-3f, 2e-3f, "CUDA infinity"),
      "CUDA infinity");
  EXPECT_FATAL_FAILURE(
      RequireCloseTensor(Tensor({1}),
                         Tensor({1}, std::numeric_limits<float>::quiet_NaN()), 2e-3f,
                         2e-3f, "CPU NaN"),
      "CPU NaN");
  EXPECT_FATAL_FAILURE(RequireCloseTensor(Tensor({2}), Tensor({1, 2})), "tensor");
}
}  // namespace
