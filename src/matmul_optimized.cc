#include <stdexcept>
#include <vector>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// ----------------------------------------
// 朴素实现(三层循环)[为确保可读性和回退而保留]
// ----------------------------------------

// matmul:标准矩阵乘法
static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int inner_dim = a.shape[1];
  int n = b.shape[1];

  Tensor c({m, n}, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < inner_dim; ++k) {
        sum += a.data[i * inner_dim + k] * b.data[n * k + j];
      }
      c.data[i * n + j] = sum;
    }
  }
  return c;
}

// 线性层计算
static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];   // 输入特征量
  int out_features = weight.shape[0];  // 输出特征量
  bool is_1d = (x.num_dims() == 1);

  Tensor result(
      (is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features}),
      0.0f);
  for (int j = 0; j < out_features; ++j) {
    float sum = 0.0f;
    for (int k = 0; k < in_features; ++k) {
      sum += x.data[k] * weight.data[j * in_features + k];
    }
    result.data[j] = sum;
  }
  return result;
}

MatmulMode DefaultMatmulMode() { return MatmulMode::kNaive; }

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  (void)mode;
  if (a.num_dims() != 2 || b.num_dims() != 2) {
    throw std::runtime_error("MatmulDispatch: expected 2D tensors");
  }
  if (a.shape[1] != b.shape[0]) {
    throw std::runtime_error("MatmulDispatch: dimension mismatch");
  }

  return MatmulNaive(a, b);
}

Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
  (void)mode;
  // 校验 weight 必须是二维
  if (weight.num_dims() != 2) {
    throw std::runtime_error(
        "LinearDispatch: expected weight shape [out_features, in_features]");
  }

  // 根据输入 x 的形状 ，来判断 in_features
  int in_features;
  if (x.num_dims() == 1) {
    in_features = x.shape[0];
  } else if (x.num_dims() == 2 && x.shape[0] == 1) {
    in_features = x.shape[1];
  } else {
    throw std::runtime_error(
        "LinearDispatch: expected x shape [in_features] or [1, in_features]");
  }

  // 输入特征量与权重维度是否匹配
  if (weight.shape[1] != in_features) {
    throw std::runtime_error("LinearDispatch: dimension mismatch");
  }

  return LinearNaive(x, weight);
}

}  // namespace mini_llama
