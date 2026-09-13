#include "mini_llama/ops.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

Tensor Matmul(const Tensor& a, const Tensor& b) {
  return MatmulDispatch(a, b, DefaultMatmulMode());
}

Tensor Linear(const Tensor& x, const Tensor& weight) {
  return LinearDispatch(x, weight, DefaultMatmulMode());
}

Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
  // x: [dim], weight: [dim]
  if (x.num_dims() != 1 || weight.num_dims() != 1) {
    throw std::runtime_error("RmsNorm: expected 1D tensors");
  }

  int dim = x.shape[0];
  if (weight.shape[0] != dim) {
    throw std::runtime_error("RmsNorm: dimension mismatch");
  }

  float ss = 0.0f;
  for (int i = 0; i < dim; ++i) {
    ss += x.data[i] * x.data[i];
  }
  ss /= static_cast<float>(dim);            // 求平均
  float scale = 1.0 / std::sqrt(ss + eps);  // 均方根

  Tensor y({dim}, 0.0f);
  // y = x/rms * weight
  for (int i = 0; i < dim; ++i) {
    y.data[i] = x.data[i] * scale * weight.data[i];
  }

  return y;
}

Tensor Softmax(const Tensor& x) {
  if (x.num_dims() != 1) {
    throw std::runtime_error("Softmax: expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error("Softmax: cannot compute Softmax of empty tensor");
  }

  // 找寻最大值
  int n = x.shape[0];
  float max_value = x.data[0];
  for (int i = 1; i < n; ++i) {
    if (x.data[i] > max_value) {
      max_value = x.data[i];
    }
  }

  // softmax 概率计算
  float sum = 0.0f;
  Tensor y({n}, 0.0f);
  for (int i = 0; i < n; ++i) {
    y.data[i] = std::exp(x.data[i] - max_value);
    sum += y.data[i];
  }
  for (int i = 0; i < n; ++i) {
    y.data[i] /= sum;
  }

  return y;
}

Tensor Silu(const Tensor& x) {
  Tensor y(x.shape, 0.0f);
  for (size_t i = 0; i < x.data.size(); ++i) {
    float val = x.data[i];
    y.data[i] = val / (1.0f + std::exp(-val));
  }

  return y;
}

Tensor ElementwiseMul(const Tensor& a, const Tensor& b) {
  if (a.shape != b.shape) {
    throw std::runtime_error("ElementwiseMul: shape mismatch " +
                             a.ShapeStringShort() + " vs " + b.ShapeStringShort());
  }

  Tensor y(a.shape, 0.0f);
  for (size_t i = 0; i < a.data.size(); ++i) {
    y.data[i] = a.data[i] * b.data[i];
  }

  return y;
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
  if (gate.shape != up.shape) {
    throw std::runtime_error("SwiGlu: shape mismatch " + gate.ShapeStringShort() +
                             " vs " + up.ShapeStringShort());
  }

  Tensor y(gate.shape, 0.0f);
  for (size_t i = 0; i < gate.data.size(); ++i) {
    float val = gate.data[i];
    float silu = val / (1.0f + std::exp(-val));
    y.data[i] = silu * up.data[i];
  }

  return y;
}

int ArgMax(const Tensor& x) {
  if (x.size() == 0) {
    throw std::runtime_error("ArgMax: cannot compute ArgMax of empty tensor");
  }
  int best = 0;
  float best_val = x.data[0];
  for (size_t i = 1; i < x.data.size(); ++i) {
    if (x.data[i] > best_val) {
      best_val = x.data[i];
      best = static_cast<int>(i);
    }
  }
  return best;
}

}  // namespace mini_llama
