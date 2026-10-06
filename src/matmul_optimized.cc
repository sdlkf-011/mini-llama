#include <stdexcept>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

#include "mini_llama/matmul_dispatch.h"
#include "mini_llama/tensor.h"
#include "mini_llama/thread_pool.h"

namespace mini_llama {

// ----------------------------------------
// Naive implementations（当前正确性基线）
// ----------------------------------------

// 按行主序计算标准二维矩阵乘法 [m, k] x [k, n]
static Tensor MatmulNaive(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int inner_dim = a.shape[1];
  int n = b.shape[1];
  Tensor c({m, n}, 0.0f);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      // B 的同一列在行主序存储中不连续，因此按行跨度读取
      for (int k = 0; k < inner_dim; ++k) {
        sum += a.data[i * inner_dim + k] * b.data[k * n + j];
      }
      c.data[i * n + j] = sum;
    }
  }
  return c;
}

// 计算 F32 Linear 的 x @ weight^T，并保持输入张量的秩
static Tensor LinearNaive(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];   // 输入特征量
  int out_features = weight.shape[0];  // 输出特征量
  bool is_1d = (x.num_dims() == 1);

  Tensor result(
      is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features},
      0.0f);
  for (int j = 0; j < out_features; ++j) {
    float sum = 0.0f;
    // weight 的每一行对应一个输出特征，行内数据连续
    for (int k = 0; k < in_features; ++k) {
      sum += x.data[k] * weight.data[j * in_features + k];
    }
    result.data[j] = sum;
  }
  return result;
}

// 按行划分通用 Matmul，B 的列仍然按行跨度读取
static Tensor MatmulThreaded(const Tensor& a, const Tensor& b) {
  int m = a.shape[0];
  int inner_dim = a.shape[1];
  int n = b.shape[1];
  Tensor c({m, n}, 0.0f);
  ParallelFor(m, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      for (int j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < inner_dim; ++k) {
          sum += a.data[i * inner_dim + k] * b.data[k * n + j];
        }
        c.data[i * n + j] = sum;
      }
    }
  });
  return c;
}

// 多线程只划分输出通道，每个线程独立计算完整点积
static Tensor LinearThreaded(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  bool is_1d = (x.num_dims() == 1);

  Tensor result(
      is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features},
      0.0f);
  ParallelFor(out_features, [&](int begin, int end) {
    for (int j = begin; j < end; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < in_features; ++k) {
        sum += x.data[k] * weight.data[j * in_features + k];
      }
      result.data[j] = sum;
    }
  });
  return result;
}

// 连续 F32 向量点积，SIMD 主循环之后用标量处理尾部
static float DotSimd(const float* a, const float* b, int n) {
  float sum = 0.0f;
  int i = 0;
#if defined(MINI_LLAMA_USE_AVX2)
  __m256 accumulator = _mm256_setzero_ps();
  for (; i <= n - 8; i += 8) {
    accumulator =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), accumulator);
  }
  __m128 halves = _mm_add_ps(_mm256_castps256_ps128(accumulator),
                             _mm256_extractf128_ps(accumulator, 1));
  halves = _mm_hadd_ps(halves, halves);
  halves = _mm_hadd_ps(halves, halves);
  sum = _mm_cvtss_f32(halves);
#elif defined(MINI_LLAMA_USE_NEON)
  float32x4_t accumulator = vdupq_n_f32(0.0f);
  for (; i <= n - 4; i += 4) {
    accumulator = vmlaq_f32(accumulator, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float32x2_t halves =
      vadd_f32(vget_low_f32(accumulator), vget_high_f32(accumulator));
  halves = vpadd_f32(halves, halves);
  sum = vget_lane_f32(halves, 0);
#endif
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

static Tensor LinearSimd(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  bool is_1d = (x.num_dims() == 1);

  Tensor result(
      is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features},
      0.0f);
  for (int j = 0; j < out_features; ++j) {
    result.data[j] =
        DotSimd(x.data.data(), weight.data.data() + j * in_features, in_features);
  }
  return result;
}

static Tensor LinearThreadedSimd(const Tensor& x, const Tensor& weight) {
  int in_features = weight.shape[1];
  int out_features = weight.shape[0];
  bool is_1d = (x.num_dims() == 1);

  Tensor result(
      is_1d ? std::vector<int>{out_features} : std::vector<int>{1, out_features},
      0.0f);
  ParallelFor(out_features, [&](int begin, int end) {
    for (int j = begin; j < end; ++j) {
      result.data[j] =
          DotSimd(x.data.data(), weight.data.data() + j * in_features, in_features);
    }
  });
  return result;
}

// ----------------------------------------
// Dispatch
// ----------------------------------------

MatmulMode DefaultMatmulMode() { return MatmulMode::kThreadedSimd; }

// 校验输入矩阵，并将请求分发到已实现的矩阵乘后端。
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode) {
  if (a.num_dims() != 2 || b.num_dims() != 2) {
    throw std::runtime_error("MatmulDispatch: expected 2D tensors");
  }
  if (a.shape[1] != b.shape[0]) {
    throw std::runtime_error("MatmulDispatch: dimension mismatch");
  }
  switch (mode) {
    case MatmulMode::kNaive:
      return MatmulNaive(a, b);
    case MatmulMode::kThreaded:
    case MatmulMode::kSimd:
    case MatmulMode::kThreadedSimd:
      // B 的列不连续，SIMD 模式暂时复用多线程标量 Matmul
      return MatmulThreaded(a, b);
  }
  throw std::runtime_error("MatmulDispatch: unsupported matmul mode");
}

// 校验 F32 Linear 的 shape，并将请求分发到已实现的后端
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode) {
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
  if (weight.shape[1] != in_features) {
    throw std::runtime_error("LinearDispatch: dimension mismatch");
  }
  switch (mode) {
    case MatmulMode::kNaive:
      return LinearNaive(x, weight);
    case MatmulMode::kThreaded:
      return LinearThreaded(x, weight);
    case MatmulMode::kSimd:
      return LinearSimd(x, weight);
    case MatmulMode::kThreadedSimd:
      return LinearThreadedSimd(x, weight);
  }
  throw std::runtime_error("LinearDispatch: unsupported matmul mode");
}

}  // namespace mini_llama
