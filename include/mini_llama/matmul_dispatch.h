#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class MatmulMode {
  kNaive,         // 朴素三层循环，正确性基线
  kThreaded,      // 多线程，Linear 按输出通道并行
  kSimd,          // SIMD 点积，仅用于 Linear
  kThreadedSimd,  // 多线程输出通道 + SIMD 点积
};

// 主推理路径默认使用 kThreadedSimd；通用 Matmul 回退到多线程实现
MatmulMode DefaultMatmulMode();

// F32 矩阵乘法：[m,k] × [k,n] -> [m,n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// F32 Linear: x @ weight^T，即 x 乘以 weight 的转置
// 支持 x 的形状为 [in_features] 或 [1, in_features]
// weight 的形状为 [out_features, in_features]
// 结果张量的秩与输入张量的秩一致
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
