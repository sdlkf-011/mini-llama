#ifndef INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
#define INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class MatmulMode {
  kNaive,         // 朴素三层循环，正确性基线
  kThreaded,      // 多线程实现
  kSimd,          // 单指令多数据实现
  kThreadedSimd,  // 多线程与 SIMD 结合
};

// 主推理路径使用的默认模式
MatmulMode DefaultMatmulMode();

// F32 矩阵乘法：[m,k] × [k,n] -> [m,n]
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatmulMode mode);

// F32 Linear: x @ weight^T，即 x 乘以 weight 的转置
// 支持 x 的形状为 [in_features] 或 [1, in_features]
// weight 的形状为 [out_features, in_features]
// 结果张量的秩与输入张量的秩一致
Tensor LinearDispatch(const Tensor& x, const Tensor& weight, MatmulMode mode);

}  // namespace mini_llama

#endif  // INCLUDE_MINI_LLAMA_MATMUL_DISPATCH_H_
