#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_OPS_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_OPS_H_

#include "mini_llama/model.h"
#include "tensor.h"

namespace mini_llama {

// 矩阵乘法：c = a * b
// a: [m, k]，b: [k, n]，c: [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b);

// 线性投影：y = x * weight^T
// x: [in_features] 或 [1, in_features]
// weight: [out_features, in_features]
// 结果保持 x 的秩：[out_features] 或 [1, out_features]
Tensor Linear(const Tensor& x, const Tensor& weight);

// RMSNorm：y = x / sqrt(mean(x^2) + eps) * weight
// x: [dim]，weight: [dim]，result: [dim]
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);

// 对一维张量执行 Softmax
// x: [n]，result: [n]
Tensor Softmax(const Tensor& x);

// SiLU 激活：Silu(x) = x * sigmoid(x)
Tensor Silu(const Tensor& x);

// 逐元素乘法
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);

// SwiGLU：SwiGlu(gate, up) = Silu(gate) * up
Tensor SwiGlu(const Tensor& gate, const Tensor& up);

// RoPE（旋转位置编码）应用于 Q 和 k
// q: [n_heads, head_dim]，k: [n_kv_heads, head_dim]
// pos: 当前 token 的位置
void Rope(Tensor& q, Tensor& k, int pos, float theta,
          RopeType rope_type = RopeType::kNormal);

// Argmax：返回最大值的 FlatIndex
int ArgMax(const Tensor& x);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_OPS_H_
