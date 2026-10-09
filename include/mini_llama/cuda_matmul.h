#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

bool CudaMatmulBuilt();

// F32 矩阵乘法，在 CUDA 上使用 cuBLAS
// a: [m, k]，b: [k, n]，result: [m, n]
Tensor CudaMatmul(const Tensor& a, const Tensor& b, int device_id = 0);

// F32 线性投影，在 CUDA 上使用 cuBLAS
// x: [in_features] 或 [batch, in_features]
// weight: [out_features, in_features]
// bias: 可选 [out_features]，在 cuBLAS 矩阵乘法之后应用
// result: 一维输入时为 [out_features]，二维输入时为 [batch, out_features]
Tensor CudaLinear(const Tensor& x, const Tensor& weight,
                  const Tensor* bias = nullptr, int device_id = 0);

// F32 线性投影，使用已存在于 CUDA 设备内存中的权重矩阵
// 输入激活从主机复制到设备，结果从设备复制到主机
// 调用方保证权重位于 device_id 且生命周期覆盖计算，接口不取得权重所有权
Tensor CudaLinearDeviceWeight(const Tensor& x, const void* w_device,
                              const std::vector<int>& w_shape,
                              const Tensor* bias = nullptr, int device_id = 0);

// F32 线性投影，输入、权重和输出均在 GPU；无输入上传或结果下载，不支持 bias
// 检查 x.device_id()；权重的设备与生命周期由调用方保证
CudaTensor CudaLinearDeviceInput(const CudaTensor& x, const void* w_device,
                                 const std::vector<int>& w_shape, int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_MATMUL_H_
