#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_OPS_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_OPS_H_

#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 仅报告编译状态，不查询 GPU；CPU 构建下所有计算入口抛出未构建异常
bool CudaOpsBuilt();

// 主机 Tensor 接口上传输入/权重并下载结果；RMSNorm 和 Softmax 要求非空一维输入
// eps 必须有限且非负。逐元素算子要求输入 shape 相同
Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps,
                   int device_id = 0);

Tensor CudaSilu(const Tensor& x, int device_id = 0);

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id = 0);

Tensor CudaSoftmax(const Tensor& x, int device_id = 0);

// 借用 [vocab_size, dim] F32 设备权重；返回设备向量，不上传权重或下载结果
// 裸指针所属设备和生命周期由调用方保证，接口不取得权重所有权
CudaTensor CudaEmbeddingLookupDeviceWeight(const void* embedding_data,
                                           const std::vector<int>& embedding_shape,
                                           int token_id, int device_id = 0);

// 激活位于设备，weight 仍从主机上传；返回设备结果
CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id = 0);

// 输入和借用的 F32 权重均位于设备，接口内没有主机上传/下载
CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape, float eps,
                                   int device_id = 0);

// 以下设备逐元素接口不上传输入或下载输出；输入设备须与 device_id 一致
CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id = 0);

CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a, const CudaTensor& b,
                                         int device_id = 0);

CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a, const CudaTensor& b,
                                         int device_id = 0);

CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a, const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id = 0);

// 主机 RoPE 上传/下载 Q/K 并原地更新；形状分别为 [n_heads, head_dim] 和
// [n_kv_heads, head_dim]，head_dim 为正偶数，pos 非负，theta 有限且为正
void CudaRope(Tensor& q, Tensor& k, int pos, float theta,
              RopeType rope_type = RopeType::kNormal, int device_id = 0);

// 设备 RoPE 原地更新；Q/K 可为扁平或分头 shape，元素数须匹配所给头配置
void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads,
                         int head_dim, int pos, float theta,
                         RopeType rope_type = RopeType::kNormal, int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_OPS_H_
