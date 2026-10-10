#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_

#include "mini_llama/cuda_kv_cache.h"
#include "mini_llama/cuda_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 仅报告 CUDA 编译状态，不保证存在可用设备
bool CudaAttentionBuilt();

// 主机调试接口：上传 q，复用设备注意力并下载 [n_heads, head_dim] 输出
// q 必须为 [n_heads, head_dim]；缓存与 device_id 必须匹配
Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache, int layer,
                           int pos, int n_heads, int n_kv_heads, int head_dim,
                           int device_id = 0);

// 设备 Decode 因果 GQA：仅访问指定层的 [0, pos] 历史，包含当前 token
// q 可为 [n_heads, head_dim] 或 [n_heads * head_dim]；返回扁平设备向量
// 查询头数必须能被 KV 头数整除，输入与缓存均位于 device_id
// 本接口不写缓存、不回读历史、不更新模型统计；调用方先写入当前 K/V
CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache, int layer,
                                          int pos, int n_heads, int n_kv_heads,
                                          int head_dim, int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_ATTENTION_H_
