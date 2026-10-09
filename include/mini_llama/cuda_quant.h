#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_QUANT_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_QUANT_H_

#include <vector>

#include "mini_llama/cuda_tensor.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 仅报告编译状态；CPU 构建下所有 Linear 入口抛出未构建异常
bool CudaQuantBuilt();

// 三种格式均要求 weight_shape=[out, in]，x=[in] 或 [batch, in]
// 每个输出行独立按 32 个元素分块，块数为 out * ceil(in / 32)
// 主机权重接口上传 X/W 并下载 Y，可选 [out] bias 在 CPU 下载结果上广播相加
// 输出保持输入的秩，为 [out] 或 [batch, out]，激活与输出均使用 F32
// Q8_0 权重由 FP16 缩放因子和有符号 int8 值组成，点积时逐块解码
Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

// DeviceWeight 借用设备块，只上传 X、下载 Y；调用方保证权重设备及生命周期
// 接口不取得权重所有权，不生成完整的 F32 权重副本。Q4_0/Q4_1 同此约定
Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr, int device_id = 0);

// DeviceInput 的 X/W/Y 均在设备，无 H2D/D2H 和 bias 参数；Q4_0/Q4_1 同此约定
CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x, const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

// Q4_0：每块使用 FP16 缩放因子和 32 个打包的 4 位值，解码为 d * (q - 8)
// 此主机接口、下面两个设备接口的数据位置与 Q8_0 对应接口相同
Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

// 借用常驻 Q4_0 设备块；仅上传激活和下载结果，block_count 不包含 batch 维
Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr, int device_id = 0);

// 输入、权重和返回结果均位于 device_id；不下载输出，不支持 bias 参数
CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x, const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

// Q4_1：在 Q4 打包值上使用 FP16 缩放因子 d 与偏移 m，解码为 d * q + m
// 接受已有 Q4_1 块，不在此接口执行权重量化
Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape,
                     const Tensor* bias = nullptr, int device_id = 0);

// 借用常驻 Q4_1 设备块；可选 bias 在结果下载后由 CPU 广播相加
Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias = nullptr, int device_id = 0);

// 完全使用设备数据完成单次 Linear；权重的设备及生命周期仍由调用方保证
CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x, const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id = 0);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_CUDA_QUANT_H_
