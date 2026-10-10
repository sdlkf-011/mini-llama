#include "mini_llama/cuda_quant.h"

#include <stdexcept>
#include <vector>

// 构建状态在此定义；CPU 构建提供量化 Linear Stub，GPU 实现位于 cuda_quant.cu
namespace mini_llama {

namespace {

// 三种量化格式共用未构建异常，不把缺失的 GPU 能力静默回退为 CPU 计算
std::runtime_error CudaQuantNotBuiltError() {
  return std::runtime_error(
      "CUDA quant kernels were not built. Reconfigure with "
      "-DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.");
}

}  // namespace

// 此结果只表示 Kernel 已加入构建，不查询 GPU 或校验权重格式
bool CudaQuantBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

#ifndef MINI_LLAMA_USE_CUDA

// Stub 不读取主机权重、设备块或 bias，直接报告未构建；(void) 用于消除参数告警
// Q8_0 的主机、常驻设备权重、设备输入三个入口使用同一异常约定
Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  (void)x;
  (void)weight;
  (void)weight_shape;
  (void)bias;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  (void)x;
  (void)q8_0_device;
  (void)q8_0_block_count;
  (void)weight_shape;
  (void)bias;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x, const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  (void)x;
  (void)q8_0_device;
  (void)q8_0_block_count;
  (void)weight_shape;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

// Q4_0 Stub 只保留公共接口，不执行反量化或分配临时 F32 权重
Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  (void)x;
  (void)weight;
  (void)weight_shape;
  (void)bias;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  (void)x;
  (void)q4_0_device;
  (void)q4_0_block_count;
  (void)weight_shape;
  (void)bias;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x, const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  (void)x;
  (void)q4_0_device;
  (void)q4_0_block_count;
  (void)weight_shape;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

// Q4_1 含块级偏移的计算同样要求 CUDA 构建，三种输入位置都明确拒绝
Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  (void)x;
  (void)weight;
  (void)weight_shape;
  (void)bias;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  (void)x;
  (void)q4_1_device;
  (void)q4_1_block_count;
  (void)weight_shape;
  (void)bias;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x, const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  (void)x;
  (void)q4_1_device;
  (void)q4_1_block_count;
  (void)weight_shape;
  (void)device_id;
  throw CudaQuantNotBuiltError();
}

#endif

}  // namespace mini_llama
