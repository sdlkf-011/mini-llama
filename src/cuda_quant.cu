#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_quant.h"
#include "mini_llama/cuda_runtime.h"

namespace mini_llama {

namespace {

constexpr int kBlockSize = 128;

// 展平 batch/output 任务，并限制一维网格大小；Kernel 跨步覆盖剩余任务
dim3 LinearGrid(int rows, int out_features) {
  const size_t tasks = static_cast<size_t>(rows) * out_features;
  const size_t blocks = (tasks - 1) / kBlockSize + 1;
  return dim3(static_cast<unsigned int>(std::min(blocks, size_t{65535})));
}

// 将 CUDA 状态转为带 API 名称的异常，保留 Runtime 提供的具体失败原因
void CheckCudaQuant(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA quant error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

// 读取启动后的错误状态而不显式等待；执行期错误还可能在主机结果下载时暴露
void CheckLastKernel(const char* name) { CheckCudaQuant(cudaGetLastError(), name); }

// Tensor 的 shape/data 可独立修改；检查正维度、元素数溢出及实际存储长度
void ValidateHostStorage(const Tensor& tensor, const char* caller) {
  size_t elements = 1;
  if (tensor.shape.empty()) {
    throw std::runtime_error(std::string(caller) + ": empty shape");
  }
  for (int dimension : tensor.shape) {
    if (dimension <= 0 || elements > std::numeric_limits<size_t>::max() /
                                         static_cast<size_t>(dimension)) {
      throw std::runtime_error(std::string(caller) + ": invalid shape");
    }
    elements *= static_cast<size_t>(dimension);
  }
  if (elements != tensor.data.size()) {
    throw std::runtime_error(std::string(caller) + ": storage size mismatch");
  }
}

// bias 为 nullptr 表示省略；非空时必须是完整的 [out_features] 主机向量
void ValidateBias(const Tensor* bias, int cols) {
  if (!bias) return;
  ValidateHostStorage(*bias, "CUDA quant bias");
  if (bias->shape != std::vector<int>{cols}) {
    throw std::runtime_error("CUDA quant bias: expected [out_features]");
  }
}

// 块中的 d/m 保存 FP16 原始位模式，先转为 F32 再参与点积累加
__device__ float HalfBitsToFloat(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(raw);
}

// 每个线程沿一维网格跨步处理展平的 batch/output 任务，计算 F32 点积
// 权重按输出行独立分块；尾块只读取真实 in_features，不读取填充元素
__global__ void Q80LinearKernel(const float* x, const BlockQ80* weight, float* y,
                                int rows, int in_features, int out_features,
                                int blocks_per_row) {
  const size_t tasks = static_cast<size_t>(rows) * out_features;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < tasks; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int out = static_cast<int>(index % out_features);
    const size_t row = index / out_features;

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ80* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
      const BlockQ80& block = w_row[block_index];
      float d = HalfBitsToFloat(block.d);
      int base_k = block_index * kQ80BlockSize;
      int k_end = base_k + kQ80BlockSize < in_features ? base_k + kQ80BlockSize
                                                       : in_features;
      for (int k = base_k; k < k_end; ++k) {
        sum += d * static_cast<float>(block.qs[k - base_k]) * x_row[k];
      }
    }
    y[index] = sum;
  }
}

// 每字节低 4 位对应前 16 个元素，高 4 位对应后 16 个；Q4_0 解码为 d*(q-8)
__device__ float Q40Value(const BlockQ40& block, int flat_index) {
  int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
                          : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return HalfBitsToFloat(block.d) * static_cast<float>(q - 8);
}

// Q4_1 使用相同的半块打包布局，但解码为 d*q+m，不减去 Q4_0 的零点 8
__device__ float Q41Value(const BlockQ41& block, int flat_index) {
  int q = flat_index < 16 ? static_cast<int>(block.qs[flat_index] & 0x0F)
                          : static_cast<int>(block.qs[flat_index - 16] >> 4);
  return HalfBitsToFloat(block.d) * static_cast<float>(q) + HalfBitsToFloat(block.m);
}

// Q4_0 点积在使用每个元素时解码 4 位权重，不展开整张 F32 权重矩阵
__global__ void Q40LinearKernel(const float* x, const BlockQ40* weight, float* y,
                                int rows, int in_features, int out_features,
                                int blocks_per_row) {
  const size_t tasks = static_cast<size_t>(rows) * out_features;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < tasks; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int out = static_cast<int>(index % out_features);
    const size_t row = index / out_features;

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ40* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
      const BlockQ40& block = w_row[block_index];
      int base_k = block_index * kQ40BlockSize;
      int k_end = base_k + kQ40BlockSize < in_features ? base_k + kQ40BlockSize
                                                       : in_features;
      for (int k = base_k; k < k_end; ++k) {
        sum += Q40Value(block, k - base_k) * x_row[k];
      }
    }
    y[index] = sum;
  }
}

// Q4_1 沿相同的逐行块布局计算，并把每块的缩放与偏移都用于点积
__global__ void Q41LinearKernel(const float* x, const BlockQ41* weight, float* y,
                                int rows, int in_features, int out_features,
                                int blocks_per_row) {
  const size_t tasks = static_cast<size_t>(rows) * out_features;
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < tasks; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int out = static_cast<int>(index % out_features);
    const size_t row = index / out_features;

    const float* x_row = x + static_cast<size_t>(row) * in_features;
    const BlockQ41* w_row = weight + static_cast<size_t>(out) * blocks_per_row;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
      const BlockQ41& block = w_row[block_index];
      int base_k = block_index * kQ41BlockSize;
      int k_end = base_k + kQ41BlockSize < in_features ? base_k + kQ41BlockSize
                                                       : in_features;
      for (int k = base_k; k < k_end; ++k) {
        sum += Q41Value(block, k - base_k) * x_row[k];
      }
    }
    y[index] = sum;
  }
}

// 结果已下载到主机后，按 batch 行广播相加；计算前的入口已校验 bias 存储
void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols,
                    const char* caller) {
  if (bias.num_dims() != 1 || bias.shape[0] != cols) {
    throw std::runtime_error(std::string(caller) +
                             ": expected bias shape [out_features]");
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
    }
  }
}

// 一维输入视为单行；二维输入沿最后一维点积，输出保留输入的秩
int Q80InFeaturesFromShape(const std::vector<int>& x_shape,
                           const std::string& x_shape_str) {
  if (x_shape.size() == 1) {
    return x_shape[0];
  }
  if (x_shape.size() == 2) {
    return x_shape[1];
  }
  throw std::runtime_error(
      "CudaQ80Linear: expected x shape [in_features] or [batch, in_features], "
      "got " +
      x_shape_str);
}

// 验证矩阵轴和逐输出行的块数；块数描述权重，不随输入 batch 数增加
void ValidateQ80LinearShape(const std::vector<int>& x_shape,
                            const std::string& x_shape_str,
                            const std::vector<int>& weight_shape,
                            size_t block_count) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error(
        "CudaQ80Linear: expected weight shape [out_features, in_features]");
  }
  int in_features = Q80InFeaturesFromShape(x_shape, x_shape_str);
  if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
    throw std::runtime_error("CudaQ80Linear: empty tensors are not supported");
  }
  if (weight_shape[1] != in_features) {
    throw std::runtime_error("CudaQ80Linear: dimension mismatch x=" + x_shape_str +
                             " weight=[" + std::to_string(weight_shape[0]) + ", " +
                             std::to_string(weight_shape[1]) + "]");
  }
  int blocks_per_row = (in_features - 1) / kQ80BlockSize + 1;
  size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
  if (block_count != expected_blocks) {
    throw std::runtime_error("CudaQ80Linear: block count mismatch: expected " +
                             std::to_string(expected_blocks) + ", got " +
                             std::to_string(block_count));
  }
}

void ValidateQ80LinearInputs(const Tensor& x, const std::vector<int>& weight_shape,
                             size_t block_count) {
  ValidateHostStorage(x, "CudaQ80Linear input");
  ValidateQ80LinearShape(x.shape, x.ShapeStringShort(), weight_shape, block_count);
}

// Q4_0/Q4_1 共用输入秩规则，但异常保留各自的入口名称
int QuantInFeaturesFromShape(const std::vector<int>& x_shape,
                             const std::string& x_shape_str, const char* caller) {
  if (x_shape.size() == 1) {
    return x_shape[0];
  }
  if (x_shape.size() == 2) {
    return x_shape[1];
  }
  throw std::runtime_error(
      std::string(caller) +
      ": expected x shape [in_features] or [batch, in_features], got " +
      x_shape_str);
}

// 校验 Q4 逻辑形状与显式块数，不探测借用设备指针的实际分配容量
void ValidateQuantLinearShape(const std::vector<int>& x_shape,
                              const std::string& x_shape_str,
                              const std::vector<int>& weight_shape,
                              size_t block_count, int block_size,
                              const char* caller) {
  if (weight_shape.size() != 2) {
    throw std::runtime_error(std::string(caller) +
                             ": expected weight shape [out_features, in_features]");
  }
  int in_features = QuantInFeaturesFromShape(x_shape, x_shape_str, caller);
  if (in_features <= 0 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
    throw std::runtime_error(std::string(caller) +
                             ": empty tensors are not supported");
  }
  if (weight_shape[1] != in_features) {
    throw std::runtime_error(std::string(caller) +
                             ": dimension mismatch x=" + x_shape_str + " weight=[" +
                             std::to_string(weight_shape[0]) + ", " +
                             std::to_string(weight_shape[1]) + "]");
  }
  int blocks_per_row = (in_features - 1) / block_size + 1;
  size_t expected_blocks = static_cast<size_t>(weight_shape[0]) * blocks_per_row;
  if (block_count != expected_blocks) {
    throw std::runtime_error(
        std::string(caller) + ": block count mismatch: expected " +
        std::to_string(expected_blocks) + ", got " + std::to_string(block_count));
  }
}

// 常驻权重路径只上传 X、下载 Y；校验主机存储和 bias 后再启动计算
Tensor CudaQ80LinearWithDevicePointer(const Tensor& x, const void* q8_0_device,
                                      size_t q8_0_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
  if (q8_0_device == nullptr) {
    throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
  }
  ValidateQ80LinearInputs(x, weight_shape, q8_0_block_count);

  ValidateHostStorage(x, "CUDA quant input");
  ValidateBias(bias, weight_shape[0]);
  CudaSetDevice(device_id);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features - 1) / kQ80BlockSize + 1;

  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  dim3 block(kBlockSize);
  dim3 grid = LinearGrid(rows, out_features);
  Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()),
                                   static_cast<const BlockQ80*>(q8_0_device),
                                   static_cast<float*>(y_dev.data()), rows,
                                   in_features, out_features, blocks_per_row);
  CheckLastKernel("Q80LinearKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    AddBiasInPlace(y, *bias, rows, out_features, "CudaQ80Linear");
  }
  return y;
}

// X/W/Y 都在设备；输入设备及块元数据必须匹配，不执行 H2D/D2H
CudaTensor CudaQ80LinearWithDeviceInput(const CudaTensor& x, const void* q8_0_device,
                                        size_t q8_0_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
  if (q8_0_device == nullptr) {
    throw std::runtime_error("CudaQ80Linear: device weight pointer is null");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaQ80Linear: input tensor is on a different CUDA device");
  }
  ValidateQ80LinearShape(x.shape(), x.ShapeStringShort(), weight_shape,
                         q8_0_block_count);

  CudaSetDevice(device_id);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape()[0];
  int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features - 1) / kQ80BlockSize + 1;

  CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

  dim3 block(kBlockSize);
  dim3 grid = LinearGrid(rows, out_features);
  Q80LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()),
                                   static_cast<const BlockQ80*>(q8_0_device),
                                   static_cast<float*>(y.data()), rows, in_features,
                                   out_features, blocks_per_row);
  CheckLastKernel("Q80LinearKernel");

  return y;
}

// Q4_0 借用设备块；临时激活按输入秩分配，bias 在主机结果上应用
Tensor CudaQ40LinearWithDevicePointer(const Tensor& x, const void* q4_0_device,
                                      size_t q4_0_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
  static constexpr const char* kCaller = "CudaQ40Linear";
  if (q4_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           q4_0_block_count, kQ40BlockSize, kCaller);

  ValidateHostStorage(x, "CUDA quant input");
  ValidateBias(bias, weight_shape[0]);
  CudaSetDevice(device_id);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features - 1) / kQ40BlockSize + 1;

  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  dim3 block(kBlockSize);
  dim3 grid = LinearGrid(rows, out_features);
  Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()),
                                   static_cast<const BlockQ40*>(q4_0_device),
                                   static_cast<float*>(y_dev.data()), rows,
                                   in_features, out_features, blocks_per_row);
  CheckLastKernel("Q40LinearKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    AddBiasInPlace(y, *bias, rows, out_features, kCaller);
  }
  return y;
}

// Q4_0 全设备 Linear 与主机路径复用同一 Kernel，返回输出显存所有权
CudaTensor CudaQ40LinearWithDeviceInput(const CudaTensor& x, const void* q4_0_device,
                                        size_t q4_0_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
  static constexpr const char* kCaller = "CudaQ40Linear";
  if (q4_0_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(std::string(kCaller) +
                             ": input tensor is on a different CUDA device");
  }
  ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), weight_shape,
                           q4_0_block_count, kQ40BlockSize, kCaller);

  CudaSetDevice(device_id);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape()[0];
  int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features - 1) / kQ40BlockSize + 1;

  CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

  dim3 block(kBlockSize);
  dim3 grid = LinearGrid(rows, out_features);
  Q40LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()),
                                   static_cast<const BlockQ40*>(q4_0_device),
                                   static_cast<float*>(y.data()), rows, in_features,
                                   out_features, blocks_per_row);
  CheckLastKernel("Q40LinearKernel");

  return y;
}

// Q4_1 主机激活路径借用常驻设备权重，保持块格式并按需应用主机 bias
Tensor CudaQ41LinearWithDevicePointer(const Tensor& x, const void* q4_1_device,
                                      size_t q4_1_block_count,
                                      const std::vector<int>& weight_shape,
                                      const Tensor* bias, int device_id) {
  static constexpr const char* kCaller = "CudaQ41Linear";
  if (q4_1_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           q4_1_block_count, kQ41BlockSize, kCaller);

  ValidateHostStorage(x, "CUDA quant input");
  ValidateBias(bias, weight_shape[0]);
  CudaSetDevice(device_id);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features - 1) / kQ41BlockSize + 1;

  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  dim3 block(kBlockSize);
  dim3 grid = LinearGrid(rows, out_features);
  Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x_dev.data()),
                                   static_cast<const BlockQ41*>(q4_1_device),
                                   static_cast<float*>(y_dev.data()), rows,
                                   in_features, out_features, blocks_per_row);
  CheckLastKernel("Q41LinearKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    AddBiasInPlace(y, *bias, rows, out_features, kCaller);
  }
  return y;
}

// Q4_1 设备输入路径只分配输出，不重新上传已有权重或下载激活
CudaTensor CudaQ41LinearWithDeviceInput(const CudaTensor& x, const void* q4_1_device,
                                        size_t q4_1_block_count,
                                        const std::vector<int>& weight_shape,
                                        int device_id) {
  static constexpr const char* kCaller = "CudaQ41Linear";
  if (q4_1_device == nullptr) {
    throw std::runtime_error(std::string(kCaller) +
                             ": device weight pointer is null");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(std::string(kCaller) +
                             ": input tensor is on a different CUDA device");
  }
  ValidateQuantLinearShape(x.shape(), x.ShapeStringShort(), weight_shape,
                           q4_1_block_count, kQ41BlockSize, kCaller);

  CudaSetDevice(device_id);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape()[0];
  int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
  int out_features = weight_shape[0];
  int blocks_per_row = (in_features - 1) / kQ41BlockSize + 1;

  CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

  dim3 block(kBlockSize);
  dim3 grid = LinearGrid(rows, out_features);
  Q41LinearKernel<<<grid, block>>>(static_cast<const float*>(x.data()),
                                   static_cast<const BlockQ41*>(q4_1_device),
                                   static_cast<float*>(y.data()), rows, in_features,
                                   out_features, blocks_per_row);
  CheckLastKernel("Q41LinearKernel");

  return y;
}

}  // namespace

// 主机权重入口每次临时上传 W，再委托设备权重路径；常驻调用应使用后两个入口
Tensor CudaQ80Linear(const Tensor& x, const std::vector<BlockQ80>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  ValidateQ80LinearInputs(x, weight_shape, weight.size());
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ80), device_id);
  w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ80));
  return CudaQ80LinearWithDevicePointer(x, w_dev.data(), weight.size(), weight_shape,
                                        bias, device_id);
}

// 只转发借用的块指针和元数据，调用方负责权重所属设备及生命周期
Tensor CudaQ80LinearDeviceWeight(const Tensor& x, const void* q8_0_device,
                                 size_t q8_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  return CudaQ80LinearWithDevicePointer(x, q8_0_device, q8_0_block_count,
                                        weight_shape, bias, device_id);
}

// 设备输入入口保留设备结果，是否下载由调用方决定
CudaTensor CudaQ80LinearDeviceInput(const CudaTensor& x, const void* q8_0_device,
                                    size_t q8_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  return CudaQ80LinearWithDeviceInput(x, q8_0_device, q8_0_block_count, weight_shape,
                                      device_id);
}

// 主机 Q4_0 入口先校验块布局，再临时上传原始块，不生成完整 F32 权重副本
Tensor CudaQ40Linear(const Tensor& x, const std::vector<BlockQ40>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           weight.size(), kQ40BlockSize, "CudaQ40Linear");
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ40), device_id);
  w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ40));
  return CudaQ40LinearWithDevicePointer(x, w_dev.data(), weight.size(), weight_shape,
                                        bias, device_id);
}

// 复用 Q4_0 常驻块路径，只在激活与主机结果之间发生传输
Tensor CudaQ40LinearDeviceWeight(const Tensor& x, const void* q4_0_device,
                                 size_t q4_0_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  return CudaQ40LinearWithDevicePointer(x, q4_0_device, q4_0_block_count,
                                        weight_shape, bias, device_id);
}

// 复用 Q4_0 设备输入路径，接口不取得借用权重的所有权
CudaTensor CudaQ40LinearDeviceInput(const CudaTensor& x, const void* q4_0_device,
                                    size_t q4_0_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  return CudaQ40LinearWithDeviceInput(x, q4_0_device, q4_0_block_count, weight_shape,
                                      device_id);
}

// 主机 Q4_1 入口上传已有的 d/m/qs 块，不在此接口重新量化权重
Tensor CudaQ41Linear(const Tensor& x, const std::vector<BlockQ41>& weight,
                     const std::vector<int>& weight_shape, const Tensor* bias,
                     int device_id) {
  ValidateQuantLinearShape(x.shape, x.ShapeStringShort(), weight_shape,
                           weight.size(), kQ41BlockSize, "CudaQ41Linear");
  CudaDeviceBuffer w_dev(weight.size() * sizeof(BlockQ41), device_id);
  w_dev.Upload(weight.data(), weight.size() * sizeof(BlockQ41));
  return CudaQ41LinearWithDevicePointer(x, w_dev.data(), weight.size(), weight_shape,
                                        bias, device_id);
}

// 转发包含偏移 m 的常驻 Q4_1 块，bias 仍是可选的主机 F32 数据
Tensor CudaQ41LinearDeviceWeight(const Tensor& x, const void* q4_1_device,
                                 size_t q4_1_block_count,
                                 const std::vector<int>& weight_shape,
                                 const Tensor* bias, int device_id) {
  return CudaQ41LinearWithDevicePointer(x, q4_1_device, q4_1_block_count,
                                        weight_shape, bias, device_id);
}

// 设备 Q4_1 入口与其他格式保持相同的设备、块数和输入秩约定
CudaTensor CudaQ41LinearDeviceInput(const CudaTensor& x, const void* q4_1_device,
                                    size_t q4_1_block_count,
                                    const std::vector<int>& weight_shape,
                                    int device_id) {
  return CudaQ41LinearWithDeviceInput(x, q4_1_device, q4_1_block_count, weight_shape,
                                      device_id);
}

}  // namespace mini_llama
