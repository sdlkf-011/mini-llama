#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_ops.h"
#include "mini_llama/cuda_runtime.h"

namespace mini_llama {

namespace {

constexpr int kBlockSize = 256;

// 保留失败 API 与 Runtime 原因，供调用方区分设备查询、分配和 Kernel 问题
void CheckCudaOps(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA ops error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

// 检查启动错误而不显式同步；执行期错误可能在后续 Runtime 操作或下载时报告
void CheckLastKernel(const char* name) { CheckCudaOps(cudaGetLastError(), name); }

// 各线程将 x[i]^2 原子累加到设备标量；调用方必须先将 sum 清零
__global__ void SumSquaresKernel(const float* x, float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(sum, x[i] * x[i]);
  }
}

// 平方和归约完成后逐元素应用 rsqrt(mean(x^2) + eps) 和 F32 缩放权重
__global__ void RmsNormKernel(const float* x, const float* weight, float* y,
                              const float* sum, int n, float eps) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float scale = rsqrtf((*sum / static_cast<float>(n)) + eps);
    y[i] = x[i] * scale * weight[i];
  }
}

// SiLU 按连续 F32 元素计算，尾部线程通过 i < n 避免越界
__global__ void SiluKernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    y[i] = v / (1.0f + expf(-v));
  }
}

// 加乘均按相同 shape 的连续元素计算，不执行广播或原地覆盖输入
__global__ void ElementwiseMulKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] * b[i];
  }
}

__global__ void ElementwiseAddKernel(const float* a, const float* b, float* y,
                                     int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a[i] + b[i];
  }
}

// 每个线程复制词表中 token_id 对应行的一个特征，输出仅包含该 token 的向量
__global__ void EmbeddingLookupKernel(const float* embedding, float* y, int token_id,
                                      int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    y[i] = embedding[token_id * dim + i];
  }
}

// 单个线程块跨步扫描全部输入，再用共享内存和同步屏障归约最大值
__global__ void SoftmaxMaxKernel(const float* x, float* max_out, int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  float local_max = -3.402823466e+38F;
  for (int i = tid; i < n; i += blockDim.x) {
    local_max = fmaxf(local_max, x[i]);
  }
  shared[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    *max_out = shared[0];
  }
}

// 减去全局最大值后计算指数，并在同一个线程块内归约分母以避免指数溢出
__global__ void SoftmaxExpSumKernel(const float* x, float* y, const float* max_value,
                                    float* sum_out, int n) {
  __shared__ float shared[kBlockSize];
  int tid = threadIdx.x;
  float local_sum = 0.0f;
  float max_v = *max_value;
  for (int i = tid; i < n; i += blockDim.x) {
    float e = expf(x[i] - max_v);
    y[i] = e;
    local_sum += e;
  }
  shared[tid] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *sum_out = shared[0];
  }
}

// 前两个 Kernel 写入指数与分母后，按元素完成归一化
__global__ void SoftmaxNormKernel(float* y, const float* sum, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] /= *sum;
  }
}

// Normal 布局在每个头内旋转相邻维度 (2*i, 2*i+1)，一个线程负责一对
__global__ void RopeNormalKernel(float* x, int n_heads, int head_dim, int pos,
                                 float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs_per_head = head_dim / 2;
  int total_pairs = n_heads * pairs_per_head;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / pairs_per_head;
  int pair = pair_index % pairs_per_head;
  int dim = pair * 2;
  int base = head * head_dim + dim;
  float freq =
      1.0f / powf(theta, static_cast<float>(dim) / static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base];
  float x1 = x[base + 1];
  x[base] = x0 * cos_val - x1 * sin_val;
  x[base + 1] = x0 * sin_val + x1 * cos_val;
}

// NeoX 布局配对前后半区 (i, i+head_dim/2)，频率仍按第 2*i 维计算
__global__ void RopeNeoXKernel(float* x, int n_heads, int head_dim, int pos,
                               float theta) {
  int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  int half_dim = head_dim / 2;
  int total_pairs = n_heads * half_dim;
  if (pair_index >= total_pairs) {
    return;
  }

  int head = pair_index / half_dim;
  int pair = pair_index % half_dim;
  int base = head * head_dim;
  float freq = 1.0f / powf(theta, static_cast<float>(2 * pair) /
                                      static_cast<float>(head_dim));
  float cos_val = cosf(static_cast<float>(pos) * freq);
  float sin_val = sinf(static_cast<float>(pos) * freq);
  float x0 = x[base + pair];
  float x1 = x[base + half_dim + pair];
  x[base + pair] = x0 * cos_val - x1 * sin_val;
  x[base + half_dim + pair] = x0 * sin_val + x1 * cos_val;
}

// 将元素/维度对数换算为线程块数，Kernel 内仍需保护非整块尾部
int GridFor(int n) { return (n + kBlockSize - 1) / kBlockSize; }

// shape/data 可独立修改，上传前核对正维度、元素数溢出和实际存储长度
void ValidateHostStorage(const Tensor& tensor, const char* caller) {
  if (tensor.shape.empty()) {
    throw std::runtime_error(std::string(caller) + ": empty shape");
  }
  size_t elements = 1;
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

// 主机逐元素接口校验 shape 和存储；设备重载还要求两个输入位于同一设备
void RequireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
  ValidateHostStorage(a, caller);
  ValidateHostStorage(b, caller);
  if (a.shape != b.shape) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.ShapeStringShort() + " vs " + b.ShapeStringShort());
  }
}

void RequireSameShape(const CudaTensor& a, const CudaTensor& b, const char* caller) {
  if (a.shape() != b.shape()) {
    throw std::runtime_error(std::string(caller) + ": shape mismatch " +
                             a.ShapeStringShort() + " vs " + b.ShapeStringShort());
  }
  if (a.device_id() != b.device_id()) {
    throw std::runtime_error(std::string(caller) +
                             ": tensors are on different CUDA devices");
  }
}

// RMSNorm/Softmax 仅接受非空一维输入，设备重载使用相同的秩约束
void Require1D(const Tensor& x, const char* caller) {
  ValidateHostStorage(x, caller);
  if (x.num_dims() != 1) {
    throw std::runtime_error(std::string(caller) + ": expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

void Require1D(const CudaTensor& x, const char* caller) {
  if (x.num_dims() != 1) {
    throw std::runtime_error(std::string(caller) + ": expected 1D tensor, got " +
                             x.ShapeStringShort());
  }
  if (x.size() == 0) {
    throw std::runtime_error(std::string(caller) + ": empty tensor");
  }
}

// 主机 RoPE 用二维 shape 推导头配置，并在分配/旋转前拒绝非法位置和配对维度
void ValidateRopeInputs(const Tensor& q, const Tensor& k, int pos, float theta) {
  ValidateHostStorage(q, "CudaRope");
  ValidateHostStorage(k, "CudaRope");
  if (q.num_dims() != 2 || k.num_dims() != 2) {
    throw std::runtime_error("CudaRope: expected 2D tensors");
  }
  if (pos < 0) {
    throw std::out_of_range("CudaRope: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error("CudaRope: theta must be finite and positive");
  }
  if (q.shape[1] != k.shape[1]) {
    throw std::runtime_error("CudaRope: q and k head_dim mismatch " +
                             q.ShapeStringShort() + " vs " + k.ShapeStringShort());
  }
  if (q.shape[1] <= 0 || q.shape[1] % 2 != 0) {
    throw std::runtime_error("CudaRope: head_dim must be positive and even");
  }
}

// 设备 RoPE 可接收扁平激活，按元素数核对头配置，同时检查所属设备
void ValidateRopeDeviceInputs(const CudaTensor& q, const CudaTensor& k, int n_heads,
                              int n_kv_heads, int head_dim, int pos, float theta,
                              int device_id) {
  if (q.device_id() != device_id || k.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRopeDeviceInput: input tensor is on a different CUDA device");
  }
  if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("CudaRopeDeviceInput: invalid head shape");
  }
  if (q.size() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("CudaRopeDeviceInput: q shape mismatch");
  }
  if (k.size() != static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim)) {
    throw std::runtime_error("CudaRopeDeviceInput: k shape mismatch");
  }
  if (pos < 0) {
    throw std::out_of_range("CudaRopeDeviceInput: position must be non-negative");
  }
  if (!std::isfinite(theta) || theta <= 0.0f) {
    throw std::runtime_error(
        "CudaRopeDeviceInput: theta must be finite and positive");
  }
}

// 校验借用权重的逻辑布局和 token 范围；裸指针的设备与容量由调用方保证
void ValidateEmbeddingDeviceWeightInputs(const void* embedding_data,
                                         const std::vector<int>& embedding_shape,
                                         int token_id) {
  if (embedding_data == nullptr) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: embedding data is null");
  }
  if (embedding_shape.size() != 2) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: expected 2D embedding weight");
  }
  int vocab_size = embedding_shape[0];
  int dim = embedding_shape[1];
  if (vocab_size <= 0 || dim <= 0) {
    throw std::runtime_error(
        "CudaEmbeddingLookupDeviceWeight: embedding shape must be positive");
  }
  if (token_id < 0 || token_id >= vocab_size) {
    throw std::out_of_range(
        "CudaEmbeddingLookupDeviceWeight: token id out of range");
  }
}

// 常驻 Norm 权重必须与输入同 shape；eps 非负且输入属于指定设备
void ValidateRmsNormDeviceWeightInputs(const CudaTensor& x, const void* weight_data,
                                       const std::vector<int>& weight_shape,
                                       float eps, int device_id) {
  Require1D(x, "CudaRmsNormDeviceWeight");
  if (weight_data == nullptr) {
    throw std::runtime_error("CudaRmsNormDeviceWeight: weight data is null");
  }
  if (x.shape() != weight_shape) {
    throw std::runtime_error("CudaRmsNormDeviceWeight: x and weight shape mismatch");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps < 0.0f) {
    throw std::runtime_error(
        "CudaRmsNormDeviceWeight: eps must be finite and non-negative");
  }
}

// 此设备权重加法只支持完整同 shape 数据，不把较短向量隐式广播
void ValidateAddDeviceWeightInputs(const CudaTensor& a, const void* b_data,
                                   const std::vector<int>& b_shape, int device_id) {
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: input tensor is on a different CUDA "
        "device");
  }
  if (b_data == nullptr) {
    throw std::runtime_error("CudaElementwiseAddDeviceWeight: weight data is null");
  }
  if (a.shape() != b_shape) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceWeight: input and weight shape mismatch");
  }
}

}  // namespace

// 主机接口临时上传激活和 Norm 权重，两步 Kernel 计算后同步下载结果
Tensor CudaRmsNorm(const Tensor& x, const Tensor& weight, float eps, int device_id) {
  Require1D(x, "CudaRmsNorm");
  Require1D(weight, "CudaRmsNorm");
  if (x.shape != weight.shape) {
    throw std::runtime_error("CudaRmsNorm: x and weight shape mismatch");
  }
  if (!std::isfinite(eps) || eps < 0.0f) {
    throw std::runtime_error("CudaRmsNorm: eps must be finite and non-negative");
  }

  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));
  w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));
  CheckCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)),
               "cudaMemset(RmsNorm sum)");

  int n = static_cast<int>(x.size());
  SumSquaresKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()), static_cast<float*>(sum_dev.data()),
      n);
  CheckLastKernel("SumSquaresKernel");
  RmsNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x_dev.data()),
      static_cast<const float*>(w_dev.data()), static_cast<float*>(y_dev.data()),
      static_cast<const float*>(sum_dev.data()), n, eps);
  CheckLastKernel("RmsNormKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

// 主机 SiLU 只上传输入和下载输出，临时显存由 Buffer 的 RAII 生命周期管理
Tensor CudaSilu(const Tensor& x, int device_id) {
  ValidateHostStorage(x, "CudaSilu");
  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());
  SiluKernel<<<GridFor(n), kBlockSize>>>(static_cast<const float*>(x_dev.data()),
                                         static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("SiluKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

// 主机乘法上传两个输入，并下载新建的输出；输入数据不被修改
Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseMul");
  CudaSetDevice(device_id);
  Tensor y(a.shape, 0.0f);
  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.Upload(a.data.data(), a.size() * sizeof(float));
  b_dev.Upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  ElementwiseMulKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()), static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("ElementwiseMulKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

// 借用常驻 Embedding 权重，仅分配并返回设备激活，不上传权重或下载结果
CudaTensor CudaEmbeddingLookupDeviceWeight(const void* embedding_data,
                                           const std::vector<int>& embedding_shape,
                                           int token_id, int device_id) {
  ValidateEmbeddingDeviceWeightInputs(embedding_data, embedding_shape, token_id);
  CudaSetDevice(device_id);

  int dim = embedding_shape[1];
  CudaTensor y({dim}, device_id);
  EmbeddingLookupKernel<<<GridFor(dim), kBlockSize>>>(
      static_cast<const float*>(embedding_data), static_cast<float*>(y.data()),
      token_id, dim);
  CheckLastKernel("EmbeddingLookupKernel");
  return y;
}

// 输入已在设备，但 Norm 权重仍位于主机；上传权重后复用常驻权重实现
CudaTensor CudaRmsNormDeviceInput(const CudaTensor& x, const Tensor& weight,
                                  float eps, int device_id) {
  Require1D(x, "CudaRmsNormDeviceInput");
  Require1D(weight, "CudaRmsNormDeviceInput");
  if (x.shape() != weight.shape) {
    throw std::runtime_error("CudaRmsNormDeviceInput: x and weight shape mismatch");
  }
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: input tensor is on a different CUDA device");
  }
  if (!std::isfinite(eps) || eps < 0.0f) {
    throw std::runtime_error(
        "CudaRmsNormDeviceInput: eps must be finite and non-negative");
  }

  CudaSetDevice(device_id);
  CudaDeviceBuffer w_dev(weight.size() * sizeof(float), device_id);
  w_dev.Upload(weight.data.data(), weight.size() * sizeof(float));
  return CudaRmsNormDeviceWeight(x, w_dev.data(), weight.shape, eps, device_id);
}

// 输入/权重/输出均在设备；仅临时分配平方和，并在每次计算前清零
CudaTensor CudaRmsNormDeviceWeight(const CudaTensor& x, const void* weight_data,
                                   const std::vector<int>& weight_shape, float eps,
                                   int device_id) {
  ValidateRmsNormDeviceWeightInputs(x, weight_data, weight_shape, eps, device_id);
  CudaSetDevice(device_id);

  CudaTensor y(x.shape(), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  CheckCudaOps(cudaMemset(sum_dev.data(), 0, sizeof(float)),
               "cudaMemset(RmsNorm sum)");

  int n = static_cast<int>(x.size());
  SumSquaresKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<float*>(sum_dev.data()), n);
  CheckLastKernel("SumSquaresKernel");
  RmsNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(x.data()), static_cast<const float*>(weight_data),
      static_cast<float*>(y.data()), static_cast<const float*>(sum_dev.data()), n,
      eps);
  CheckLastKernel("RmsNormKernel");

  return y;
}

// 使用已有设备输入并返回独占的设备输出，校验输入设备后再启动 Kernel
CudaTensor CudaSiluDeviceInput(const CudaTensor& x, int device_id) {
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaSiluDeviceInput: input tensor is on a different CUDA device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(x.shape(), device_id);

  int n = static_cast<int>(x.size());
  SiluKernel<<<GridFor(n), kBlockSize>>>(static_cast<const float*>(x.data()),
                                         static_cast<float*>(y.data()), n);
  CheckLastKernel("SiluKernel");

  return y;
}

// 两个激活均驻留设备，直接读取设备指针完成乘法，无主机传输
CudaTensor CudaElementwiseMulDeviceInput(const CudaTensor& a, const CudaTensor& b,
                                         int device_id) {
  RequireSameShape(a, b, "CudaElementwiseMulDeviceInput");
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseMulDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseMulKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseMulKernel");

  return y;
}

// 设备残差加法生成新输出，保持两个输入的显存和数据不变
CudaTensor CudaElementwiseAddDeviceInput(const CudaTensor& a, const CudaTensor& b,
                                         int device_id) {
  RequireSameShape(a, b, "CudaElementwiseAddDeviceInput");
  if (a.device_id() != device_id) {
    throw std::runtime_error(
        "CudaElementwiseAddDeviceInput: input tensor is on a different CUDA "
        "device");
  }
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b.data()),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  return y;
}

// 借用同 shape 的常驻 F32 数据完成加法，不复制或取得其所有权
CudaTensor CudaElementwiseAddDeviceWeight(const CudaTensor& a, const void* b_data,
                                          const std::vector<int>& b_shape,
                                          int device_id) {
  ValidateAddDeviceWeightInputs(a, b_data, b_shape, device_id);
  CudaSetDevice(device_id);
  CudaTensor y(a.shape(), device_id);

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a.data()), static_cast<const float*>(b_data),
      static_cast<float*>(y.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  return y;
}

// 主机加法与乘法采用相同的上传两个输入、下载输出的边界
Tensor CudaElementwiseAdd(const Tensor& a, const Tensor& b, int device_id) {
  RequireSameShape(a, b, "CudaElementwiseAdd");
  CudaSetDevice(device_id);
  Tensor y(a.shape, 0.0f);
  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  a_dev.Upload(a.data.data(), a.size() * sizeof(float));
  b_dev.Upload(b.data.data(), b.size() * sizeof(float));

  int n = static_cast<int>(a.size());
  ElementwiseAddKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<const float*>(a_dev.data()),
      static_cast<const float*>(b_dev.data()), static_cast<float*>(y_dev.data()), n);
  CheckLastKernel("ElementwiseAddKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

// Q/K 可以具有不同头数，但 head_dim 相同；按所选布局在设备上原地更新
void CudaRopeDeviceInput(CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads,
                         int head_dim, int pos, float theta, RopeType rope_type,
                         int device_id) {
  ValidateRopeDeviceInputs(q, k, n_heads, n_kv_heads, head_dim, pos, theta,
                           device_id);
  CudaSetDevice(device_id);

  int q_pairs = n_heads * (head_dim / 2);
  int k_pairs = n_kv_heads * (head_dim / 2);
  if (rope_type == RopeType::kNeoX) {
    RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(static_cast<float*>(q.data()),
                                                     n_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNeoXKernel(q)");
    RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNeoXKernel(k)");
  } else {
    RopeNormalKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q.data()), n_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNormalKernel(q)");
    RopeNormalKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k.data()), n_kv_heads, head_dim, pos, theta);
    CheckLastKernel("RopeNormalKernel(k)");
  }
}

// 三步稳定 Softmax 在同一默认流中顺序执行：最大值、指数求和、归一化
Tensor CudaSoftmax(const Tensor& x, int device_id) {
  Require1D(x, "CudaSoftmax");
  CudaSetDevice(device_id);
  Tensor y(x.shape, 0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  CudaDeviceBuffer max_dev(sizeof(float), device_id);
  CudaDeviceBuffer sum_dev(sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  int n = static_cast<int>(x.size());
  SoftmaxMaxKernel<<<1, kBlockSize>>>(static_cast<const float*>(x_dev.data()),
                                      static_cast<float*>(max_dev.data()), n);
  CheckLastKernel("SoftmaxMaxKernel");
  SoftmaxExpSumKernel<<<1, kBlockSize>>>(static_cast<const float*>(x_dev.data()),
                                         static_cast<float*>(y_dev.data()),
                                         static_cast<const float*>(max_dev.data()),
                                         static_cast<float*>(sum_dev.data()), n);
  CheckLastKernel("SoftmaxExpSumKernel");
  SoftmaxNormKernel<<<GridFor(n), kBlockSize>>>(
      static_cast<float*>(y_dev.data()), static_cast<const float*>(sum_dev.data()),
      n);
  CheckLastKernel("SoftmaxNormKernel");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  return y;
}

// 主机 Q/K 上传后在设备上原地旋转，再分别下载覆盖调用方的原 Tensor
void CudaRope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type,
              int device_id) {
  ValidateRopeInputs(q, k, pos, theta);
  CudaSetDevice(device_id);
  CudaDeviceBuffer q_dev(q.size() * sizeof(float), device_id);
  CudaDeviceBuffer k_dev(k.size() * sizeof(float), device_id);
  q_dev.Upload(q.data.data(), q.size() * sizeof(float));
  k_dev.Upload(k.data.data(), k.size() * sizeof(float));

  int q_pairs = q.shape[0] * (q.shape[1] / 2);
  int k_pairs = k.shape[0] * (k.shape[1] / 2);
  if (rope_type == RopeType::kNeoX) {
    RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    CheckLastKernel("RopeNeoXKernel(q)");
    RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    CheckLastKernel("RopeNeoXKernel(k)");
  } else {
    RopeNormalKernel<<<GridFor(q_pairs), kBlockSize>>>(
        static_cast<float*>(q_dev.data()), q.shape[0], q.shape[1], pos, theta);
    CheckLastKernel("RopeNormalKernel(q)");
    RopeNormalKernel<<<GridFor(k_pairs), kBlockSize>>>(
        static_cast<float*>(k_dev.data()), k.shape[0], k.shape[1], pos, theta);
    CheckLastKernel("RopeNormalKernel(k)");
  }

  q_dev.Download(q.data.data(), q.size() * sizeof(float));
  k_dev.Download(k.data.data(), k.size() * sizeof(float));
}

}  // namespace mini_llama
