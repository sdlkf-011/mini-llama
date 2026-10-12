#include "mini_llama/cuda_attention.h"

#ifdef MINI_LLAMA_USE_CUDA

#include <cuda_runtime_api.h>

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {

namespace {

// 默认 stream 保证分数读写顺序；按线程和设备保留一个可增长的工作缓冲区
CudaDeviceBuffer& AttentionScoreBuffer(size_t bytes, int device_id) {
  thread_local std::map<int, CudaDeviceBuffer> buffers;
  auto& buffer = buffers[device_id];
  if (buffer.bytes() < bytes) {
    size_t capacity = bytes;
    if (bytes <= 64 * 1024 * 1024) {
      capacity = 1;
      while (capacity < bytes) capacity *= 2;
    }
    buffer.Reset(capacity, device_id);
  }
  return buffer;
}

constexpr int kAttentionBlockSize = 128;

// 保留失败的 CUDA API 名称和 Runtime 错误原因
void CheckCudaAttention(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA attention error in " + std::string(expr) + ": " +
                             cudaGetErrorString(err));
  }
}

// 只检查启动错误；执行期错误由后续同步拷贝或 Runtime 操作报告
void CheckLastAttentionKernel(const char* kernel_name) {
  CheckCudaAttention(cudaGetLastError(), kernel_name);
}

// 计算临时分数与输出容量前检查 size_t 乘法溢出
size_t CheckedMul(size_t a, size_t b, const char* label) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    throw std::runtime_error(std::string("CudaAttentionDecode: size overflow for ") +
                             label);
  }
  return a * b;
}

// 在设备访存前验证缓存、GQA 头映射、因果端点及输出维度
void ValidateAttentionCache(const CudaKvCache& cache, int layer, int pos, int heads,
                            int kv_heads, int dim, int device) {
  if (cache.empty() || heads <= 0 || kv_heads <= 0 || dim <= 0 ||
      heads % kv_heads != 0 || cache.n_kv_heads() != kv_heads ||
      cache.head_dim() != dim || cache.device_id() != device) {
    throw std::runtime_error("CudaAttentionDecode: invalid cache, heads or device");
  }
  if (layer < 0 || layer >= cache.n_layers() || pos < 0 ||
      pos >= cache.max_seq_len()) {
    throw std::out_of_range("CudaAttentionDecode: layer or position out of range");
  }
  if (CheckedMul(heads, dim, "query elements") >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("CudaAttentionDecode: output dimension overflow");
  }
}

// 设备查询向量允许扁平或分头形状，但物理容量和所属设备必须匹配
void ValidateAttentionDeviceInputs(const CudaTensor& q, const CudaKvCache& cache,
                                   int layer, int pos, int heads, int kv_heads,
                                   int dim, int device) {
  ValidateAttentionCache(cache, layer, pos, heads, kv_heads, dim, device);
  const size_t elements = CheckedMul(heads, dim, "query elements");
  if (q.empty() || q.device_id() != device || q.size() != elements ||
      q.bytes() != CheckedMul(elements, sizeof(float), "query bytes") ||
      (q.shape() != std::vector<int>{heads, dim} &&
       q.shape() != std::vector<int>{static_cast<int>(elements)})) {
    throw std::runtime_error(
        "CudaAttentionDecodeDeviceInput: query shape/storage/device mismatch");
  }
}

// 每个线程块计算一个查询头与历史位置的缩放点积，分头读取连续 Key
// 将头和位置展开到网格 x 维，避免网格 y 维较小的上限
__global__ void AttentionScoresKernel(const float* q, const float* keys,
                                      float* scores, int layer, int max_seq_len,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      int pos, float scale) {
  int h = blockIdx.x / (pos + 1);
  int t = blockIdx.x % (pos + 1);
  if (h >= n_heads || t > pos) {
    return;
  }

  int kv_group = n_heads / n_kv_heads;
  int kv_head = h / kv_group;
  size_t key_base = ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) +
                      static_cast<size_t>(t)) *
                         static_cast<size_t>(n_kv_heads) +
                     static_cast<size_t>(kv_head)) *
                    static_cast<size_t>(head_dim);
  size_t q_base = static_cast<size_t>(h) * static_cast<size_t>(head_dim);

  __shared__ float partial[kAttentionBlockSize];
  float sum = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    sum += q[q_base + d] * keys[key_base + d];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scores[static_cast<size_t>(h) * static_cast<size_t>(pos + 1) +
           static_cast<size_t>(t)] = partial[0] * scale;
  }
}

// 每个线程块处理一个查询头：先用减最大值法归约 Softmax，再累加历史 Value
// 同组查询头共享 KV 头；各线程跨步负责输出维度
__global__ void AttentionReduceKernel(const float* scores, const float* values,
                                      float* out, int layer, int max_seq_len,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      int pos) {
  int h = blockIdx.x;
  if (h >= n_heads) {
    return;
  }

  int kv_group = n_heads / n_kv_heads;
  int kv_head = h / kv_group;
  size_t score_base = static_cast<size_t>(h) * static_cast<size_t>(pos + 1);

  __shared__ float partial[kAttentionBlockSize];
  float local_max = -INFINITY;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    local_max = fmaxf(local_max, scores[score_base + static_cast<size_t>(t)]);
  }
  partial[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] =
          fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  float max_score = partial[0];
  __syncthreads();  // 所有线程读完最大值后，才能复用共享内存

  float local_sum = 0.0f;
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    local_sum += expf(scores[score_base + static_cast<size_t>(t)] - max_score);
  }
  partial[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float denom = partial[0];

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float value = 0.0f;
    for (int t = 0; t <= pos; ++t) {
      float prob =
          expf(scores[score_base + static_cast<size_t>(t)] - max_score) / denom;
      size_t value_idx =
          ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) +
            static_cast<size_t>(t)) *
               static_cast<size_t>(n_kv_heads) +
           static_cast<size_t>(kv_head)) *
              static_cast<size_t>(head_dim) +
          static_cast<size_t>(d);
      value += prob * values[value_idx];
    }
    out[static_cast<size_t>(h) * static_cast<size_t>(head_dim) +
        static_cast<size_t>(d)] = value;
  }
}

}  // namespace

// 主机入口只负责输入校验与上传下载，计算逻辑统一复用设备入口
Tensor CudaAttentionDecode(const Tensor& q, const CudaKvCache& kv_cache, int layer,
                           int pos, int n_heads, int n_kv_heads, int head_dim,
                           int device_id) {
  ValidateAttentionCache(kv_cache, layer, pos, n_heads, n_kv_heads, head_dim,
                         device_id);
  if (q.shape != std::vector<int>{n_heads, head_dim} ||
      q.size() != CheckedMul(n_heads, head_dim, "query elements")) {
    throw std::runtime_error("CudaAttentionDecode: query shape/storage mismatch");
  }
  auto device_q = CudaTensorFromHost(q, device_id);
  auto out = CudaAttentionDecodeDeviceInput(device_q, kv_cache, layer, pos, n_heads,
                                            n_kv_heads, head_dim, device_id);
  return out.Download().ReshapeChecked({n_heads, head_dim}, "CudaAttentionDecode");
}

CudaTensor CudaAttentionDecodeDeviceInput(const CudaTensor& q,
                                          const CudaKvCache& kv_cache, int layer,
                                          int pos, int n_heads, int n_kv_heads,
                                          int head_dim, int device_id) {
  ValidateAttentionDeviceInputs(q, kv_cache, layer, pos, n_heads, n_kv_heads,
                                head_dim, device_id);
  CudaSetDevice(device_id);

  const size_t out_elements =
      CheckedMul(static_cast<size_t>(n_heads), static_cast<size_t>(head_dim),
                 "output elements");
  const size_t score_elements = CheckedMul(
      static_cast<size_t>(n_heads), static_cast<size_t>(pos + 1), "score elements");

  // 分数临时存储始终位于设备，输出保持一维以衔接现有 Linear 接口
  const size_t score_bytes =
      CheckedMul(score_elements, sizeof(float), "score bytes");
  CudaDeviceBuffer transient_scores;
  if (score_bytes > 64 * 1024 * 1024) {
    transient_scores.Reset(score_bytes, device_id);
  }
  auto& scores_dev = transient_scores.empty()
                         ? AttentionScoreBuffer(score_bytes, device_id)
                         : transient_scores;
  CudaTensor out({static_cast<int>(out_elements)}, device_id);

  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // 启动前检查实际设备的网格限制，拒绝无法表示的头/位置组合
  cudaDeviceProp properties;
  CheckCudaAttention(cudaGetDeviceProperties(&properties, device_id),
                     "cudaGetDeviceProperties");
  if (score_elements > static_cast<size_t>(properties.maxGridSize[0]) ||
      n_heads > properties.maxGridSize[0]) {
    throw std::runtime_error(
        "CudaAttentionDecode: launch grid exceeds device limit");
  }
  dim3 score_grid(static_cast<unsigned int>(score_elements));
  AttentionScoresKernel<<<score_grid, kAttentionBlockSize>>>(
      static_cast<const float*>(q.data()),
      static_cast<const float*>(kv_cache.keys_data()),
      static_cast<float*>(scores_dev.data()), layer, kv_cache.max_seq_len(), n_heads,
      n_kv_heads, head_dim, pos, scale);
  CheckLastAttentionKernel("AttentionScoresKernel");

  AttentionReduceKernel<<<n_heads, kAttentionBlockSize>>>(
      static_cast<const float*>(scores_dev.data()),
      static_cast<const float*>(kv_cache.values_data()),
      static_cast<float*>(out.data()), layer, kv_cache.max_seq_len(), n_heads,
      n_kv_heads, head_dim, pos);
  CheckLastAttentionKernel("AttentionReduceKernel");

  return out;
}

}  // namespace mini_llama

#endif
