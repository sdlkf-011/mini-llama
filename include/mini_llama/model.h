#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "mini_llama/cuda_runtime.h"
#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// RoPE 的维度配对方式：Normal 用于默认 Llama 路径，NeoX 已用于 Qwen2。
enum class RopeType {
  kNormal,  // 相邻维度成对旋转
  kNeoX,    // 前后半区的对应维度成对旋转
};

// 模型结构与推理超参数。head_dim 由加载器根据 dim / n_heads 推导
struct ModelConfig {
  int vocab_size = 128;                    // 词表大小
  int dim = 32;                            // 隐藏状态维度
  int hidden_dim = 86;                     // 前馈网络中间维度
  int n_layers = 2;                        // Transformer 层数
  int n_heads = 4;                         // 查询头数量
  int n_kv_heads = 4;                      // 键值头数量，允许使用 GQA
  int head_dim = 8;                        // 单个注意力头维度
  int max_seq_len = 128;                   // 最大上下文长度
  float rope_theta = 10000.0f;             // RoPE 基频参数
  float rms_norm_eps = 1e-5f;              // RMSNorm 数值稳定项
  RopeType rope_type = RopeType::kNormal;  // Q/K 的旋转配对布局
};

// 单个 Transformer 层的 CPU 权重。线性权重保持 [out, in] 排列并保留量化格式
struct LayerWeights {
  Tensor attention_norm;   // [dim] F32，注意力子层的 RMSNorm 缩放权重
  QuantizedTensor wq;      // [n_heads * head_dim, dim] F32/Q8_0/Q4_0/Q4_1
  QuantizedTensor wk;      // [n_kv_heads * head_dim, dim]，键投影权重
  QuantizedTensor wv;      // [n_kv_heads * head_dim, dim]，值投影权重
  Tensor bq;               // 可选 [n_heads * head_dim] F32，查询投影偏置
  Tensor bk;               // 可选 [n_kv_heads * head_dim] F32，键投影偏置
  Tensor bv;               // 可选 [n_kv_heads * head_dim] F32，值投影偏置
  QuantizedTensor wo;      // [dim, n_heads * head_dim]，注意力输出投影权重
  Tensor ffn_norm;         // [dim] F32，前馈子层的 RMSNorm 缩放权重
  QuantizedTensor w_gate;  // [hidden_dim, dim]，SwiGLU 门控投影权重
  QuantizedTensor w_up;    // [hidden_dim, dim]，SwiGLU 上投影权重
  QuantizedTensor w_down;  // [dim, hidden_dim]，SwiGLU 下投影权重
};

// 单个已上传的 CUDA 权重。保留原始存储格式，显存由 CudaDeviceBuffer 独占管理
struct CudaUploadedWeight {
  std::string name;                  // 权重名称，如 lm_head 或 layers.0.wq
  QuantType type = QuantType::kF32;  // 原始权重的存储格式
  bool linear = true;                // 是否为线性层权重
  std::vector<int> shape;            // 逻辑形状，线性权重为 [out, in]
  size_t block_count = 0;            // 量化块数量，F32 权重为 0
  CudaDeviceBuffer buffer;  // 权重数据的设备存储，独占显存所有权
};

// 模型的 CUDA 权重快照与统计信息。上传统计独立记录，运行统计由推理调用方累计
// 默认 CPU 前向不更新运行字段；显式 CUDA Linear 路径记录实际调用与边界拷贝
// 设备 Attention/KV Cache 尚未实现，CPU Attention fallback 单独计数
struct CudaModelWeights {
  int device_id = 0;                 // 权重所在的 CUDA 设备编号
  size_t uploaded_weight_count = 0;  // 已上传的权重数量，包含稠密与线性权重
  size_t uploaded_bytes = 0;    // 已上传权重数据占用的显存字节数
  size_t linear_calls = 0;      // CUDA 线性层累计调用次数
  size_t activation_calls = 0;  // CUDA 激活与辅助算子累计调用次数
  size_t attention_calls = 0;   // CUDA 注意力累计调用次数
  size_t attention_cpu_fallbacks = 0;  // 注意力回退到 CPU 的累计次数
  size_t kv_cache_write_bytes = 0;     // 累计写入 KV Cache 的字节数
  size_t kv_cache_read_bytes = 0;      // 累计读取 KV Cache 的字节数
  size_t host_to_device_copies = 0;  // 推理期间 CPU 到 GPU 的累计拷贝次数
  size_t device_to_host_copies = 0;  // 推理期间 GPU 到 CPU 的累计拷贝次数
  size_t host_to_device_bytes = 0;  // 推理期间 CPU 到 GPU 的累计拷贝字节数
  size_t device_to_host_bytes = 0;  // 推理期间 GPU 到 CPU 的累计拷贝字节数
  std::vector<CudaUploadedWeight> weights;  // [uploaded_weight_count]，设备权重列表
};

// 完整模型对象。只有 loaded 为 true 时，其配置与权重才可用于前向计算
struct MiniLlamaModel {
  ModelConfig config;                // 模型结构与推理超参数
  bool loaded = false;               // 模型权重是否已经成功加载
  std::string load_error;            // 模型加载失败时的错误信息
  Tensor token_embedding;            // [vocab_size, dim] F32，词元嵌入表
  std::vector<LayerWeights> layers;  // [n_layers]，各 Transformer 层权重
  Tensor final_norm;                 // [dim] F32，最终 RMSNorm 缩放权重
  QuantizedTensor lm_head;           // [vocab_size, dim]，词表输出投影权重
  // 模型副本共享设备快照，但该引用不选择计算后端；修改 CPU 权重后需重新上传
  std::shared_ptr<CudaModelWeights> cuda_weights;
};

// 返回模型权重以当前格式存储时占用的数据字节数
size_t ModelWeightBytes(const MiniLlamaModel& model);

// 返回相同模型权重全部以 F32 存储时需要的数据字节数
size_t ModelWeightBytesF32(const MiniLlamaModel& model);

// 当前支持 F32/Q8_0 Linear 权重；Embedding、Norm 和可选 Bias 上传为 F32
// 保留 CPU 权重；全部上传成功后替换快照，失败保留原设备权重
// 此接口只管理设备权重，不改变当前前向计算的后端
void UploadModelWeightsToCuda(MiniLlamaModel& model, int device_id = 0);

// 解除当前模型的快照引用；其他模型副本持有同一快照时，显存继续保留
void ClearModelCudaWeights(MiniLlamaModel& model);

// 仅判断快照引用非空，不代表权重完整、设备可用或 CUDA 前向已接入
bool ModelHasCudaWeights(const MiniLlamaModel& model);

// 读取全部已上传权重数量，包含 embedding/norm/bias；无快照时返回零
size_t ModelCudaUploadedWeightCount(const MiniLlamaModel& model);

// 分类型计数只包含 Linear 权重，不包含 F32 Embedding、Norm 和 Bias
// 读取 F32 Linear 数量；同为 F32 的稠密权重不计入此分类
size_t ModelCudaUploadedF32WeightCount(const MiniLlamaModel& model);

// 读取 Q8_0 Linear 数量
size_t ModelCudaUploadedQ80WeightCount(const MiniLlamaModel& model);

// 当前上传入口不支持 Q4_0/Q4_1，正常上传后的这两项为 0
// 读取 Q4_0 Linear 数量，当前正常上传结果为零
size_t ModelCudaUploadedQ40WeightCount(const MiniLlamaModel& model);

// 读取 Q4_1 Linear 数量，当前正常上传结果为零
size_t ModelCudaUploadedQ41WeightCount(const MiniLlamaModel& model);

// 读取上传权重数据占用的显存字节数，不包含 CPU 副本、激活或 KV Cache
size_t ModelCudaMemoryBytes(const MiniLlamaModel& model);

// 仅清零运行统计，保留设备权重及上传统计；共享快照的模型副本也会看到清零
void ResetModelCudaRuntimeStats(MiniLlamaModel& model);

// 读取已记录的 CUDA Linear 调用次数；以下运行统计接口无快照时均返回零
size_t ModelCudaLinearCalls(const MiniLlamaModel& model);

// 读取已记录的 CUDA 激活与辅助算子调用次数
size_t ModelCudaActivationCalls(const MiniLlamaModel& model);

// 读取已记录的设备 Attention 调用次数
size_t ModelCudaAttentionCalls(const MiniLlamaModel& model);

// 读取设备推理中的 CPU Attention 回退次数，普通 CPU 前向不计作回退
size_t ModelCudaAttentionCpuFallbacks(const MiniLlamaModel& model);

// 读取已记录的设备 KV Cache 写入字节数
size_t ModelCudaKvCacheWriteBytes(const MiniLlamaModel& model);

// 读取已记录的设备 KV Cache 读取字节数
size_t ModelCudaKvCacheReadBytes(const MiniLlamaModel& model);

// 读取推理阶段主机到设备的拷贝次数，不计权重上传
size_t ModelCudaHostToDeviceCopies(const MiniLlamaModel& model);

// 读取推理阶段设备到主机的拷贝次数
size_t ModelCudaDeviceToHostCopies(const MiniLlamaModel& model);

// 读取推理阶段主机到设备的拷贝字节数
size_t ModelCudaHostToDeviceBytes(const MiniLlamaModel& model);

// 读取推理阶段设备到主机的拷贝字节数
size_t ModelCudaDeviceToHostBytes(const MiniLlamaModel& model);

// 原地转换所有 Linear 权重，支持从已有 F32/Q8_0/Q4_0/Q4_1 格式重新量化
// Embedding、Norm 和 Bias 保持不变；成功改变权重后清除旧 CUDA 快照
// 失败保留原 CPU/设备权重；已是目标格式的权重仅校验，不重复量化
// 目标格式为 Q8_0；重新量化使用 F32 临时权重，不切换推理后端
void QuantizeModelToQ80(MiniLlamaModel& model);

// 目标格式为 Q4_0；复用现有 Q4_0 编码与舍入规则
void QuantizeModelToQ40(MiniLlamaModel& model);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_
