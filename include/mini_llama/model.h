#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// RoPE 的维度配对方式。默认模式用于 Llama，NeoX 模式为后续兼容模型保留
enum class RopeType {
  kNormal,
  kNeoX,
};

// 模型结构与推理超参数。head_dim 由加载器根据 dim / n_heads 推导
struct ModelConfig {
  int vocab_size = 128;         // 词表大小
  int dim = 32;                 // 隐藏状态维度
  int hidden_dim = 86;          // 前馈网络中间维度
  int n_layers = 2;             // Transformer 层数
  int n_heads = 4;              // 查询头数量
  int n_kv_heads = 4;           // 键值头数量，允许使用 GQA
  int head_dim = 8;             // 单个注意力头维度
  int max_seq_len = 128;        // 最大上下文长度
  float rope_theta = 10000.0f;  // RoPE 基频参数
  float rms_norm_eps = 1e-5f;   // RMSNorm 数值稳定项
  RopeType rope_type = RopeType::kNormal;
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

// 完整模型对象。只有 loaded 为 true 时，其配置与权重才可用于前向计算
struct MiniLlamaModel {
  ModelConfig config;                // 模型结构与推理超参数
  bool loaded = false;               // 模型权重是否已经成功加载
  std::string load_error;            // 模型加载失败时的错误信息
  Tensor token_embedding;            // [vocab_size, dim] F32，词元嵌入表
  std::vector<LayerWeights> layers;  // [n_layers]，各 Transformer 层权重
  Tensor final_norm;                 // [dim] F32，最终 RMSNorm 缩放权重
  QuantizedTensor lm_head;           // [vocab_size, dim]，词表输出投影权重
};

// 返回模型权重以当前格式存储时占用的数据字节数
size_t ModelWeightBytes(const MiniLlamaModel& model);

// 返回相同模型权重全部以 F32 存储时需要的数据字节数
size_t ModelWeightBytesF32(const MiniLlamaModel& model);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_
