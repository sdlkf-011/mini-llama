#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_

#include <string>
#include <vector>

#include "mini_llama/tensor.h"

namespace mini_llama {

enum class RopeType {
  kNormal,
  kNeoX,
};

struct ModelConfig {                       // 模型配置
  int vocab_size = 128;                    // 词表大小
  int dim = 32;                            // 隐藏维度
  int hidden_dim = 86;                     // 前馈网络隐藏维度
  int n_layers = 2;                        // 层数
  int n_heads = 4;                         // 注意力头数
  int n_kv_heads = 4;                      // KV 头数
  int head_dim = 8;                        // 每个头的维度
  int max_seq_len = 128;                   // 最大序列长度
  float rope_theta = 10000.0f;             // RoPE theta
  float rms_norm_eps = 1e-5f;              // RMSNorm epsilon
  RopeType rope_type = RopeType::kNormal;  // RoPE 类型
};

struct LayerWeights {     // 单个 Transformer 层的权重
  Tensor attention_norm;  // [dim] F32，注意力子层的 RMSNorm 缩放权重
  Tensor wq;              // [n_heads * head_dim, dim] F32，查询投影权重
  Tensor wk;              // [n_kv_heads * head_dim, dim] F32，键投影权重
  Tensor wv;              // [n_kv_heads * head_dim, dim] F32，值投影权重
  Tensor wo;        // [dim, n_heads * head_dim] F32，注意力输出投影权重
  Tensor bq;        // 可选 [n_heads * head_dim] F32，查询投影偏置
  Tensor bk;        // 可选 [n_kv_heads * head_dim] F32，键投影偏置
  Tensor bv;        // 可选 [n_kv_heads * head_dim] F32，值投影偏置
  Tensor ffn_norm;  // [dim] F32，前馈网络子层的 RMSNorm 缩放权重
  Tensor w_gate;    // [hidden_dim, dim] F32，SwiGLU 门控投影权重
  Tensor w_up;      // [hidden_dim, dim] F32，SwiGLU 上投影权重
  Tensor w_down;    // [dim, hidden_dim] F32，前馈网络下投影权重
};

struct MiniLlamaModel {
  ModelConfig config;                // 模型结构与推理超参数
  bool loaded = false;               // 模型权重是否已经成功加载
  std::string load_error;            // 模型加载失败时的错误信息
  Tensor token_embedding;            // [vocab_size, dim] F32，词元嵌入表
  std::vector<LayerWeights> layers;  // [n_layers]，各 Transformer 层权重
  Tensor final_norm;                 // [dim] F32，最终 RMSNorm 缩放权重
  Tensor lm_head;  // [vocab_size, dim] F32，输出词表投影权重
};

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_MODEL_H_
