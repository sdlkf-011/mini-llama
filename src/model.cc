#include "mini_llama/model.h"

#include <cstddef>

namespace mini_llama {

// Tensor 仅承载 F32，实际存储字节数就是元素数乘 sizeof(float)
static size_t TensorBytes(const Tensor& tensor) {
  return tensor.data.size() * sizeof(float);
}

// 只统计 type 指定的活动存储数组，不把未使用 vector 的容量计入权重数据
static size_t QuantizedTensorBytes(const QuantizedTensor& tensor) {
  switch (tensor.type) {
    case QuantType::kF32:
      return tensor.f32_data.size() * sizeof(float);
    case QuantType::kQ80:
      return tensor.q8_0_data.size() * sizeof(BlockQ80);
    case QuantType::kQ40:
      return tensor.q4_0_data.size() * sizeof(BlockQ40);
    case QuantType::kQ41:
      return tensor.q4_1_data.size() * sizeof(BlockQ41);
  }
  return 0;
}

// 等价 F32 大小按逻辑元素数计算，与当前量化块格式无关
static size_t QuantizedTensorF32Bytes(const QuantizedTensor& qt) {
  return qt.num_elements() * sizeof(float);
}

// 统计模型持有的权重数据，不包含对象、shape vector 和 allocator 元数据
size_t ModelWeightBytes(const MiniLlamaModel& model) {
  size_t bytes = TensorBytes(model.token_embedding) + TensorBytes(model.final_norm) +
                 QuantizedTensorBytes(model.lm_head);
  for (const LayerWeights& layer : model.layers) {
    // Norm 和 bias 按 F32 统计，Linear 权重按当前实际存储格式统计
    bytes += TensorBytes(layer.attention_norm) + TensorBytes(layer.bq) +
             TensorBytes(layer.bk) + TensorBytes(layer.bv) +
             TensorBytes(layer.ffn_norm) + QuantizedTensorBytes(layer.wq) +
             QuantizedTensorBytes(layer.wk) + QuantizedTensorBytes(layer.wv) +
             QuantizedTensorBytes(layer.wo) + QuantizedTensorBytes(layer.w_gate) +
             QuantizedTensorBytes(layer.w_up) + QuantizedTensorBytes(layer.w_down);
  }
  return bytes;
}

// 稠密张量沿用实际大小，QuantizedTensor 则按逻辑元素数换算为 F32
size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
  size_t bytes = TensorBytes(model.token_embedding) + TensorBytes(model.final_norm) +
                 QuantizedTensorF32Bytes(model.lm_head);
  for (const LayerWeights& layer : model.layers) {
    // 稠密权重保持原大小，Linear 权重按逻辑元素数统一换算为 F32
    bytes +=
        TensorBytes(layer.attention_norm) + TensorBytes(layer.bq) +
        TensorBytes(layer.bk) + TensorBytes(layer.bv) + TensorBytes(layer.ffn_norm) +
        QuantizedTensorF32Bytes(layer.wq) + QuantizedTensorF32Bytes(layer.wk) +
        QuantizedTensorF32Bytes(layer.wv) + QuantizedTensorF32Bytes(layer.wo) +
        QuantizedTensorF32Bytes(layer.w_gate) + QuantizedTensorF32Bytes(layer.w_up) +
        QuantizedTensorF32Bytes(layer.w_down);
  }
  return bytes;
}

}  // namespace mini_llama
