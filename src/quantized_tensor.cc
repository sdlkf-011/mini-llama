#include "mini_llama/quantized_tensor.h"

#include <stdexcept>

#include "mini_llama/tensor.h"

namespace mini_llama {

// Tensor 与 QuantizedTensor 之间的 F32 转换辅助函数
Tensor ToTensor(const QuantizedTensor& q) {
  if (q.type != QuantType::kF32) {
    throw std::runtime_error(
        "ToTensor: expected F32 QuantizedTensor, got quantized type");
  }
  Tensor t(q.shape, 0.0f);
  if (q.f32_data.size() == t.data.size()) {
    t.data = q.f32_data;
  }
  return t;
}

// 仅在两种类型之间转换 F32 数据，用于兼容当前仅支持 F32 的计算路径
QuantizedTensor ToQuantizedTensor(const Tensor& t) {
  QuantizedTensor q;
  q.type = QuantType::kF32;
  q.shape = t.shape;
  q.f32_data = t.data;
  return q;
}

}  // namespace mini_llama
