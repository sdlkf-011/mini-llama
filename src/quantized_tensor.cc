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
  if (q.f32_data.size() != t.data.size()) {
    throw std::runtime_error("ToTensor: F32 storage does not match shape");
  }
  t.data = q.f32_data;
  return t;
}

// 将 F32 数据和形状复制到带格式标签的权重容器，不执行量化编码
QuantizedTensor ToQuantizedTensor(const Tensor& t) {
  QuantizedTensor q;
  q.type = QuantType::kF32;
  q.shape = t.shape;
  q.f32_data = t.data;
  return q;
}

}  // namespace mini_llama
