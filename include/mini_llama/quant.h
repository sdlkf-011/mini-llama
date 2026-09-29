#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_QUANT_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_QUANT_H_

#include <vector>

#include "mini_llama/quantized_tensor.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 将 F32 张量按最后一维逐行量化为 Q8_0 块
std::vector<BlockQ80> QuantizeToQ80(const Tensor& src);

// 将 Q8_0 块还原为指定形状的 F32 张量
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape);

// 将 F32 张量按最后一维逐行量化为 Q4_0 块
std::vector<BlockQ40> QuantizeToQ40(const Tensor& src);

// 将 Q4_0 块还原为指定形状的 F32 张量
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape);

// 将 GGUF 中已有的 Q4_1 块还原为指定形状的 F32 张量
Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape);

// 权重保持量化块存储，点积时逐块即时反量化，不分配完整 F32 权重
Tensor LinearQ80(const Tensor& input, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape);

// 使用 Q4_0 权重执行线性层，输入和输出保持为 F32
Tensor LinearQ40(const Tensor& input, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape);

// 使用 Q4_1 权重执行线性层，输入和输出保持为 F32
Tensor LinearQ41(const Tensor& input, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_QUANT_H_
