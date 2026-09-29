#include "mini_llama/quant.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

// 将 F32 缩放因子编码为 GGUF 量化块使用的 IEEE FP16 位表示
uint16_t FloatToFp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    uint32_t shifted = mantissa >> (1 - exponent);
    if ((shifted & 0x1000U) != 0) {
      shifted += 0x2000U;
    }
    return static_cast<uint16_t>(sign | (shifted >> 13));
  }

  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00U);
  }

  if ((mantissa & 0x1000U) != 0) {
    mantissa += 0x2000U;
    if ((mantissa & 0x800000U) != 0) {
      mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
      }
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               (mantissa >> 13));
}

float Fp16ToFloat(uint16_t half) {
  // 手动展开 FP16，避免依赖平台特定的半精度浮点类型
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000U) << 16;
  const uint32_t exponent = (half >> 10) & 0x1fU;
  const uint32_t mantissa = half & 0x03ffU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      const float value = std::ldexp(static_cast<float>(mantissa), -24);
      return sign == 0 ? value : -value;
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }

  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  // 在计算元素总数的同时拒绝空形状、非正维度和乘法溢出
  if (shape.empty()) {
    throw std::runtime_error(std::string(caller) + ": shape must not be empty");
  }

  size_t total = 1;
  for (int dimension : shape) {
    if (dimension <= 0 || total > std::numeric_limits<size_t>::max() /
                                      static_cast<size_t>(dimension)) {
      throw std::runtime_error(std::string(caller) + ": invalid shape");
    }
    total *= static_cast<size_t>(dimension);
  }
  return total;
}

struct RowLayout {
  int row_size;
  int row_count;
};

RowLayout GetRowLayout(const std::vector<int>& shape, const char* caller) {
  // 量化块沿最后一维排列，其余维度统一折叠为行数
  const size_t total = CheckedNumel(shape, caller);
  const int row_size = shape.back();
  const size_t row_count = total / static_cast<size_t>(row_size);
  if (row_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(caller) + ": row count overflow");
  }
  return {row_size, static_cast<int>(row_count)};
}

template <int BlockSize>
size_t ExpectedBlockCount(const RowLayout& layout) {
  // 每行独立补齐到完整块，块不能跨越行边界
  const size_t blocks_per_row =
      static_cast<size_t>((layout.row_size + BlockSize - 1) / BlockSize);
  return static_cast<size_t>(layout.row_count) * blocks_per_row;
}

template <typename Block, int BlockSize, typename Decode>
Tensor DequantizeBlocks(const std::vector<Block>& blocks,
                        const std::vector<int>& shape, const char* caller,
                        Decode decode) {
  // 共用遍历负责形状和块数校验，具体格式由 decode 解码单个元素
  const RowLayout layout = GetRowLayout(shape, caller);
  const int blocks_per_row = (layout.row_size + BlockSize - 1) / BlockSize;
  if (blocks.size() != ExpectedBlockCount<BlockSize>(layout)) {
    throw std::runtime_error(std::string(caller) + ": block count mismatch");
  }

  Tensor output(shape, 0.0f);
  for (int row = 0; row < layout.row_count; ++row) {
    const size_t row_offset = static_cast<size_t>(row) * layout.row_size;
    const size_t block_offset = static_cast<size_t>(row) * blocks_per_row;
    for (int column = 0; column < layout.row_size; ++column) {
      output.data[row_offset + column] =
          decode(blocks[block_offset + static_cast<size_t>(column / BlockSize)],
                 column % BlockSize);
    }
  }
  return output;
}

struct LinearLayout {
  int in_features;
  int out_features;
  bool input_is_1d;
};

LinearLayout ValidateLinear(const Tensor& input,
                            const std::vector<int>& weight_shape,
                            const char* caller) {
  // 当前推理路径支持单个 token 的一维输入或批大小为 1 的二维输入
  if (weight_shape.size() != 2 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
    throw std::runtime_error(std::string(caller) +
                             ": expected positive 2D weight shape");
  }
  int in_features = 0;
  bool input_is_1d = false;
  if (input.num_dims() == 1) {
    in_features = input.shape[0];
    input_is_1d = true;
  } else if (input.num_dims() == 2 && input.shape[0] == 1) {
    in_features = input.shape[1];
  } else {
    throw std::runtime_error(
        std::string(caller) +
        ": expected input shape [in_features] or [1, in_features]");
  }
  if (weight_shape[1] != in_features) {
    throw std::runtime_error(std::string(caller) + ": dimension mismatch");
  }
  return {in_features, weight_shape[0], input_is_1d};
}

template <typename Block, int BlockSize, typename Decode>
Tensor LinearBlocks(const Tensor& input, const std::vector<Block>& blocks,
                    const std::vector<int>& weight_shape, const char* caller,
                    Decode decode) {
  // 点积过程中即时解码权重元素，避免构造完整的 F32 权重张量
  const LinearLayout layout = ValidateLinear(input, weight_shape, caller);
  const int blocks_per_row = (layout.in_features + BlockSize - 1) / BlockSize;
  const size_t expected_blocks =
      static_cast<size_t>(layout.out_features) * blocks_per_row;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(std::string(caller) + ": block count mismatch");
  }

  Tensor output(layout.input_is_1d ? std::vector<int>{layout.out_features}
                                   : std::vector<int>{1, layout.out_features},
                0.0f);

  for (int output_index = 0; output_index < layout.out_features; ++output_index) {
    float sum = 0.0f;
    const size_t block_offset = static_cast<size_t>(output_index) * blocks_per_row;

    for (int input_index = 0; input_index < layout.in_features; ++input_index) {
      const Block& block =
          blocks[block_offset + static_cast<size_t>(input_index / BlockSize)];
      sum += input.data[static_cast<size_t>(input_index)] *
             decode(block, input_index % BlockSize);
    }
    output.data[static_cast<size_t>(output_index)] = sum;
  }
  return output;
}

float DecodeQ80(const BlockQ80& block, int index) {
  // Q8_0 使用一个块级缩放因子和有符号 8 位整数值
  return Fp16ToFloat(block.d) * static_cast<float>(block.qs[index]);
}

float DecodeQ40(const BlockQ40& block, int index) {
  // Q4_0 的低、高半块分别存放在每个字节的低、高四位中
  const uint8_t packed = block.qs[index % (kQ40BlockSize / 2)];
  const int value = (index < kQ40BlockSize / 2 ? packed & 0x0f : packed >> 4) - 8;
  return Fp16ToFloat(block.d) * static_cast<float>(value);
}

float DecodeQ41(const BlockQ41& block, int index) {
  // Q4_1 在无符号四位值缩放后额外加上块级最小值
  const uint8_t packed = block.qs[index % (kQ41BlockSize / 2)];
  const int value = index < kQ41BlockSize / 2 ? packed & 0x0f : packed >> 4;
  return Fp16ToFloat(block.d) * static_cast<float>(value) + Fp16ToFloat(block.m);
}

}  // namespace

std::vector<BlockQ80> QuantizeToQ80(const Tensor& src) {
  // 每行分别计算块缩放因子，使行尾不足一个块时也能独立补零
  if (src.size() == 0) {
    return {};
  }
  const RowLayout layout = GetRowLayout(src.shape, "QuantizeToQ80");
  const int blocks_per_row = (layout.row_size + kQ80BlockSize - 1) / kQ80BlockSize;
  std::vector<BlockQ80> blocks(ExpectedBlockCount<kQ80BlockSize>(layout));

  for (int row = 0; row < layout.row_count; ++row) {
    const size_t row_offset = static_cast<size_t>(row) * layout.row_size;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
      BlockQ80& block =
          blocks[static_cast<size_t>(row) * blocks_per_row + block_index];
      const int begin = block_index * kQ80BlockSize;
      const int end = std::min(begin + kQ80BlockSize, layout.row_size);
      float maximum = 0.0f;

      // Q8_0 用块内最大绝对值确定对称量化尺度
      for (int column = begin; column < end; ++column) {
        maximum = std::max(maximum, std::abs(src.data[row_offset + column]));
      }

      const float scale = maximum / 127.0f;
      block.d = FloatToFp16(scale);
      const float stored_scale = Fp16ToFloat(block.d);

      for (int column = begin; column < end; ++column) {
        const int quantized =
            stored_scale == 0.0f
                ? 0
                : std::clamp(static_cast<int>(std::round(
                                 src.data[row_offset + column] / stored_scale)),
                             -127, 127);
        block.qs[column - begin] = static_cast<int8_t>(quantized);
      }
    }
  }
  return blocks;
}

Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape) {
  // 复用通用块遍历，并按 Q8_0 规则解码每个元素
  return DequantizeBlocks<BlockQ80, kQ80BlockSize>(blocks, shape,
                                                   "DequantizeFromQ80", DecodeQ80);
}

std::vector<BlockQ40> QuantizeToQ40(const Tensor& src) {
  // Q4_0 每块 32 个值，由 16 个字节交错保存低、高半块
  if (src.size() == 0) {
    return {};
  }
  const RowLayout layout = GetRowLayout(src.shape, "QuantizeToQ40");
  const int blocks_per_row = (layout.row_size + kQ40BlockSize - 1) / kQ40BlockSize;
  std::vector<BlockQ40> blocks(ExpectedBlockCount<kQ40BlockSize>(layout));

  for (int row = 0; row < layout.row_count; ++row) {
    const size_t row_offset = static_cast<size_t>(row) * layout.row_size;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
      BlockQ40& block =
          blocks[static_cast<size_t>(row) * blocks_per_row + block_index];
      const int begin = block_index * kQ40BlockSize;
      const int end = std::min(begin + kQ40BlockSize, layout.row_size);
      float maximum = 0.0f;

      // 有符号 Q4_0 的有效整数范围为 [-8, 7]，正侧尺度取 7
      for (int column = begin; column < end; ++column) {
        maximum = std::max(maximum, std::abs(src.data[row_offset + column]));
      }

      block.d = FloatToFp16(maximum / 7.0f);
      const float scale = Fp16ToFloat(block.d);

      for (int index = 0; index < kQ40BlockSize / 2; ++index) {
        const int low_column = begin + index;
        const int high_column = low_column + kQ40BlockSize / 2;
        const auto quantize = [&](int column) {
          // 存储时加 8 转为四位无符号值，越界的行尾填充为零值编码
          if (column >= end || scale == 0.0f) {
            return 8;
          }
          return std::clamp(
              static_cast<int>(std::round(src.data[row_offset + column] / scale)) +
                  8,
              0, 15);
        };
        const int low = quantize(low_column);
        const int high = quantize(high_column);
        block.qs[index] = static_cast<uint8_t>(low | (high << 4));
      }
    }
  }
  return blocks;
}

// 复用通用块遍历，并按 Q4_0 的四位打包规则解码
Tensor DequantizeFromQ40(const std::vector<BlockQ40>& blocks,
                         const std::vector<int>& shape) {
  return DequantizeBlocks<BlockQ40, kQ40BlockSize>(blocks, shape,
                                                   "DequantizeFromQ40", DecodeQ40);
}

// Q4_1 仅需解码已有 GGUF 块，本项目当前不提供重新量化入口
Tensor DequantizeFromQ41(const std::vector<BlockQ41>& blocks,
                         const std::vector<int>& shape) {
  return DequantizeBlocks<BlockQ41, kQ41BlockSize>(blocks, shape,
                                                   "DequantizeFromQ41", DecodeQ41);
}

// 直接消费量化块，避免加载或计算时展开完整权重
Tensor LinearQ80(const Tensor& input, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearBlocks<BlockQ80, kQ80BlockSize>(input, weight, weight_shape,
                                               "LinearQ80", DecodeQ80);
}

Tensor LinearQ40(const Tensor& input, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
  // 直接消费量化块，避免加载或计算时展开完整权重
  return LinearBlocks<BlockQ40, kQ40BlockSize>(input, weight, weight_shape,
                                               "LinearQ40", DecodeQ40);
}

Tensor LinearQ41(const Tensor& input, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape) {
  // 直接消费量化块，避免加载或计算时展开完整权重
  return LinearBlocks<BlockQ41, kQ41BlockSize>(input, weight, weight_shape,
                                               "LinearQ41", DecodeQ41);
}

}  // namespace mini_llama
