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

#include "mini_llama/ops.h"
#include "mini_llama/thread_pool.h"

// 量化 SIMD 当前使用 ARM NEON，其余平台走 scalar 点积
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

#define MINI_LLAMA_USE_NEON 1
#endif

namespace mini_llama {
namespace {

// ----------------------------------------
// 块级缩放因子的 FP16 位编码与解码
// ----------------------------------------

// 将 F32 缩放因子编码为 GGUF 量化块使用的 IEEE FP16 位表示
uint16_t FloatToFp16(float value) {
  // 用 memcpy 读取位模式，避免通过不兼容类型的指针访问浮点对象
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const uint32_t sign = (bits >> 16) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;

  if (exponent <= 0) {
    // 小数值编码为 FP16 非正规数；超出其表示范围时保留符号并归零
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

  // 截去低 13 位前按舍入位进位，尾数溢出时需要同步调整指数
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

// 手动展开 FP16 位模式，避免依赖平台特定的半精度浮点类型
float Fp16ToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000U) << 16;
  const uint32_t exponent = (half >> 10) & 0x1fU;
  const uint32_t mantissa = half & 0x03ffU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      // FP16 非正规数等于 mantissa * 2^-24，再单独恢复符号
      const float value = std::ldexp(static_cast<float>(mantissa), -24);
      return sign == 0 ? value : -value;
    }
  } else if (exponent == 0x1fU) {
    // 指数全为 1 时保留无穷大或 NaN 的编码
    bits = sign | 0x7f800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }

  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// ----------------------------------------
// 按行量化布局、形状校验与通用反量化遍历
// ----------------------------------------

// 计算元素总数，拒绝空形状、非正维度及 size_t 乘法溢出
size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
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

// 最后一维作为行宽，其余维度折叠为行数；一维张量视为一行
struct RowLayout {
  int row_size;
  int row_count;
};

RowLayout GetRowLayout(const std::vector<int>& shape, const char* caller) {
  const size_t total = CheckedNumel(shape, caller);
  const int row_size = shape.back();
  const size_t row_count = total / static_cast<size_t>(row_size);
  if (row_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(caller) + ": row count overflow");
  }
  return {row_size, static_cast<int>(row_count)};
}

// 块不能跨行，每行独立向上取整；行尾不足一块也占用一个完整存储块
template <int BlockSize>
size_t ExpectedBlockCount(const RowLayout& layout) {
  const size_t blocks_per_row =
      static_cast<size_t>((layout.row_size + BlockSize - 1) / BlockSize);
  return static_cast<size_t>(layout.row_count) * blocks_per_row;
}

// 校验块数后按原始 shape 恢复 F32 张量，忽略各行尾块中的填充值
// decode 负责具体量化格式的单元素解码
template <typename Block, int BlockSize, typename Decode>
Tensor DequantizeBlocks(const std::vector<Block>& blocks,
                        const std::vector<int>& shape, const char* caller,
                        Decode decode) {
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

// ----------------------------------------
// 单 token 与批量 Linear 的布局与校验
// ----------------------------------------

struct LinearLayout {
  int in_features;
  int out_features;
  int rows;
  bool input_is_1d;
};

// 权重为 [out_features, in_features]，输入为 [in_features] 或 [batch, in_features]
// input_is_1d 用于保持输出秩，分别返回 [out_features] 或 [batch, out_features]
LinearLayout ValidateLinear(const Tensor& input,
                            const std::vector<int>& weight_shape,
                            const char* caller) {
  if (weight_shape.size() != 2 || weight_shape[0] <= 0 || weight_shape[1] <= 0) {
    throw std::runtime_error(std::string(caller) +
                             ": expected positive 2D weight shape");
  }
  int in_features = 0;
  int rows = 1;
  bool input_is_1d = false;
  if (input.num_dims() == 1) {
    in_features = input.shape[0];
    input_is_1d = true;
  } else if (input.num_dims() == 2) {
    in_features = input.shape[1];
    rows = input.shape[0];
    if (rows <= 0) {
      throw std::runtime_error(std::string(caller) +
                               ": batch size must be positive");
    }
  } else {
    throw std::runtime_error(
        std::string(caller) +
        ": expected input shape [in_features] or [batch, in_features]");
  }
  if (weight_shape[1] != in_features) {
    throw std::runtime_error(std::string(caller) + ": dimension mismatch");
  }
  return {in_features, weight_shape[0], rows, input_is_1d};
}

// ----------------------------------------
// NEON 块内点积：即时解码、缩放、乘加及 scalar 尾部
// ----------------------------------------

#if defined(MINI_LLAMA_USE_NEON)
// 支持 FMA 的 ARM 平台使用融合乘加，其他 NEON 平台保留普通乘加路径
float32x4_t MultiplyAddNeon(float32x4_t accumulator, float32x4_t input,
                            float32x4_t weight) {
#if defined(__aarch64__) || defined(__ARM_FEATURE_FMA)
  return vfmaq_f32(accumulator, input, weight);
#else
  return vmlaq_f32(accumulator, input, weight);
#endif
}

// 将四条 SIMD 累加通道水平求和为一个标量
float ReduceNeon(float32x4_t accumulator) {
  float32x2_t halves =
      vadd_f32(vget_low_f32(accumulator), vget_high_f32(accumulator));
  halves = vpadd_f32(halves, halves);
  return vget_lane_f32(halves, 0);
}

// 计算一个 Q8_0 块与对应 F32 输入片段的点积，count 是块内有效元素数
// 每轮加载 8 个 int8，依次扩展为 int16/int32，再转为两组 F32 并缩放
float DotBlockNeon(const float* input, const BlockQ80& block, int count) {
  const float scale = Fp16ToFloat(block.d);
  float32x4_t accumulator = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i <= count - 8; i += 8) {
    const int16x8_t values = vmovl_s8(vld1_s8(block.qs + i));
    const float32x4_t low =
        vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(values))), scale);
    const float32x4_t high =
        vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(values))), scale);
    accumulator = MultiplyAddNeon(accumulator, vld1q_f32(input + i), low);
    accumulator = MultiplyAddNeon(accumulator, vld1q_f32(input + i + 4), high);
  }
  float sum = ReduceNeon(accumulator);

  // 剩余不足 8 个元素时逐个计算，避免 SIMD 加载越过有效输入片段
  for (; i < count; ++i) {
    sum += input[i] * (scale * static_cast<float>(block.qs[i]));
  }
  return sum;
}

// 计算一个 Q4_0 块的点积：qs[j] 的低、高 nibble 对应第 j、j+16 个值
// 每轮处理 8 个值，步长不会跨越 16 元素半块边界
float DotBlockNeon(const float* input, const BlockQ40& block, int count) {
  const float scale = Fp16ToFloat(block.d);
  float32x4_t accumulator = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i <= count - 8; i += 8) {
    // 前 16 个值取低 nibble，后 16 个值取高 nibble
    const uint8x8_t packed = vld1_u8(block.qs + i % (kQ40BlockSize / 2));
    const uint8x8_t nibbles = i < kQ40BlockSize / 2
                                  ? vand_u8(packed, vdup_n_u8(0x0f))
                                  : vshr_n_u8(packed, 4);

    // nibble 的 [0, 15] 编码减 8 后恢复为 [-8, 7]，再有符号扩展并缩放
    const int16x8_t values =
        vmovl_s8(vsub_s8(vreinterpret_s8_u8(nibbles), vdup_n_s8(8)));
    const float32x4_t low =
        vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(values))), scale);
    const float32x4_t high =
        vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(values))), scale);
    accumulator = MultiplyAddNeon(accumulator, vld1q_f32(input + i), low);
    accumulator = MultiplyAddNeon(accumulator, vld1q_f32(input + i + 4), high);
  }
  float sum = ReduceNeon(accumulator);

  // 尾部仍按其所在半块选择 nibble，且只读取 count 个有效输入元素
  for (; i < count; ++i) {
    const uint8_t packed = block.qs[i % (kQ40BlockSize / 2)];
    const int value = (i < kQ40BlockSize / 2 ? packed & 0x0f : packed >> 4) - 8;
    sum += input[i] * (scale * static_cast<float>(value));
  }
  return sum;
}
#endif

// ----------------------------------------
// 量化 Linear 共用遍历与单元素解码
// ----------------------------------------

// 计算 input @ weight^T，在点积过程中解码权重，不分配完整 F32 权重矩阵
// UseNeon 为 true 时使用 Q8_0/Q4_0 块点积；false 时通过 decode 逐元素计算
template <typename Block, int BlockSize, bool UseNeon, typename Decode>
Tensor LinearBlocks(const Tensor& input, const std::vector<Block>& blocks,
                    const std::vector<int>& weight_shape, const char* caller,
                    Decode decode) {
  const LinearLayout layout = ValidateLinear(input, weight_shape, caller);
  // ParallelFor 使用 int 任务数，在乘法及输出分配前检查 batch * out_features。
  if (layout.rows > std::numeric_limits<int>::max() / layout.out_features) {
    throw std::runtime_error(std::string(caller) + ": output task count overflow");
  }
  const int task_count = layout.rows * layout.out_features;
  const int blocks_per_row = (layout.in_features + BlockSize - 1) / BlockSize;
  const size_t expected_blocks =
      static_cast<size_t>(layout.out_features) * blocks_per_row;
  if (blocks.size() != expected_blocks) {
    throw std::runtime_error(std::string(caller) + ": block count mismatch");
  }

  Tensor output(layout.input_is_1d
                    ? std::vector<int>{layout.out_features}
                    : std::vector<int>{layout.rows, layout.out_features},
                0.0f);

  // 将输入行与输出通道展平为任务；输入和量化块只读，每个任务写一个输出元素。
  // 各线程独立完成完整点积，不创建内层线程。
  // 小任务及单线程配置由 ParallelFor 在调用线程执行
  ParallelFor(task_count, [&](int begin, int end) {
    for (int flat_index = begin; flat_index < end; ++flat_index) {
      const int row = flat_index / layout.out_features;
      const int output_index = flat_index % layout.out_features;
      const float* input_row =
          input.data.data() + static_cast<size_t>(row) * layout.in_features;
      float sum = 0.0f;
      const size_t block_offset = static_cast<size_t>(output_index) * blocks_per_row;

#if defined(MINI_LLAMA_USE_NEON)
      if constexpr (UseNeon) {
        // 最后一块仅传入剩余的有效长度，填充值不参与点积
        for (int input_index = 0; input_index < layout.in_features;
             input_index += BlockSize) {
          const Block& block =
              blocks[block_offset + static_cast<size_t>(input_index / BlockSize)];
          sum += DotBlockNeon(input_row + input_index, block,
                              std::min(BlockSize, layout.in_features - input_index));
        }
      } else
#endif
      {
        for (int input_index = 0; input_index < layout.in_features; ++input_index) {
          const Block& block =
              blocks[block_offset + static_cast<size_t>(input_index / BlockSize)];
          sum += input_row[input_index] * decode(block, input_index % BlockSize);
        }
      }
      output.data[static_cast<size_t>(flat_index)] = sum;
    }
  });
  return output;
}

// Q8_0 单元素解码：F32 权重 = FP16 缩放因子 * 有符号 int8 值
float DecodeQ80(const BlockQ80& block, int index) {
  return Fp16ToFloat(block.d) * static_cast<float>(block.qs[index]);
}

// Q4_0 单元素解码：前 16 个值取低 nibble，后 16 个值取高 nibble，再减 8
float DecodeQ40(const BlockQ40& block, int index) {
  const uint8_t packed = block.qs[index % (kQ40BlockSize / 2)];
  const int value = (index < kQ40BlockSize / 2 ? packed & 0x0f : packed >> 4) - 8;
  return Fp16ToFloat(block.d) * static_cast<float>(value);
}

// Q4_1 使用同样的半块布局，解码公式为 scale * 无符号 nibble + minimum
float DecodeQ41(const BlockQ41& block, int index) {
  const uint8_t packed = block.qs[index % (kQ41BlockSize / 2)];
  const int value = index < kQ41BlockSize / 2 ? packed & 0x0f : packed >> 4;
  return Fp16ToFloat(block.d) * static_cast<float>(value) + Fp16ToFloat(block.m);
}

}  // namespace

// ----------------------------------------
// Q8_0 量化与反量化
// ----------------------------------------

// 每行独立分成 32 元素块，以块内最大绝对值 / 127 确定缩放因子
std::vector<BlockQ80> QuantizeToQ80(const Tensor& src) {
  if (src.size() == 0) {
    return {};
  }
  const RowLayout layout = GetRowLayout(src.shape, "QuantizeToQ80");
  const int blocks_per_row = (layout.row_size + kQ80BlockSize - 1) / kQ80BlockSize;

  // 值初始化保留行尾不足一块时未写入的 int8 零值编码
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

      // 用实际存储的 FP16 scale 量化，使编码与后续解码采用相同尺度
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

// 恢复原始 shape 的有效元素，行尾填充不进入输出张量
Tensor DequantizeFromQ80(const std::vector<BlockQ80>& blocks,
                         const std::vector<int>& shape) {
  return DequantizeBlocks<BlockQ80, kQ80BlockSize>(blocks, shape,
                                                   "DequantizeFromQ80", DecodeQ80);
}

// ----------------------------------------
// Q4_0 量化与反量化，以及已有 Q4_1 块的反量化
// ----------------------------------------

// 每块 32 个值保存为 16 字节，低 nibble 存前半块，高 nibble 存后半块
std::vector<BlockQ40> QuantizeToQ40(const Tensor& src) {
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

      // 此量化器采用最大绝对值 / 7 作为 scale，对应整数范围 [-8, 7] 的正端
      for (int column = begin; column < end; ++column) {
        maximum = std::max(maximum, std::abs(src.data[row_offset + column]));
      }

      block.d = FloatToFp16(maximum / 7.0f);

      // 与 Q8_0 一样，量化使用实际存储的 FP16 scale
      const float scale = Fp16ToFloat(block.d);

      for (int index = 0; index < kQ40BlockSize / 2; ++index) {
        const int low_column = begin + index;
        const int high_column = low_column + kQ40BlockSize / 2;
        const auto quantize = [&](int column) {
          // 先舍入有符号值，再加 8 编码；行尾或零 scale 用 nibble 8 表示零
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

// 按 Q4_0 半块布局恢复原始 shape，忽略行尾填充
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

// ----------------------------------------
// 公开量化 Linear 接口及编译期平台分支
// ----------------------------------------

// NEON 与 scalar 分支保持相同签名、shape 校验和外层并行调度
// 编译时只保留其中一个 LinearQ80 定义，二者均直接消费量化块
#if defined(MINI_LLAMA_USE_NEON)
Tensor LinearQ80(const Tensor& input, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearBlocks<BlockQ80, kQ80BlockSize, true>(input, weight, weight_shape,
                                                     "LinearQ80", DecodeQ80);
}
#else
Tensor LinearQ80(const Tensor& input, const std::vector<BlockQ80>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearBlocks<BlockQ80, kQ80BlockSize, false>(input, weight, weight_shape,
                                                      "LinearQ80", DecodeQ80);
}
#endif

// Q4_0 沿用相同分发方式，NEON 分支在块内拆解 nibble 后计算点积
#if defined(MINI_LLAMA_USE_NEON)
Tensor LinearQ40(const Tensor& input, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearBlocks<BlockQ40, kQ40BlockSize, true>(input, weight, weight_shape,
                                                     "LinearQ40", DecodeQ40);
}
#else
Tensor LinearQ40(const Tensor& input, const std::vector<BlockQ40>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearBlocks<BlockQ40, kQ40BlockSize, false>(input, weight, weight_shape,
                                                      "LinearQ40", DecodeQ40);
}
#endif

// Q4_1 当前使用 scalar 即时解码，输入行与输出通道仍通过共用遍历并行计算
Tensor LinearQ41(const Tensor& input, const std::vector<BlockQ41>& weight,
                 const std::vector<int>& weight_shape) {
  return LinearBlocks<BlockQ41, kQ41BlockSize, false>(input, weight, weight_shape,
                                                      "LinearQ41", DecodeQ41);
}

// ----------------------------------------
// 兼容 Matmul 接口：完整反量化后调用通用 F32 Matmul
// ----------------------------------------

// 权重为 [m, k]，输入为 [k, n]，输出为 [m, n]
// 此兼容路径会分配完整 F32 权重张量，量化 Linear 则在点积中即时解码
Tensor MatmulQ80(const std::vector<BlockQ80>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
  return Matmul(DequantizeFromQ80(weight, weight_shape), input);
}

// Q4_0 兼容 Matmul 使用相同的矩阵方向和完整反量化流程
Tensor MatmulQ40(const std::vector<BlockQ40>& weight, const Tensor& input,
                 const std::vector<int>& weight_shape) {
  return Matmul(DequantizeFromQ40(weight, weight_shape), input);
}

// ----------------------------------------
// 量化误差比较辅助函数
// ----------------------------------------

// 比较同一权重、输入下的 F32 Matmul 与 Q8_0 兼容 Matmul，返回最大绝对误差
float CompareMatmulError(const Tensor& weight, const Tensor& input) {
  const Tensor reference = Matmul(weight, input);
  const auto blocks = QuantizeToQ80(weight);
  const Tensor quantized = MatmulQ80(blocks, input, weight.shape);
  if (reference.shape != quantized.shape) {
    throw std::runtime_error("CompareMatmulError: shape mismatch");
  }

  float maximum_error = 0.0f;
  for (size_t i = 0; i < reference.size(); ++i) {
    maximum_error =
        std::max(maximum_error, std::abs(reference.data[i] - quantized.data[i]));
  }
  return maximum_error;
}

// 比较 F32 Linear 与 Q4_0 即时解码 Linear，输入支持 [k] 或 [1, k]
// 返回输出元素的最大绝对误差，不是权重 roundtrip 误差
float CompareQ40Error(const Tensor& weight, const Tensor& input) {
  const Tensor reference = Linear(input, weight);
  const auto blocks = QuantizeToQ40(weight);
  const Tensor quantized = LinearQ40(input, blocks, weight.shape);
  if (reference.shape != quantized.shape) {
    throw std::runtime_error("CompareQ40Error: shape mismatch");
  }

  float maximum_error = 0.0f;
  for (size_t i = 0; i < reference.size(); ++i) {
    maximum_error =
        std::max(maximum_error, std::abs(reference.data[i] - quantized.data[i]));
  }
  return maximum_error;
}

}  // namespace mini_llama
