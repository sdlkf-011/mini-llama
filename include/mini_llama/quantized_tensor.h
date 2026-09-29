#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_llama/tensor.h"

namespace mini_llama {

// -------------------------------------------------
// 量化类型
// -------------------------------------------------
enum class QuantType {
  kF32,
  kQ80,
  kQ40,
  kQ41,
};

// -------------------------------------------------
// Q8_0 块布局（兼容 ggml / llama.cpp）
// 每个块覆盖 32 个元素
// -------------------------------------------------
constexpr int kQ80BlockSize = 32;

struct BlockQ80 {
  uint16_t d;     // FP16 增量（缩放因子）
  int8_t qs[32];  // 量化值
};

static_assert(sizeof(BlockQ80) == sizeof(uint16_t) + kQ80BlockSize,
              "错误的 Q8_0 块大小");

// -------------------------------------------------
// Q4_0 块布局（兼容 ggml / llama.cpp）
// 每个块覆盖 32 个元素，打包进 16 字节
// -------------------------------------------------
constexpr int kQ40BlockSize = 32;

struct BlockQ40 {
  uint16_t d;      // FP16 增量（缩放因子）
  uint8_t qs[16];  // 32 个 4 位值，打包存储（每字节 2 个）
};

static_assert(sizeof(BlockQ40) == sizeof(uint16_t) + kQ40BlockSize / 2,
              "错误的 Q4_0 块大小");

// -------------------------------------------------
// Q4_1 块布局（兼容 ggml / llama.cpp）
// 每个块覆盖 32 个元素，打包进 16 字节。
// 值公式：x = d * q + m，q 在 [0, 15] 范围内
// -------------------------------------------------
constexpr int kQ41BlockSize = 32;

struct BlockQ41 {
  uint16_t d;      // FP16 增量（缩放因子）
  uint16_t m;      // FP16 最小值/偏移量
  uint8_t qs[16];  // 32 个 4 位值，打包存储（每字节 2 个）
};

static_assert(sizeof(BlockQ41) == sizeof(uint16_t) * 2 + kQ41BlockSize / 2,
              "错误的 Q4_1 块大小");

// -------------------------------------------------
// QuantizedTensor：保存量化数据或 F32 数据的带标签结构体
// 为教学清晰起见，使用类似带标签联合体的布局，而不是 std::variant
// 根据 `type`，只有 {f32_data, q8_0_data, q4_0_data, q4_1_data} 中
// 的一个成员是有效的
// -------------------------------------------------
struct QuantizedTensor {
  QuantType type = QuantType::kF32;
  std::vector<int> shape;

  // 当 type == F32 时有效
  std::vector<float> f32_data;

  // 当 type == Q8_0 时有效
  std::vector<BlockQ80> q8_0_data;

  // 当 type == Q4_0 时有效
  std::vector<BlockQ40> q4_0_data;

  // 当 type == Q4_1 时有效
  std::vector<BlockQ41> q4_1_data;

  // -------------------------------------------------
  // 辅助函数
  // -------------------------------------------------
  size_t num_elements() const {
    size_t n = 1;
    for (int d : shape) {
      n *= static_cast<size_t>(d);
    }
    return n;
  }

  std::string ShapeStringShort() const {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      s += std::to_string(shape[i]);
      if (i + 1 < shape.size()) {
        s += ", ";
      }
    }
    s += "]";
    return s;
  }

  void AssertShape(const std::vector<int>& expected, const char* caller) const {
    if (shape != expected) {
      throw std::runtime_error(
          std::string(caller) + ": 形状不匹配，期望 " +
          QuantizedTensor{QuantType::kF32, expected}.ShapeStringShort() +
          "，实际为 " + ShapeStringShort());
    }
  }

  int num_dims() const { return static_cast<int>(shape.size()); }
};

// 将 QuantizedTensor（type==F32）转换为普通 Tensor
Tensor ToTensor(const QuantizedTensor& q);

// 将普通 Tensor 转换为 QuantizedTensor（type==F32）
QuantizedTensor ToQuantizedTensor(const Tensor& t);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_QUANTIZED_TENSOR_H_
