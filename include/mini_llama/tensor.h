#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_TENSOR_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_TENSOR_H_

#include <cstddef>
#include <string>
#include <vector>

namespace mini_llama {

// 用于 F32 的数据最小稠密张量
struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;

  Tensor() = default;
  explicit Tensor(const std::vector<int>& input_shape, float fill = 0.0f);

  // 元素总数
  size_t size() const { return data.size(); }
  size_t num_elements() const { return data.size(); }

  // 维度数量
  int num_dims() const { return static_cast<int>(shape.size()); }

  // 根据多维索引计算展开后的 FlatIndex
  size_t FlatIndex(const std::vector<int>& indices) const;

  // 获取多索引处的元素
  float& At(const std::vector<int>& indices);
  float At(const std::vector<int>& indices) const;

  // 便捷的一维/二维/三维/四维访问器
  float& At1(int i);
  float At1(int i) const;
  float& At2(int i, int j);
  float At2(int i, int j) const;
  float& At3(int i, int j, int k);
  float At3(int i, int j, int k) const;
  float& At4(int i, int j, int k, int l);
  float At4(int i, int j, int k, int l) const;

  // 获取二维张量中某一行的指针
  float* RowPtr(int row);
  const float* RowPtr(int row) const;

  // 校验 shape 是否与预期一致:不匹配时抛出异常
  void AssertShape(const std::vector<int>& expected, const char* caller) const;

  // 在确认元素数量保持不变后重塑 shape
  Tensor ReshapeChecked(const std::vector<int>& new_shape, const char* caller) const;

  // 一维访问辅助
  float& operator[](size_t i) { return data[i]; }
  float operator[](size_t i) const { return data[i]; }

  // 打印 shape,并可选择打印数据
  std::string ShapeStringShort() const;
  std::string ShapeString() const { return ShapeStringShort(); }
  void Print(const std::string& name = "", bool print_data = false) const;
};

// 创建一个一维张量
Tensor MakeTensor1D(int d0, float fill = 0.0f);

// 创建一个二维张量
Tensor MakeTensor2D(int d0, int d1, float fill = 0.0f);

// 创建一个三维张量
Tensor MakeTensor3D(int d0, int d1, int d2, float fill = 0.0f);

// 创建一个四维张量
Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill = 0.0f);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAmA_TENSOR_H_
