#include "mini_llama/tensor.h"

#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {

namespace {

// 获取调用者名称
const char* CallerName(const char* caller) {
  return caller == nullptr ? "Tensor" : caller;
}

// 把{2, 3, 4}格式化为"[2, 3, 4]"，用于异常信息和调试输出。注意空 shape 应输出"[]"
std::string ShapeToString(const std::vector<int>& shape) {
  std::string s = "[";

  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      s += ", ";
    }
    s += std::to_string(shape[i]);
  }
  s += "]";

  return s;
}

// 计算 shape 的元素总量，拒绝非法的维度
// shape:[123, 23], shape_num = 123 x 23
size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  size_t total = 1;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    const int dim = shape[axis];  // 取出当前元素

    if (dim <= 0) {
      throw std::runtime_error(std::string(CallerName(caller)) +
                               ": dimension at axis " + std::to_string(axis) +
                               " must be positive, got " + std::to_string(dim) +
                               " in shape " + ShapeToString(shape));
    }

    // 判断是否会溢出
    const size_t dim_size = static_cast<size_t>(dim);
    if (total > std::numeric_limits<size_t>::max() / dim_size) {
      throw std::runtime_error(std::string(CallerName(caller)) +
                               "shape element count overflow for " +
                               ShapeToString(shape));
    }

    total *= dim_size;
  }
  return total;
}

// 检查维度数量是否合法
void CheckRank(const Tensor& tensor, int expected, const char* caller) {
  if (tensor.num_dims() != expected) {
    throw std::runtime_error(std::string(caller) + " expected " +
                             std::to_string(expected) + "D tensor, got " +
                             tensor.ShapeStringShort());
  }
}

// 检查访问某个维度的某个元素是否合法
void CheckAxisIndex(const Tensor& tensor, int axis, int float_index,
                    const char* caller) {
  const int dim = tensor.shape[axis];
  if (float_index < 0 || float_index >= dim) {
    throw std::out_of_range(
        std::string(caller) + ": float_index " + std::to_string(float_index) +
        " out of range on axis " + std::to_string(axis) + " with size " +
        std::to_string(dim) + " in tensor " + tensor.ShapeStringShort());
  }
}

}  // namespace

Tensor::Tensor(const std::vector<int>& input_shape, float fill)
    : shape(input_shape) {
  // 计算并校验维度数量
  const size_t total = CheckedNumel(input_shape, "Tensor constructor");
  data.resize(total, fill);  // 预分配
}

size_t Tensor::FlatIndex(const std::vector<int>& indices) const {
  // 检查坐标是否正确
  if (indices.size() != shape.size()) {
    throw std::runtime_error("Tensor::FlatIndex expected " +
                             std::to_string(shape.size()) + " indices for tensor " +
                             ShapeStringShort() + ", got " +
                             std::to_string(indices.size()));
  }

  if (shape.empty()) {
    return 0;
  }

  // 检查值是否越界
  for (size_t axis = 0; axis < indices.size(); ++axis) {
    CheckAxisIndex(*this, static_cast<int>(axis), indices[axis],
                   "Tensor::FlatIndex");
  }

  // 多维索引的降维计算
  size_t flat = 0;
  size_t stride = 1;  // 步长(最后一维度)
  for (int i = static_cast<int>(shape.size() - 1); i >= 0; --i) {
    flat += static_cast<int>(indices[i]) * stride;
    stride *= static_cast<int>(shape[i]);
  }
  return flat;
}

float& Tensor::At(const std::vector<int>& indices) {
  return data[FlatIndex(indices)];
}

float Tensor::At(const std::vector<int>& indices) const {
  return data[FlatIndex(indices)];
}

float& Tensor::At1(int i) {
  CheckRank(*this, 1, "At1");
  CheckAxisIndex(*this, 0, i, "At1");
  return data[i];
}

float Tensor::At1(int i) const {
  CheckRank(*this, 1, "At1");
  CheckAxisIndex(*this, 0, i, "At1");
  return data[i];
}

float& Tensor::At2(int i, int j) {
  CheckRank(*this, 2, "At2");
  CheckAxisIndex(*this, 0, i, "At2");
  CheckAxisIndex(*this, 1, j, "At2");
  return data[(i * shape[1]) + j];
}

float Tensor::At2(int i, int j) const {
  CheckRank(*this, 2, "At2");
  CheckAxisIndex(*this, 0, i, "At2");
  CheckAxisIndex(*this, 1, j, "At2");
  return data[(i * shape[1]) + j];
}

float& Tensor::At3(int i, int j, int k) {
  CheckRank(*this, 3, "At3");
  CheckAxisIndex(*this, 0, i, "At3");
  CheckAxisIndex(*this, 1, j, "At3");
  CheckAxisIndex(*this, 2, k, "At3");
  return data[((i * shape[1]) + j) * shape[2] + k];
}

float Tensor::At3(int i, int j, int k) const {
  CheckRank(*this, 3, "At3");
  CheckAxisIndex(*this, 0, i, "At3");
  CheckAxisIndex(*this, 1, j, "At3");
  CheckAxisIndex(*this, 2, k, "At3");
  return data[((i * shape[1]) + j) * shape[2] + k];
}

float& Tensor::At4(int i, int j, int k, int l) {
  CheckRank(*this, 4, "At4");
  CheckAxisIndex(*this, 0, i, "At4");
  CheckAxisIndex(*this, 1, j, "At4");
  CheckAxisIndex(*this, 2, k, "At4");
  CheckAxisIndex(*this, 3, l, "At4");
  return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

float Tensor::At4(int i, int j, int k, int l) const {
  CheckRank(*this, 4, "At4");
  CheckAxisIndex(*this, 0, i, "At4");
  CheckAxisIndex(*this, 1, j, "At4");
  CheckAxisIndex(*this, 2, k, "At4");
  CheckAxisIndex(*this, 3, l, "At4");
  return data[((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
}

float* Tensor::RowPtr(int row) {
  CheckRank(*this, 2, "row_ptr");
  CheckAxisIndex(*this, 0, row, "row_ptr");
  return data.data() + row * shape[1];
}

const float* Tensor::RowPtr(int row) const {
  CheckRank(*this, 2, "row_ptr");
  CheckAxisIndex(*this, 0, row, "row_ptr");
  return data.data() + row * shape[1];
}

void Tensor::AssertShape(const std::vector<int>& expected,
                         const char* caller) const {
  if (shape != expected) {
    throw std::runtime_error(
        std::string(CallerName(caller)) + ": shape mismatch. expected " +
        ShapeToString(expected) + ", got " + ShapeStringShort());
  }
}

Tensor Tensor::ReshapeChecked(const std::vector<int>& new_shape,
                              const char* caller) const {
  size_t new_total = CheckedNumel(new_shape, CallerName(caller));
  if (new_total != data.size()) {
    throw std::runtime_error(
        std::string(CallerName(caller)) + ": cannot reshape tensor with " +
        std::to_string(data.size()) + " elements into shape with " +
        std::to_string(new_total) + " elements");
  }

  Tensor result = *this;
  result.shape = new_shape;

  return result;
}

std::string Tensor::ShapeStringShort() const { return ShapeToString(shape); }

void Tensor::Print(const std::string& name, bool print_data) const {
  if (!name.empty()) {
    std::cout << name << " ";
  }

  std::cout << "shape=" << ShapeToString(shape) << " size=" << size() << "\n";
  if (print_data) {
    for (size_t i = 0; i < data.size(); ++i) {
      std::cout << std::fixed << std::setprecision(6) << data[i];
      if (i + 1 < data.size()) {
        std::cout << " ";
      }
    }
    std::cout << "\n";
  }
}

Tensor MakeTensor1D(int d0, float fill) { return Tensor({d0}, fill); }

Tensor MakeTensor2D(int d0, int d1, float fill) { return Tensor({d0, d1}, fill); }

Tensor MakeTensor3D(int d0, int d1, int d2, float fill) {
  return Tensor({d0, d1, d2}, fill);
}

Tensor MakeTensor4D(int d0, int d1, int d2, int d3, float fill) {
  return Tensor({d0, d1, d2, d3}, fill);
}

}  // namespace mini_llama
