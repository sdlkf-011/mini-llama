#include "mini_llama/cuda_matmul.h"

#include <stdexcept>

// CPU 构建不依赖 CUDA SDK；各计算入口在未启用 CUDA 时抛出统一异常
#ifdef MINI_LLAMA_USE_CUDA
#include <cublas_v2.h>
#endif

namespace mini_llama {

namespace {

std::runtime_error CudaMatmulNotBuiltError() {
  return std::runtime_error(
      "CUDA Matmul was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a "
      "NVIDIA CUDA machine.");
}

#ifdef MINI_LLAMA_USE_CUDA
// Tensor 的 shape/data 可被调用方修改，不能只依赖构造时的校验
// 拒绝非正维度，并检查元素数和 F32 字节数溢出，避免错误的显存分配长度
size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
  size_t count = 1;
  for (int dim : shape) {
    if (dim <= 0 ||
        count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      throw std::runtime_error(std::string(caller) + ": invalid tensor dimensions");
    }
    count *= static_cast<size_t>(dim);
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    throw std::runtime_error(std::string(caller) + ": tensor byte count overflow");
  }
  return count;
}

// 同时核对逻辑形状与实际主机存储，避免 GEMM 按 shape 越界读取设备输入
void ValidateStorage(const Tensor& tensor, const char* caller) {
  if (tensor.size() != CheckedNumel(tensor.shape, caller)) {
    throw std::runtime_error(std::string(caller) + ": tensor storage size mismatch");
  }
}

// bias 为 nullptr 表示不加偏置；非空时在 GPU 分配和计算前检查形状与存储
void ValidateBias(const Tensor* bias, int out_features) {
  if (bias == nullptr) {
    return;
  }
  bias->AssertShape({out_features}, "CudaLinear bias");
  ValidateStorage(*bias, "CudaLinear bias");
}

const char* CublasStatusName(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "CUBLAS_STATUS_UNKNOWN";
}

// 将 cuBLAS 状态转换为 C++ 异常，保留 API 名称与状态以便定位失败位置
void CheckCublas(cublasStatus_t status, const char* expr) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("cuBLAS error in " + std::string(expr) + ": " +
                             CublasStatusName(status));
  }
}

// 每次计算独占一个句柄，创建前先选择设备
// 构造成功后，无论正常返回还是后续计算抛异常，均通过析构释放句柄
class CublasHandle {
 public:
  explicit CublasHandle(int device_id) {
    CudaSetDevice(device_id);
    CheckCublas(cublasCreate(&handle_), "cublasCreate");
  }

  ~CublasHandle() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_ = nullptr;
};

void ValidateMatmulShapes(const Tensor& a, const Tensor& b) {
  if (a.num_dims() != 2 || b.num_dims() != 2) {
    throw std::runtime_error("CudaMatmul: expected a and b to be 2D tensors");
  }
  if (a.shape[1] != b.shape[0]) {
    throw std::runtime_error("CudaMatmul: dimension mismatch " +
                             a.ShapeStringShort() + " vs " + b.ShapeStringShort());
  }
  ValidateStorage(a, "CudaMatmul a");
  ValidateStorage(b, "CudaMatmul b");
}

// 在已下载的 CPU 结果上逐行累加 bias[col]，实现沿 batch 维广播
void AddBiasInPlace(Tensor& y, const Tensor& bias, int rows, int cols) {
  if (bias.num_dims() != 1 || bias.shape[0] != cols) {
    throw std::runtime_error("CudaLinear: expected bias shape [out_features]");
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      y.data[static_cast<size_t>(row) * cols + col] += bias.data[col];
    }
  }
}

int LinearInFeaturesFromShape(const std::vector<int>& x_shape,
                              const std::string& x_shape_str, const char* caller) {
  if (x_shape.size() == 1) {
    return x_shape[0];
  }
  if (x_shape.size() == 2) {
    return x_shape[1];
  }
  throw std::runtime_error(
      std::string(caller) +
      ": expected x shape [in_features] or [batch, in_features], got " +
      x_shape_str);
}

// CPU/GPU 输入共用形状约定：X 为 [in] 或 [batch,in]，W 为 [out,in]
void ValidateLinearShape(const std::vector<int>& x_shape,
                         const std::string& x_shape_str,
                         const std::vector<int>& w_shape, const char* caller) {
  if (w_shape.size() != 2) {
    throw std::runtime_error(std::string(caller) +
                             ": expected weight shape [out_features, in_features]");
  }
  int in_features = LinearInFeaturesFromShape(x_shape, x_shape_str, caller);
  if (w_shape[1] != in_features) {
    throw std::runtime_error(
        std::string(caller) + ": dimension mismatch x=" + x_shape_str + " weight=[" +
        std::to_string(w_shape[0]) + ", " + std::to_string(w_shape[1]) + "]");
  }
  CheckedNumel(x_shape, caller);
  CheckedNumel(w_shape, caller);
}

void ValidateLinearInputs(const Tensor& x, const std::vector<int>& w_shape) {
  ValidateLinearShape(x.shape, x.ShapeStringShort(), w_shape, "CudaLinear");
  ValidateStorage(x, "CudaLinear x");
}

// 借用设备权重，只管理输入和输出的临时显存；权重的设备与生命周期由调用方保证
// 每次上传一次 CPU 输入、下载一次结果；可选 bias 随后在 CPU 上应用
Tensor CudaLinearWithDevicePointer(const Tensor& x, const void* w_device,
                                   const std::vector<int>& w_shape,
                                   const Tensor* bias, int device_id) {
  if (w_device == nullptr) {
    throw std::runtime_error("CudaLinear: device weight pointer is null");
  }
  ValidateLinearInputs(x, w_shape);
  ValidateBias(bias, w_shape[0]);

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape[0];
  int in_features = input_is_1d ? x.shape[0] : x.shape[1];
  int out_features = w_shape[0];

  // 向量按单行参与 GEMM，但返回时保留向量/批量输入对应的输出秩
  Tensor y(input_is_1d ? std::vector<int>{out_features}
                       : std::vector<int>{rows, out_features},
           0.0f);
  CudaDeviceBuffer x_dev(x.size() * sizeof(float), device_id);
  CudaDeviceBuffer y_dev(y.size() * sizeof(float), device_id);
  x_dev.Upload(x.data.data(), x.size() * sizeof(float));

  CublasHandle handle(device_id);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // 行主序 Y[R,O] = X[R,I] * W[O,I]^T，等价于列主序 Y^T[O,R] = W * X^T
  // W 的字节布局对应列主序 W^T[I,O]，使用 OP_T 得到 W[O,I]；物理步长 lda=I
  // X 的字节布局对应列主序 X^T[I,R]，使用 OP_N，ldb=I
  // 因此 GEMM 的 (m,n,k)=(O,R,I)，ldc=O；alpha=1、beta=0，不读取原输出
  // Y^T(o,r) 的偏移 o+r*O 等于行主序 Y(r,o)，结果无需物理转置
  CheckCublas(
      cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows,
                  in_features, &alpha, static_cast<const float*>(w_device),
                  in_features, static_cast<const float*>(x_dev.data()), in_features,
                  &beta, static_cast<float*>(y_dev.data()), out_features),
      "cublasSgemm(CudaLinear)");

  y_dev.Download(y.data.data(), y.size() * sizeof(float));
  if (bias != nullptr) {
    AddBiasInPlace(y, *bias, rows, out_features);
  }
  return y;
}

// 全设备路径：直接读取 X/W 的设备指针，只分配设备输出，不上传输入或下载结果
CudaTensor CudaLinearDeviceInputWithDevicePointer(const CudaTensor& x,
                                                  const void* w_device,
                                                  const std::vector<int>& w_shape,
                                                  int device_id) {
  if (w_device == nullptr) {
    throw std::runtime_error("CudaLinearDeviceInput: device weight pointer is null");
  }

  // CudaTensor 记录输入设备；裸权重指针的同设备约束仍由调用方保证
  if (x.device_id() != device_id) {
    throw std::runtime_error(
        "CudaLinearDeviceInput: input tensor is on a different CUDA device");
  }
  ValidateLinearShape(x.shape(), x.ShapeStringShort(), w_shape,
                      "CudaLinearDeviceInput");

  bool input_is_1d = x.num_dims() == 1;
  int rows = input_is_1d ? 1 : x.shape()[0];
  int in_features = input_is_1d ? x.shape()[0] : x.shape()[1];
  int out_features = w_shape[0];

  CudaTensor y(input_is_1d ? std::vector<int>{out_features}
                           : std::vector<int>{rows, out_features},
               device_id);

  CublasHandle handle(device_id);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // 布局推导与上面的 Linear 相同：权重 OP_T、输入 OP_N，(m,n,k)=(out,rows,in)
  // lda/ldb 保持原存储步长 in，ldc=out；输出由返回的 CudaTensor 独占管理
  CheckCublas(
      cublasSgemm(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, out_features, rows,
                  in_features, &alpha, static_cast<const float*>(w_device),
                  in_features, static_cast<const float*>(x.data()), in_features,
                  &beta, static_cast<float*>(y.data()), out_features),
      "cublasSgemm(CudaLinearDeviceInput)");

  return y;
}
#endif

}  // namespace

// 仅反映构建配置，不查询驱动或 GPU；实际设备错误由运行时操作报告
bool CudaMatmulBuilt() {
#ifdef MINI_LLAMA_USE_CUDA
  return true;
#else
  return false;
#endif
}

Tensor CudaMatmul(const Tensor& a, const Tensor& b, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
  ValidateMatmulShapes(a, b);
  const int m = a.shape[0];
  const int k = a.shape[1];
  const int n = b.shape[1];
  Tensor c({m, n});
  CudaDeviceBuffer a_dev(a.size() * sizeof(float), device_id);
  CudaDeviceBuffer b_dev(b.size() * sizeof(float), device_id);
  CudaDeviceBuffer c_dev(c.size() * sizeof(float), device_id);
  a_dev.Upload(a.data.data(), a_dev.bytes());
  b_dev.Upload(b.data.data(), b_dev.bytes());
  CublasHandle handle(device_id);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // 行主序 C[M,N] = A[M,K] * B[K,N]，等价于列主序 C^T[N,M] = B^T[N,K] * A^T[K,M]
  // A/B 的原始字节已对应各自转置的列主序布局，因此交换指针顺序，均使用 OP_N
  // GEMM 的 (m,n,k)=(N,M,K)，三个物理列步长分别为 lda=N、ldb=K、ldc=N
  // C^T(j,i) 的偏移 j+i*N 等于行主序 C(i,j)；下载后可直接返回 [M,N]
  CheckCublas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                          static_cast<const float*>(b_dev.data()), n,
                          static_cast<const float*>(a_dev.data()), k, &beta,
                          static_cast<float*>(c_dev.data()), n),
              "cublasSgemm(CudaMatmul)");
  c_dev.Download(c.data.data(), c_dev.bytes());
  return c;
#else
  (void)a;
  (void)b;
  (void)device_id;
  throw CudaMatmulNotBuiltError();
#endif
}

Tensor CudaLinear(const Tensor& x, const Tensor& weight, const Tensor* bias,
                  int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
  ValidateLinearInputs(x, weight.shape);
  ValidateStorage(weight, "CudaLinear weight");
  ValidateBias(bias, weight.shape[0]);

  // CPU 权重接口每次上传 W，局部 Buffer 的生命周期覆盖整个计算并在返回时释放
  // 复用设备指针路径上传 X、计算并下载 Y；Decode 应使用常驻权重接口
  CudaDeviceBuffer weight_dev(weight.size() * sizeof(float), device_id);
  weight_dev.Upload(weight.data.data(), weight_dev.bytes());
  return CudaLinearWithDevicePointer(x, weight_dev.data(), weight.shape, bias,
                                     device_id);
#else
  (void)x;
  (void)weight;
  (void)bias;
  (void)device_id;
  throw CudaMatmulNotBuiltError();
#endif
}

Tensor CudaLinearDeviceWeight(const Tensor& x, const void* w_device,
                              const std::vector<int>& w_shape, const Tensor* bias,
                              int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
  // w_device 为借用指针，本入口不上传或释放权重
  return CudaLinearWithDevicePointer(x, w_device, w_shape, bias, device_id);
#else
  (void)x;
  (void)w_device;
  (void)w_shape;
  (void)bias;
  (void)device_id;
  throw CudaMatmulNotBuiltError();
#endif
}

CudaTensor CudaLinearDeviceInput(const CudaTensor& x, const void* w_device,
                                 const std::vector<int>& w_shape, int device_id) {
#ifdef MINI_LLAMA_USE_CUDA
  // 返回设备 Tensor 的所有权；是否下载由调用方决定，本接口没有 bias 参数
  return CudaLinearDeviceInputWithDevicePointer(x, w_device, w_shape, device_id);
#else
  (void)x;
  (void)w_device;
  (void)w_shape;
  (void)device_id;
  throw CudaMatmulNotBuiltError();
#endif
}

}  // namespace mini_llama
