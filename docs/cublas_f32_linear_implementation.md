# cuBLAS F32 Matmul / Linear

本阶段补齐已有 CUDA Linear 草稿的公共接口，复用 `Tensor`、`CudaTensor`、
`CudaDeviceBuffer`、设备选择、模型上传快照与运行统计。CPU 构建有明确的异常
stub，不依赖 CUDA Toolkit。CUDA 构建链接 `CUDA::cudart` 和 `CUDA::cublas`。

**当前仅完成 CUDA Linear 与 CPU 辅助算子混合的 Forward。** 本仓库没有设备
Attention、RMSNorm、RoPE、SwiGLU、残差加法或 CUDA KV Cache。本模块遵循
不扩展 Attention 的范围限制，没有实现这些依赖，也没有声称中间激活全程在 GPU。

## 修改文件

- `include/mini_llama/cuda_matmul.h`：Matmul、三种 Linear 的公共接口与所有权约定。
- `src/cuda_matmul.cc`：cuBLAS GEMM、形状/存储校验、RAII、CPU stub。
- `src/forward.cc`：按权重名称和类型查找快照，各层七种投影与 LM Head 分流。
- `CMakeLists.txt`、`tests/CMakeLists.txt`：cuBLAS 链接与测试注册。
- `tests/cuda_matmul_test.cc`、`tests/cuda_forward_linear_test.cc`：算子与集成测试。
- `tests/unit/forward_test.cc`：保留上传前后的数值对照，按 F32 分流核对运行统计。

保留用户回退后的 Model、Context 定义，不修改两者或添加后端开关成员，
`include/mini_llama/forward.h`、`src/batch.cc` 保持原始三参数调用，不引入新接口
或后端参数，不创建额外测试辅助头文件。
GPU fixture、确定性数据和 `RequireCloseTensor`
分别放在两个测试文件的匿名命名空间内。

测试源码使用 `_test.cc` 后缀。单算子绝对/相对组合容差为 `2e-5`，Forward 为
`1e-4`；参考值使用项目 CPU `Matmul`、`Linear` 和 `ForwardBatch` 计算。
CPU Linear 目前只支持向量或单行，批量 Golden 逐行复用 CPU Linear 生成。

## 内存布局与 GEMM 参数

cuBLAS GEMM 使用列主序，`lda/ldb/ldc` 表示物理存储中相邻列的步长，转置操作
不会改变这个物理步长。接口定义见 [NVIDIA cuBLAS GEMM 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)。

### Matmul

行主序 `A[M,K]` 的内存对应列主序 `Aᵀ[K,M]`，行主序 `B[K,N]` 的内存对应
列主序 `Bᵀ[N,K]`。计算 `Cᵀ[N,M] = Bᵀ[N,K] × Aᵀ[K,M]`：

| 参数 | 配置 | 来源 |
| --- | --- | --- |
| 第一个指针、transa | B、`CUBLAS_OP_N` | B 的字节已表示列主序 Bᵀ |
| 第二个指针、transb | A、`CUBLAS_OP_N` | A 的字节已表示列主序 Aᵀ |
| m / n / k | N / M / K | Cᵀ 的行数/列数及归约维 |
| lda / ldb / ldc | N / K / N | Bᵀ、Aᵀ、Cᵀ 的列步长 |
| alpha / beta | 1 / 0 | 计算乘积，不读取原输出 |

列主序 `Cᵀ(j,i)` 的偏移为 `j+i*N`，等于行主序 `C(i,j)` 的 `i*N+j`。
因此下载后可直接返回 `[M,N]`，没有物理转置或数据重排。

### Linear

令 batch 为 R、in_features 为 I、out_features 为 O。向量对应 R=1。
`Y[R,O] = X[R,I] × W[O,I]ᵀ`，所以 `Yᵀ[O,R] = W[O,I] × Xᵀ[I,R]`。

W 的行主序字节表示列主序 `Wᵀ[I,O]`，须使用 `OP_T` 得到 `[O,I]`。
X 的字节表示列主序 `Xᵀ[I,R]`，使用 `OP_N` 即可。

| 参数 | 配置 |
| --- | --- |
| 第一个指针、transa | W、`CUBLAS_OP_T` |
| 第二个指针、transb | X、`CUBLAS_OP_N` |
| m / n / k | O / R / I |
| lda / ldb / ldc | I / I / O |
| alpha / beta | 1 / 0 |

W 在转置前的物理列步长是 I，因此 `lda=I`，不是 O。输出 `Yᵀ(o,r)` 的
偏移 `o+r*O` 等于行主序 Y 的偏移。向量返回 `[O]`，批量返回 `[R,O]`。

## 三种接口与资源管理

| 接口 | 数据位置与拷贝 | 场景 |
| --- | --- | --- |
| `CudaLinear` | CPU X/W；上传 X/W；下载 Y | 单算子验证，每次上传权重 |
| `CudaLinearDeviceWeight` | CPU X、GPU W；仅上传 X、下载 Y | 常驻权重与 CPU 算子边界 |
| `CudaLinearDeviceInput` | GPU X/W/Y；接口内无 H2D/D2H | 串联 GPU 投影与后续设备链路 |

前两个接口可选 bias，在下载的 CPU 结果上按输出特征维广播。bias 的 shape 和
存储在计算前校验。第三个接口没有 bias 参数。所有输入须为合法的向量/二维
矩阵，权重须为二维且输入特征匹配；检查正维度、元素数/字节数溢出及主机存储。
设备输入检查 `x.device_id() == device_id`；裸权重指针的设备和生命周期由调用方
保证，接口不上传或释放借用的权重。

保留草稿中的 `CublasHandle`：构造先 `CudaSetDevice` 再 `cublasCreate`，状态经
`CheckCublas` 转为项目使用的 `std::runtime_error`；析构 `cublasDestroy`，禁止
复制。临时显存由现有 Buffer/Tensor 析构释放。每次计算创建/销毁一个句柄；
句柄池、复用和 stream 管理留作后续性能优化。

## Forward 分流与计数范围

保持既有 `ForwardToken`、`ForwardBatch` 三参数接口。Forward 内部按权重名称
与类型查询上传快照，F32 命中后执行 CUDA Linear。无需添加公开调用声明、
后端参数或 Model/Context 开关。库调用示例：

```cpp
#include "mini_llama/forward.h"

UploadModelWeightsToCuda(model, 0);
MiniLlamaContext ctx(&model);
ResetModelCudaRuntimeStats(model);
Tensor logits = ForwardBatch(ctx, model, MiniBatch::Single(token, pos));
```

没有上传快照、空快照、权重查找未命中及量化权重均使用原 CPU Linear。
`ClearModelCudaWeights(model)` 后，后续前向恢复 CPU 路径。

`FindCudaWeight` 按 `name/type` 查找 Linear 权重；F32 命中后上传 CPU 激活，
调用 `CudaLinearDeviceInput`，再下载以衔接现有 CPU 算子。每次成功的真实上传、
Linear 调用、下载分别记录次数和字节；Q/K/V 实际执行 CUDA 投影后，CPU
Attention 记录 fallback 次数，
不冒充 CUDA Attention。未命中或量化权重仍走原 CPU `Linear`。

循环覆盖模型实际层数中的 wq/wk/wv/wo/w_gate/w_up/w_down 和 lm_head。
Decode 借用上传快照，不重新上传静态权重。无设备权重时的 CPU 前向和 CLI
既有边界保留；CLI 尚不提供完整 CUDA 推理。

无 bias 两层测试模型上传 21 项，其中 15 项是 F32 Linear，六项是
embedding/final_norm/各层 norm。bias 会增加上传项，但不增加 Linear 调用数。
ForwardBatch 当前逐 token 调用 ForwardToken，所以多 token 的调用数成倍增加。
初始化和权重上传完成后重置统计；Prefill 后再次重置，以独立验证 Decode。

| 单 token 两层模型 | 完整 GPU 链路验收目标 | 当前 CPU fallback 路径 |
| --- | --- | --- |
| 上传总数 / F32 Linear 数 | 21 / 15 | 21 / 15 |
| CUDA Linear calls | 15 | 15 |
| CUDA Attention calls | 2 | 0 |
| CPU Attention fallbacks | 0 | 2 |
| H2D / D2H copies | 0 / 1 | 15 / 15 |

完整设备链路的验收目标保持不变。当前集成测试明确测试 fallback 边界，其通过
不能证明 0 H2D / 1 D2H 或激活常驻 GPU；这些目标等待已有 CUDA 依赖接入。

## 验证

```bash
ctest --test-dir build-cuda \
  -R 'cuda_matmul|cuda_linear|cuda_forward_linear' --output-on-failure
./build-cuda/mini-llama-cuda-matmul-tests matmul
./build-cuda/mini-llama-cuda-matmul-tests linear
./build-cuda/mini-llama-cuda-forward-linear-tests
```

2026-10-09 在保留 Model、Context 回退，恢复 Forward/Batch 原始调用框架后，
通过内部 F32 权重查找分流重新构建验证。环境为 NVIDIA GeForce RTX 4060
GPU、CUDA Runtime 12.9、cuBLAS
12.9。沙箱不允许访问 GPU，真实 GPU 测试在沙箱外执行。原临时 SDK 已
失效，使用 NVIDIA Runtime/cuBLAS/NVCC/CCCL 包恢复 `/tmp` SDK；包 SHA256
逐项校验，环境路径未写入项目构建配置。

| 检查 | 实际结果 |
| --- | --- |
| CPU 构建 | 成功 |
| CPU 全量 CTest | 158 通过、22 跳过、0 失败；跳过项需要 CUDA/GPU |
| CUDA 构建 | 成功 |
| CUDA Matmul/Linear/Forward 用例 | 本次全量回归内执行：10 通过、0 跳过、0 失败 |
| CUDA 全量 CTest | 176 通过、4 跳过、0 失败，耗时 7.74 秒 |
| 格式与 diff 检查 | clang-format / git diff --check 通过 |

CUDA 全量中的跳过项是三个 CPU stub 专用测试和一个需要第二块 GPU 的
Runtime 测试。本模块的数值、bias、批量输入、GPU 输出串联、设备不匹配、
CPU fallback 和权重常驻探针均已实际执行。
上传快照清除后，后续三参数调用按 CPU 执行，并保持 CUDA 运行计数为零。

两层模型 Decode 的实际输出统计：

```text
uploaded=21, F32=15, linear=15, attention=0,
attention_cpu_fallbacks=2, H2D=15, D2H=15
```

对应激活传输字节为 H2D=1112、D2H=1100。Prefill 的两个 token 对应调用和
传输均为两倍，Decode 前重置运行统计。上传快照对象、每项设备权重指针和
上传总数保持不变；CPU 投影副本置零后，Decode 仍与原模型 Golden 一致。

尚未完成完整设备 Forward 验收：CUDA Attention=2、H2D=0、D2H=1 以及
中间激活全程 GPU 常驻需要仓库缺失的 CUDA 辅助算子和 KV Cache/Attention
依赖。本次未扩展 Attention，也未更改这些验收目标。
