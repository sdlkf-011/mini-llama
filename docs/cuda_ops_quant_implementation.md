# CUDA 基础算子、量化 Linear 与权重上传调整阶段总结（历史阶段）

> 本文主体记录 `15643de` 阶段，下面的“当前”指该阶段。此后四种格式权重
> 上传、完整设备前向、GPU Attention/KV Cache 和 CLI CUDA/量化均已接入，
> 必需空权重校验已修复。当前契约与验证见
> [设备驻留 Decode](device_resident_cuda_decode.md)及[CLI 同步说明](cli_integration_status.md)。

本阶段对应提交 `15643de`（`CUDA Kernels: Act, Quant, Upload`）。
新增独立 CUDA 基础算子和 Q8_0/Q4_0/Q4_1 Linear，接入 CPU Stub、
C++17/CUDA 构建与专项测试，并维护已有权重上传入口。既定公共接口、CPU
计算实现和模型/上下文结构在该阶段保持原样。

## 本阶段与此前阶段的关系

| 阶段 | 主要内容 | 对应文档 |
| --- | --- | --- |
| 设备存储与模型上传，`385cb8a` | CudaTensor、F32/Q8_0 上传快照、共享生命周期、统计接口、整模型重新量化 | [历史阶段总结](cuda_tensor_model_weights_implementation.md) |
| cuBLAS F32 Linear，`0866516` | F32 Matmul/Linear、借用设备权重、库层 F32 混合前向 | [F32 阶段总结](cublas_f32_linear_implementation.md) |
| 本阶段，`15643de` | 独立 Ops/Quant Kernel、CPU Stub、上传入口维护和算子验证 | 本文 |

`cuda_tensor_model_weights_implementation.md` 的主体描述此前的设备存储阶段。
它在本次提交中增加了当前状态提示，但 CudaTensor、上传快照和整模型量化接口
并不是本阶段新建的模块。本阶段对已有上传实现的维护在下文单独列出。

## 文件与职责

| 文件 | 本阶段职责 |
| --- | --- |
| [cuda_ops.h](../include/mini_llama/cuda_ops.h) | 基础算子的主机、设备输入和借用设备权重接口 |
| [cuda_ops.cc](../src/cuda_ops.cc) | 构建状态查询与 CPU Stub |
| [cuda_ops.cu](../src/cuda_ops.cu) | RMSNorm、SiLU、Softmax、逐元素加乘、Embedding Lookup、Normal/NeoX RoPE |
| [cuda_quant.h](../include/mini_llama/cuda_quant.h) | 三种量化格式的主机、设备权重和设备输入 Linear 接口 |
| [cuda_quant.cc](../src/cuda_quant.cc) | 量化 Kernel 构建状态与 CPU Stub |
| [cuda_quant.cu](../src/cuda_quant.cu) | 逐块解码的量化 Linear、尾块处理、输入与 bias 校验 |
| [model.cc](../src/model.cc) | 上传前明确选择设备，集中处理空稠密权重与可选 bias |
| [CMakeLists.txt](../CMakeLists.txt)、[tests/CMakeLists.txt](../tests/CMakeLists.txt) | 条件编译 CUDA 源文件，链接 Runtime/cuBLAS，注册独立算子测试 |
| [cuda_ops_test.cc](../tests/cuda_ops_test.cc) | 基础算子 CPU Stub 与真实 GPU 数值对照 |
| [cuda_quant_test.cc](../tests/cuda_quant_test.cc) | Q8_0/Q4_0 的 GPU 数值、元数据/存储校验；三种格式 CPU Stub |
| [cuda_forward_linear_test.cc](../tests/cuda_forward_linear_test.cc) | 保留 F32 混合前向回归，新增上传快照共享所有权测试 |

未新增独立 CUDA Forward 模块、设备上下文成员或后端开关。原有头文件声明、
结构成员和默认值未改变；新增 Ops/Quant 接口也保持已经确定的签名。

## CUDA 基础算子

| 算子 | 实现方式与边界 |
| --- | --- |
| RMSNorm | 原子累加平方和，再逐元素应用 `rsqrt(mean(x^2)+eps)*weight`；输入和权重为同形状一维 F32，eps 有限且非负 |
| SiLU | 对连续 F32 元素计算 `x/(1+exp(-x))` |
| 逐元素加乘 | 同 shape 输入按元素计算，新建输出；设备权重加法不隐式广播 |
| Softmax | 对非空一维输入依次归约最大值、计算移位指数及总和、归一化 |
| Embedding Lookup | 借用设备上的 `[vocab_size, dim]` F32 权重，返回指定 token 行的设备向量 |
| Normal RoPE | 每个头内按相邻维度 `(2*i, 2*i+1)` 原地旋转 |
| NeoX RoPE | 每个头内按前后半区 `(i, i+head_dim/2)` 原地旋转；Q/K 可有不同头数 |

主机接口临时上传输入并下载结果。`CudaRmsNormDeviceInput` 的激活已在设备，
但 Norm 权重仍从主机上传；`CudaRmsNormDeviceWeight` 同时借用设备输入和权重。
设备逐元素、Embedding 和 RoPE 接口不下载输出。裸权重指针的设备、容量和
生命周期由调用方保证，算子不取得这些显存的所有权。

临时显存继续使用已有 `CudaDeviceBuffer` / `CudaTensor` 的 RAII。启动后读取
CUDA 错误状态，不为每次启动增加显式同步；执行期错误可能在后续 Runtime
操作或主机下载时报告。独立算子不持有模型，不更新模型运行统计。

## CUDA 量化 Linear

权重保持 `[out_features, in_features]` 的逐输出行块布局。每行独立按 32 个
元素分块，尾部补齐只影响存储，Kernel 只计算实际输入特征：

```text
blocks_per_row = ceil(in_features / 32)
block_count = out_features * blocks_per_row
```

| 格式 | 块内解码 |
| --- | --- |
| Q8_0 | FP16 缩放 `d` 与有符号 int8 值，`w=d*q` |
| Q4_0 | FP16 缩放与打包 4 位值，`w=d*(q-8)` |
| Q4_1 | FP16 缩放 `d`、偏移 `m` 与打包 4 位值，`w=d*q+m` |

Q4 每字节的低 4 位对应块内前 16 个元素，高 4 位对应后 16 个元素。
Kernel 网格的 y 维选择 batch 行，x 维选择输出通道；一个线程计算一个输出
点积，在使用权重元素时解码，不生成完整 F32 权重副本。输入和输出均为 F32，
支持 `[in]` / `[batch, in]`，输出保持相应的秩 `[out]` / `[batch, out]`。

| 每种格式的入口 | 数据位置与应用层传输 |
| --- | --- |
| `CudaQ*Linear` | 主机 X/W；临时上传 X/W，下载 Y |
| `CudaQ*LinearDeviceWeight` | 主机 X、借用设备 W；只上传 X、下载 Y |
| `CudaQ*LinearDeviceInput` | 设备 X/W/Y；接口内没有 H2D/D2H，返回输出显存所有权 |

前两个入口可选 `[out]` 主机 bias，在结果下载后沿 batch 行广播相加。
设备输入入口没有 bias 参数。借用设备权重的入口不上传或释放已有权重。
这些接口接收已有量化块，不执行模型重新量化。

计算前校验输入秩、特征轴、权重 shape、块数、主机实际存储和 bias 长度；
设备输入还校验所属设备。逻辑 shape 的元素数检查乘法溢出，块数计算使用
`(in_features-1)/block_size+1`，避免向上取整加法的整数溢出。
裸设备指针的实际容量无法由块数参数自行证明，仍是调用方的约束。

## 已有上传入口的维护

`UploadModelWeightsToCuda` 继续只支持 F32/Q8_0 Linear，保留原始 CPU 权重，
成功上传全部权重后发布快照。该快照及统计接口来自设备存储阶段。
本阶段实际调整为：

- 查询目标设备后显式调用 `CudaSetDevice(device_id)`，随后复用原有 Buffer 上传。
- 空稠密 Tensor 由 `UploadDenseWeight` 统一跳过，Q/K/V bias 不再在每个调用点单独判断。
- 新增共享所有权回归：重复上传释放旧快照，模型副本继续共享新快照，最后一个持有者清除引用后资源才释放。

这里没有扩展 Q4_0/Q4_1 整模型上传。独立量化算子可以借用调用方自行上传的
Q4 块，但不代表模型上传入口已经支持同样的格式。

当前空存储跳过逻辑也作用于必需的 Embedding/Norm。之前的验证发现，必需
稠密权重数据被清空时可能被当成未提供而跳过。这是已有上传实现仍需处理的
边界问题，不能将可选 bias 的正常行为视为必需权重校验已完备；本阶段未修复。

## CPU Stub 与构建

`CudaOpsBuilt()` / `CudaQuantBuilt()` 只报告编译状态，不保证 GPU 可用。
CPU 构建中的主机和设备计算入口均抛未构建异常，不访问设备指针，也不静默
执行 CPU 替代计算。CUDA 构建时 `.cc` 保留构建状态查询，计算定义由 `.cu` 提供。

CMake 默认关闭 `MINI_LLAMA_CUDA`；启用后开启 CUDA 语言并查找 CUDAToolkit。
C++ 和 CUDA 均使用 C++17，链接 `CUDA::cudart` 与 `CUDA::cublas`。
复现命令不包含本机专属 SDK 路径，其他设备须先安装/配置自己的 CUDA 编译器和 Toolkit。

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure --no-tests=error \
  -R 'cuda_ops|cuda_quant_q8_0|cuda_quant_q4'
ctest --test-dir build --output-on-failure -j4

cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure --no-tests=error \
  -R 'cuda_ops|cuda_quant_q8_0|cuda_quant_q4'
ctest --test-dir build-cuda --output-on-failure -j4
```

GPU 算子测试在未构建、无设备或设备查询不可用时明确跳过；查询成功后的
计算/传输异常仍使测试失败。`--no-tests=error` 防止未匹配测试被误报为通过。

## 测试覆盖与已执行结果

以下是本阶段已实际执行的记录。本次总结整理没有重新运行数值测试。
验证环境为 Ubuntu 24.04.4 / WSL2、CMake 3.28.3、GCC 13.3、Clang 18.1.3
CUDA 编译器、CUDA Toolkit/Runtime 12.9、RTX 4060 Laptop（CC 8.9）。
GPU 结果来自授权后可访问实际设备的运行环境，未把沙箱内的跳过计为通过。

| 范围 | 注册/匹配 | 实际通过 | 失败 | 跳过 |
| --- | ---: | ---: | ---: | ---: |
| CPU Ops/Quant 专项 | 8 | 3 | 0 | 5 |
| GPU Ops/Quant 专项 | 8 | 5 | 0 | 3 |
| CPU 全量回归 | 189 | 161 | 0 | 28 |
| GPU 全量回归 | 189 | 182 | 0 | 7 |

专项是全量回归的子集，不重复相加。GPU 全量跳过 6 项 CPU Stub 和 1 项需要
两张 GPU 的测试。CPU 专项的 3 项通过验证 Ops、Q8_0 和 Q4_0/Q4_1 Stub；
GPU 专项的 5 项实际执行基础算子数值对照及 Q8_0/Q4_0 数值/输入校验。

- Ops 对照 RMSNorm、SiLU、Softmax、加乘与两种 RoPE 布局；设备入口覆盖已经提供的设备版本，Softmax 仅有主机版本。
- Q8_0/Q4_0 覆盖输入宽度 16/32/35/64/86、向量与三行 batch、尾块、主机 bias、常驻设备权重及设备输入；下载原始块核对字节未被改写。
- 非法指针、块数、形状、设备，以及截断的输入/bias 均验证异常处理。
- 原有 F32 混合前向回归仍验证 prefill/decode、GQA、NeoX RoPE、bias、KV、设备权重指针与真实 fallback 计数；本阶段补充共享所有权测试目的注释。

本机原始日志、JUnit XML 和跳过名称清单保存在 `.cache/` 下的验证缓存中；
缓存不属于提交内容，清理后不影响本文的结果记录。

## 本阶段接入状态与后续边界

没有快照时库前向走 CPU；命中已有快照的 F32 Linear 执行 cuBLAS，量化或
未命中 Linear 继续走 CPU。Norm、RoPE、FFN 辅助运算、Attention 和 KV Cache
仍使用 CPU。独立 `cuda_ops` / `cuda_quant` 尚未接入这条模型前向链路，
也不会自动更新模型的激活、设备 Attention 或设备 KV 计数。

CLI 没有调用上传或整模型量化接口；`generate/run/bench --backend cuda`
仍在校验中拒绝，未提供 `--quant`。本阶段不包含完整设备前向、GPU Attention、
GPU KV Cache、设备常驻 decode 或 CUDA CLI benchmark。

Q4_1 的 GPU 数值和 Embedding Lookup 的独立 GPU 专项尚未验证；只有一张
GPU，多 GPU 测试未执行。验证使用小模型/算子输入，不构成大模型、长上下文、
显存耗尽或量化质量评估。相关后续模块完成接入后需要重新验证整条链路。
