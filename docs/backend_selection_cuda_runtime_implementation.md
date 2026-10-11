# 后端选择、CUDA Runtime 与工作区扩展实现说明

本文按当前工作区同步。Runtime、F32 设备 Tensor、模型权重上传、cuBLAS F32
Matmul/Linear、Ops/Quant Kernel、GPU KV Cache 和 Attention 已有实现。
库前向在存在完整上传快照时执行设备驻留计算；CLI generate/run/bench 已接入。
历史阶段与最近一次实际回归分别记录，独立算子与模型接入不混作同一状态。

此前设备 Tensor、模型上传阶段见
[历史阶段总结](cuda_tensor_model_weights_implementation.md)；当前 Ops/Quant 与
上传维护阶段见[阶段实现总结](cuda_ops_quant_implementation.md)；完整链路见
[设备驻留 CUDA Decode](device_resident_cuda_decode.md)。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| [backend.h](../include/mini_llama/backend.h)、[backend.cc](../src/backend.cc) | 后端配置、名称解析、环境校验及执行说明 |
| [cuda_runtime.h](../include/mini_llama/cuda_runtime.h)、[cuda_runtime.cc](../src/cuda_runtime.cc) | 设备查询、显存分配与拷贝、Buffer 所有权、CPU stub |
| [cuda_tensor.h](../include/mini_llama/cuda_tensor.h)、[cuda_tensor.cc](../src/cuda_tensor.cc) | 连续 F32 激活 Tensor、形状与字节数校验、主机上传下载 |
| [cuda_matmul.h](../include/mini_llama/cuda_matmul.h)、[cuda_matmul.cc](../src/cuda_matmul.cc) | cuBLAS F32 Matmul/Linear，主机、设备权重和设备输入接口 |
| [cuda_ops.h](../include/mini_llama/cuda_ops.h)、[cuda_ops.cu](../src/cuda_ops.cu) | 独立 RMSNorm、SiLU、Softmax、逐元素算子、Embedding 和 RoPE；`.cc` 提供 CPU Stub |
| [cuda_quant.h](../include/mini_llama/cuda_quant.h)、[cuda_quant.cu](../src/cuda_quant.cu) | 独立 Q8_0/Q4_0/Q4_1 块格式 Linear；`.cc` 提供 CPU Stub |
| [cuda_attention.h](../include/mini_llama/cuda_attention.h)、[cuda_attention.cu](../src/cuda_attention.cu) | 因果 GQA Attention 的主机/设备查询接口 |
| [cuda_kv_cache.h](../include/mini_llama/cuda_kv_cache.h)、[cuda_kv_cache.cc](../src/cuda_kv_cache.cc) | F32 GPU KV 历史、设备写入、调试回读与清零 |
| [model.h](../include/mini_llama/model.h)、[model.cc](../src/model.cc) | 权重上传快照、权重/运行统计、整模型重新量化 |
| [forward.cc](../src/forward.cc) | 完整快照校验、CPU/设备前向分发、设备算子与缓存串联、运行计数 |
| [main.cc](../src/main.cc) | CLI 参数、CUDA 诊断和推理接入边界 |
| [cuda_runtime_test.cc](../tests/cuda_runtime_test.cc) | Runtime、所有权、失败状态及 CLI 行为测试 |
| [tensor_test.cc](../tests/tensor_test.cc) | 空 Tensor、形状与 CPU stub 校验；真实 GPU 上传下载、移动与失败状态保持 |
| [forward_test.cc](../tests/forward_test.cc)、[cuda_forward_linear_test.cc](../tests/cuda_forward_linear_test.cc) | CPU/完整 GPU 前向、四种 Linear、GQA、bias、KV、快照生命周期与数值对照 |
| [cuda_ops_test.cc](../tests/cuda_ops_test.cc)、[cuda_quant_test.cc](../tests/cuda_quant_test.cc) | 独立算子的 CPU Stub、GPU 数值及量化输入校验 |

## 后端配置与 CLI

`BackendConfig` 默认 `kind = kCpu`、`device_id = 0`、`device_id_set = false`。
`ParseBackendKind()` 只接受小写 `cpu/cuda`，失败不改变原值；未知枚举由
`ValidateBackend()` 拒绝。显式传入 `--device 0` 也需要 `--backend cuda`。

`generate/run/bench/inspect` 接受后端和设备参数。设备编号接受 `0..INT_MAX`，
拒绝负数、尾随字符、溢出及重复选项。`inspect` 只校验参数并读取元数据，
不查询 GPU 或上传权重，因此 CPU 构建也可检查 CUDA 配置。

`CudaBackendBuilt()` 表示编译宏已启用；`CudaBackendAvailable()` 查询设备数量，
查询异常转为 false；`CudaDeviceSummary()` 返回设备说明或具体异常信息。
设备可见不保证显存分配成功。

```text
CLI 推理：ParseOptions -> ValidateBackendOrPrint -> ValidateBackend
  CPU  -> 加载模型与 tokenizer -> 可选 --quant -> CPU 生成/聊天/基准
  CUDA -> 检查构建与设备 -> 加载 -> 可选 --quant -> 上传 -> GPU 前向 + CPU 采样

诊断：--cuda-info -> CudaDeviceCount -> CudaGetDeviceInfo -> CudaFormatDeviceInfo
检查：inspect -> 参数校验 -> 元数据读取 -> compute: none
```

`BackendExecutionNote()` 的 CUDA 分支返回
`compute: cuda (device-resident decode; CPU sampler)`。`src/main.cc` 在推理前
调用 `UploadModelWeightsToCuda()`；`--quant q8_0|q4_0` 先转换 Linear 再上传。
库层 `ForwardToken()` 根据 `model.cuda_weights` 分发；库入口 `Run()` 则按后端
参数准备模型副本，选择 CPU 时解除副本的设备快照。

## 当前 Runtime 行为

| 接口 | 当前行为 |
| --- | --- |
| `CudaRuntimeBuilt()` | 检查编译宏，无需 GPU |
| `CudaDeviceCount()` | CUDA 查询设备数，无设备返回 0；其他 API 错误及 CPU stub 抛异常 |
| `CudaSetDevice(id)` | CUDA 构建校验设备编号后调用 cudaSetDevice；CPU 构建抛异常 |
| `CudaGetDeviceInfo(id)` | 查询数量、校验范围，再读取名称、CC、显存与驱动/Runtime 版本 |
| `CudaFormatDeviceInfo(info)` | GPU、CC、显存字节/GiB、驱动/Runtime 版本及原始编码 |
| `CudaMallocBytes/FreeBytes/MemcpyBytes` | 零长度/空释放返回；非零检查 API，CPU stub 抛异常 |

CC 表示 Compute Capability；驱动与 Runtime 版本同时显示解码值及原始整数编码。

拷贝方向包括 Host→Device、Device→Host、Device→Device。非零拷贝拒绝空指针
及未知方向。原始接口使用调用线程的当前设备。

CPU stub 的统一错误为以下 C++ 字符串拼接得到的单行文本：

```cpp
"CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON "
"on a NVIDIA CUDA machine."
```

CUDA API 错误包含接口名和 cudaGetErrorString 的原因。cudaErrorNoDevice 转为
零设备，驱动与初始化失败仍报告异常。版本按 `major * 1000 + minor * 10` 解码，
例如 12090 显示为 `12.9 (12090)`。

## Buffer 与 CudaTensor

`CudaDeviceBuffer` 禁止复制，支持 noexcept 移动与析构。非零操作通过私有
DeviceGuard 临时选择所属设备并恢复原设备。`Reset()` 释放成功后清空状态，
释放失败时保留所有权；`Reset(bytes, id)` 先分配替换对象，再释放旧 Buffer，
分配失败保留原数据。`Reset(0, id)` 释放旧资源并记录编号，不创建分配。
Upload/Download 先检查容量，零长度直接返回。析构捕获释放异常并输出诊断。

`CudaTensor` 保存 shape、元素数、设备编号与 Buffer，支持上传、下载、reset、
移动。非空 shape 要求正维度，并检查元素数和字节乘法溢出；默认/空 shape 表示
空设备 Tensor。与主机 Tensor 传输前检查 shape 和实际存储大小。重新分配失败
保留旧显存与元数据；移动后源对象恢复为空 Tensor，自移动保持状态。

## 模型上传、重新量化与统计

`UploadModelWeightsToCuda(model, device_id)` 要求已加载的有效模型，验证配置、
线性权重形状与块数；接受 F32/Q8_0/Q4_0/Q4_1 Linear。Embedding、norm、可选
Q/K/V bias 以 F32 上传。CPU 权重保留，全部上传成功后替换快照；失败保留旧快照。

模型副本共享 `shared_ptr<CudaModelWeights>`。快照包含名称、类型、shape、
块数和独占显存 Buffer。手工修改 CPU 权重后需要重新上传；快照不会自动同步。
`ClearModelCudaWeights()` 解除当前模型的引用，不影响其他仍持有快照的副本。

上传总数和字节数包括所有权重；F32/Q8_0/Q4_0/Q4_1 分类计数只统计 Linear。
运行统计包含 Linear、辅助激活、
Attention/CPU 回退、KV 读写、Host/Device 拷贝次数与字节。
`ResetModelCudaRuntimeStats()` 只清运行统计，共享快照的模型副本也会看到清零。
完整设备前向在成功操作后累计 Linear、Norm/RoPE/SiLU/Mul/Add、Attention、
KV 读写和最终 logits 下载；正常路径不做 CPU Attention 回退，不上传激活。
每次 Forward 下载 `vocab_size * sizeof(float)` 字节。KV Read 按查询头累计
逻辑访问，GQA 共享数据重复计入，不等于物理 DRAM 流量。权重上传、缓存设备
清零与显式调试回读不计入前向拷贝统计。独立 Ops/Quant 接口不接受模型，
不自行更新这些计数；普通 CPU 前向也不计作 CUDA fallback。
计数器共享且非原子，沿用单调用线程约定。

`QuantizeModelToQ80/Q40()` 可从 F32/Q8_0/Q4_0/Q4_1 转换所有 Linear，保持
embedding/norm/bias 不变。先准备全部替换数据，失败保留原模型；成功改变权重后
清除当前模型的旧 CUDA 快照。已是目标类型的权重只校验，不重复量化。

## 前向路径与上下文约定

`ForwardToken()` 在无快照时执行 CPU 算子；存在快照时先校验所有必需权重的
名称、类型、shape、块数、设备与实际容量，再执行完整 GPU 前向。不完整或
手工空快照报错，不以 CPU 回退掩盖缺失权重。必需 Embedding/Norm 为空时上传
失败；可选 bias 可省略。所有投影支持 F32/Q8_0/Q4_0/Q4_1 设备权重。

Embedding、Norm、QKV/bias、RoPE、Attention、O 投影、FFN、残差和 LM Head
均使用设备输入，最终 logits 回到主机供采样。`MiniLlamaContext` 构造仍分配
CPU Cache；GPU Cache 在首次 CUDA Forward 时分配并设备清零，之后保留历史，
GPU 路径不镜像更新 CPU Cache。Prefill 与 Decode 仍共用逐 token Forward。

首次 GPU 前向必须从位置 0 开始；已有 Cache 必须匹配设备与模型配置。
ForwardBatch 的历史连续性检查依赖 `token_history`；直接调用 ForwardToken
时位置和历史由调用方维护。同配置、同权重的重新上传可继续使用旧 GPU Cache；
实际改变权重、切换后端或外部清空 Cache 时须新建上下文并重放历史。
清除快照后不能继续使用含 GPU 历史的上下文，新建 CPU 上下文可恢复 CPU 前向。

测试覆盖 CPU/GPU logits 与 KV 对照、Normal/NeoX RoPE、GQA、bias、四种
Linear、设备权重所有权、缺失/损坏快照和缓存边界，以及 0 H2D / 1 D2H。
多 GPU、compute-sanitizer、临时 Tensor 和句柄复用仍未验证或实现。

## 构建与验证

CMake 默认关闭 `MINI_LLAMA_CUDA`，CPU 构建编译 `.cc` Stub，不需要 CUDA
编译器或 Toolkit。打开后启用 CUDA 语言并寻找 CUDAToolkit，编译
`cuda_ops.cu` / `cuda_quant.cu` / `cuda_attention.cu`，链接 `CUDA::cudart` 和 `CUDA::cublas`，
公开 `MINI_LLAMA_USE_CUDA=1`。C++ / CUDA 标准均为 C++17。

测试只注册存在的源文件；Runtime 与 Tensor 分别使用 `cuda_runtime.` 和
`cuda_tensor.` 前缀。需要 GPU 的用例在未构建、无设备或设备查询失败时明确
跳过；设备查询成功后的上传、下载等操作失败仍计为测试失败。复现命令：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON
cmake --build build-cuda -j4
ctest --test-dir build-cuda -R 'cuda_runtime|cuda_tensor|ForwardTest' --output-on-failure
ctest --test-dir build-cuda --output-on-failure --no-tests=error \
  -R 'cuda_matmul|cuda_linear|cuda_forward|cuda_ops|cuda_quant|cuda_kv_cache|cuda_attention'
./build-cuda/mini-llama --cuda-info
```

### clangd 的 CUDA 文件诊断

需要先成功配置 `MINI_LLAMA_CUDA=ON`，生成
`build-cuda/compile_commands.json`。CPU 构建数据库不包含 `.cu`，clangd 若从
同名 `.cc` 推断命令，会遗漏 CUDA 路径并产生 `cuda_fp16.h` / `cuda_runtime_api.h`
找不到及 `__global__`、`blockIdx` 未定义等连锁错误。

不添加项目配置文件时，可在检查命令中显式选择 CUDA 编译数据库。
本机 Clangd 18.1.3 使用 CPU 数据库时两个文件各有 23 个错误，使用下面的
CUDA 数据库检查命令时均为 0 错误；普通 `.cc` 仍按默认规则检查。

```bash
clangd --check=src/cuda_ops.cu --compile-commands-dir=build-cuda
clangd --check=src/cuda_quant.cu --compile-commands-dir=build-cuda
clangd --check=src/cuda_ops.cc
```

编辑器可在 clangd 的启动参数中指定
`--compile-commands-dir=/path/to/mini-llama/build-cuda`，直接读取构建目录中的
数据库。这里需要换成实际绝对路径；无需在项目根目录添加配置或数据库文件。

最近一次实际全量回归与 CLI/真实模型验证见
[CLI 同步说明](cli_integration_status.md#回归验证)。
2026-10-10 单 GPU 回归已实际执行完整设备前向、Q4_1 数值与 CUDA CLI；
对应链路与逐 token 拷贝验收见[设备驻留 Decode](device_resident_cuda_decode.md)。

以下为 Ops/Quant 阶段的历史回归：CPU 注册 189 项，161 项通过、28 项跳过；
CUDA 单 GPU 注册 189 项，182 项通过、7 项跳过，失败均为 0。
Ops/Quant 专项匹配 8 项：CPU 3 项通过、5 项 GPU 用例跳过；GPU 5 项通过、
3 项 CPU Stub 跳过。Q8_0/Q4_0 已验证向量、batch、尾块、bias、设备指针与输入。
Q4_1 GPU 计算尚未专项验证。原始日志见
[本阶段验证记录](cuda_ops_quant_implementation.md)。
这些历史结果包括 F32 混合前向和独立算子，当时未覆盖完整设备前向或 CLI CUDA 推理。

历史 Runtime/Tensor 阶段回归：CPU 全量 169 项中 157 项执行通过、12 项跳过；
CUDA 单 GPU 全量 169 项中 166 项执行通过、3 项跳过，失败均为 0。
Runtime 专项共 13 项：CPU 5 项通过、8 项跳过，CUDA 单 GPU 11 项通过、2 项跳过。
CUDA Tensor 专项共 5 项：CPU 3 项通过、2 项跳过，CUDA 单 GPU 4 项通过、1 项跳过。
GPU 上已执行 Tensor 上传下载、移动和失败 Reset，以及 F32/Q8_0 模型上传。
当时未验证多 GPU 或 GPU 前向。完整配置和跳过原因见 CLI 同步说明。

完整模块清单见 [README](../README.md)，CLI 行为见
[命令行与模块同步说明](cli_integration_status.md)。
