# 后端选择、CUDA Runtime 与工作区扩展实现说明

本文按当前工作区同步。CUDA Runtime、F32 CUDA Tensor 和模型权重上传已有库
实现；当前前向继续使用 CPU 算子，上传快照不切换计算后端。GPU 推理尚未接入。
历史 Runtime 阶段的运行结果与本次回归结果分别记录。

设备 Tensor、权重上传与 CPU 前向兼容的本阶段总结见
[CUDA Tensor 与模型权重上传](cuda_tensor_model_weights_implementation.md)。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| [backend.h](../include/mini_llama/backend.h)、[backend.cc](../src/backend.cc) | 后端配置、名称解析、环境校验及执行说明 |
| [cuda_runtime.h](../include/mini_llama/cuda_runtime.h)、[cuda_runtime.cc](../src/cuda_runtime.cc) | 设备查询、显存分配与拷贝、Buffer 所有权、CPU stub |
| [cuda_tensor.h](../include/mini_llama/cuda_tensor.h)、[cuda_tensor.cc](../src/cuda_tensor.cc) | 连续 F32 激活 Tensor、形状与字节数校验、主机上传下载 |
| [model.h](../include/mini_llama/model.h)、[model.cc](../src/model.cc) | 权重上传快照、权重/运行统计、整模型重新量化 |
| [forward.cc](../src/forward.cc) | CPU 前向，F32/Q8_0/Q4_0/Q4_1 Linear 分发，不消费 CUDA 上传快照 |
| [main.cc](../src/main.cc) | CLI 参数、CUDA 诊断和推理接入边界 |
| [cuda_runtime_test.cc](../tests/cuda_runtime_test.cc) | Runtime、所有权、失败状态及 CLI 行为测试 |
| [cuda_tensor_test.cc](../tests/cuda_tensor_test.cc) | 空 Tensor、形状与 CPU stub 校验；真实 GPU 上传下载、移动与失败状态保持 |
| [forward_test.cc](../tests/unit/forward_test.cc) | CPU F32/Q8_0 前向、GQA、bias、KV 对照；GPU 上传快照与上传不切换后端的契约 |

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
CLI 推理：ParseOptions -> ValidateInferenceBackend -> ValidateBackend
  CPU  -> 加载模型与 tokenizer -> CPU 生成/聊天/基准
  CUDA -> 检查构建与设备 -> 报 CUDA inference is not integrated yet

诊断：--cuda-info -> CudaDeviceCount -> CudaGetDeviceInfo -> CudaFormatDeviceInfo
检查：inspect -> 参数校验 -> 元数据读取 -> compute: none
```

`BackendExecutionNote()` 的 CUDA 分支仍返回执行路径未实现的说明。
`src/main.cc` 没有调用 `UploadModelWeightsToCuda()`，也没有模型量化 CLI 选项。
库层的上传快照与 CLI 的 `BackendConfig` 并非同一个状态。

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
线性权重形状与块数；当前只接受 F32/Q8_0 Linear。Embedding、norm、可选
Q/K/V bias 以 F32 上传。CPU 权重保留，全部上传成功后替换快照；失败保留旧快照。

模型副本共享 `shared_ptr<CudaModelWeights>`。快照包含名称、类型、shape、
块数和独占显存 Buffer。手工修改 CPU 权重后需要重新上传；快照不会自动同步。
`ClearModelCudaWeights()` 解除当前模型的引用，不影响其他仍持有快照的副本。

上传总数和字节数包括所有权重；F32/Q8_0/Q4_0/Q4_1 分类计数只统计 Linear。
正常上传快照中的 Q4_0/Q4_1 分类计数为 0。运行统计包含 Linear、辅助激活、
Attention/CPU 回退、KV 读写、Host/Device 拷贝次数与字节。
`ResetModelCudaRuntimeStats()` 只清运行统计，共享快照的模型副本也会看到清零。
当前 CPU 前向不更新这些运行字段；CUDA Linear、激活、Attention 与 GPU KV
统计均没有对应的已接入执行路径，不能根据预留字段报告 GPU 推理已实现。

`QuantizeModelToQ80/Q40()` 可从 F32/Q8_0/Q4_0/Q4_1 转换所有 Linear，保持
embedding/norm/bias 不变。先准备全部替换数据，失败保留原模型；成功改变权重后
清除当前模型的旧 CUDA 快照。已是目标类型的权重只校验，不重复量化。

## 前向路径与未完成部分

`ForwardToken()` 不根据 `model.cuda_weights` 切换后端。Embedding、Linear、
bias、RMSNorm、SwiGLU、残差与 RoPE 均使用现有 CPU 实现。`MiniLlamaContext`
只持有 CPU `KvCache`，Attention 在 CPU 上计算。

`forward_test.cc` 中的 F32/Q8_0 参数均测试 CPU 前向；手工构造空上传快照，
验证快照存在时 logits、KV、序列状态与运行统计不受影响。另有两项真实 GPU
上传测试，验证权重字节、数量、显存统计、上传失败保留旧快照，以及上传后
CPU logits 不变。这些用例不执行 GPU 前向，也不构成 CPU/CUDA logits 对齐
验证。CUDA 算子、GPU Attention、GPU KV Cache 和设备驻留 decode 属于后续模块。

## 构建与验证

CMake 默认关闭 `MINI_LLAMA_CUDA`。打开后寻找 CUDAToolkit 的头文件与 Runtime
库，以 C++17 编译并链接 `CUDA::cudart`，公开 `MINI_LLAMA_USE_CUDA=1`。当前
没有 kernel 或 CUDA 语言源码，不启用 CUDA 编译器。

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
./build-cuda/mini-llama --cuda-info
```

2026-10-08 重新构建与回归结果：CPU 全量 169 项中 157 项执行通过、12 项跳过；
CUDA 单 GPU 全量 169 项中 166 项执行通过、3 项跳过，失败均为 0。
Runtime 专项共 13 项：CPU 5 项通过、8 项跳过，CUDA 单 GPU 11 项通过、2 项跳过。
CUDA Tensor 专项共 5 项：CPU 3 项通过、2 项跳过，CUDA 单 GPU 4 项通过、1 项跳过。
GPU 上已执行 Tensor 上传下载、移动和失败 Reset，以及 F32/Q8_0 模型上传。
未验证多 GPU 或 GPU 前向。完整配置和跳过原因见 CLI 同步说明。

完整模块清单见 [README](../README.md)，CLI 行为见
[命令行与模块同步说明](cli_integration_status.md)。
