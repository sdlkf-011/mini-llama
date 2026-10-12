# CUDA Tensor、模型权重上传与 CPU 前向兼容阶段总结（历史阶段）

本阶段以提交 `385cb8a`（`CudaDeviceBuffer 与 CudaTensor`）为实现基线，
在已有 Backend/CUDA Runtime 基础上补充设备 Tensor、模型权重上传快照、
统计接口与整模型重新量化，并保持现有 CPU 推理路径和 CLI 行为。
本文同时记录随后完成的阶段文档同步与 Markdown 警告修复。

本文是该提交阶段的历史记录，下面的“当前”指该阶段。此后的 cuBLAS F32
混合前向、Ops/Quant、GPU Attention/KV Cache 和完整设备前向均已接入；当前状态见
[README](../README.md)、[F32 Linear](cublas_f32_linear_implementation.md) 和
[设备驻留 Decode](device_resident_cuda_decode.md)，不应用历史描述覆盖这些实现。

随后 `15643de`（`CUDA Kernels: Act, Quant, Upload`）阶段的完整总结见
[CUDA 基础算子、量化 Linear 与上传调整](cuda_ops_quant_implementation.md)。
该提交新增的 Ops/Quant Kernel 与已有上传入口的设备选择、空稠密权重处理、
共享所有权回归在那里记录；本文的 CudaTensor、快照和整模型量化属于此前实现。

该阶段结果是可验证的设备存储与模型上传能力。当时 CUDA 算子、GPU Attention、
GPU KV Cache 和设备驻留 decode 属于后续实现；它们现在已经接入并通过单 GPU 验证。
后端参数与 Runtime 的完整约定见
[后端与 CUDA Runtime 说明](backend_selection_cuda_runtime_implementation.md)。

## 文件与职责

| 文件 | 本阶段职责 |
| --- | --- |
| [cuda_tensor.h][tensor-h] | F32 设备 Tensor 公共接口与独占所有权约定 |
| [cuda_tensor.cc][tensor-cc] | 形状校验、上传下载、移动与失败状态保持 |
| [cuda_runtime.h][runtime-h] | 显存 Buffer 与 Runtime 的接口说明 |
| [model.h][model-h] | 上传快照、统计查询与模型重新量化接口 |
| [model.cc][model-cc] | 权重遍历、校验、上传与量化转换 |
| [forward.cc][forward-cc] | 整理前向包装层，保持 CPU 计算路径 |
| [tensor_test.cc][tensor-test] | 空 Tensor、CPU stub 与真实 GPU 测试 |
| [forward_test.cc][forward-test] | CPU 前向兼容及 F32/Q8_0 上传测试 |
| [CMakeLists.txt][cmake] | 将设备 Tensor 加入核心库 |
| [tests/CMakeLists.txt][test-cmake] | 注册实际存在的 Tensor 与前向测试 |

## 设备 Buffer 与 F32 Tensor

### 复用已有 Buffer 约定

`CudaDeviceBuffer` 来自上一 Runtime 阶段。本阶段继续采用其显存所有权与
异常安全约定，新增 Tensor 和模型上传均复用该接口：

- Buffer 独占显存，禁止复制，支持 noexcept 移动与析构。
- 非零操作临时选择所属设备，完成后恢复调用线程原设备。
- 重新分配先准备替换 Buffer；分配失败保留旧指针、容量、设备与数据。
- 零字节分配、零长度拷贝和空指针释放在 CPU/CUDA 构建中均安全返回。
- 显式释放失败保留所有权并抛异常；析构捕获异常并输出诊断。

设备信息继续显示显存字节数/GiB，以及解码后的 CUDA 版本与原始整数编码。
`src/cuda_runtime.cc` 沿用已有实现，没有新增 CUDA kernel。

### CudaTensor 的形状与传输

新增 `CudaTensor` 表示连续布局的 F32 设备数据，记录 `shape`、元素数、
设备编号和底层 Buffer。默认对象与空 shape 表示零元素 Tensor。
非空 shape 的各维必须为正，并分别检查元素数与 F32 字节数乘法溢出。

| 接口 | 行为 |
| --- | --- |
| `Reset(shape, device_id)` | 校验并分配，成功后更新元数据 |
| `Reset()` | 显式释放，成功后恢复默认空状态 |
| `UploadFrom(host)` | 校验主机 shape、元素数和设备容量后上传 |
| `Download()` | 构造同 shape 的主机 Tensor 并下载 |
| `DownloadTo(host)` | 下载到已有匹配存储，不隐式改变 shape |
| `CudaTensorFromHost(host, id)` | 按主机 shape 分配并上传 |

`data()` 返回借用的设备指针，调用方不取得释放权限，也不能在 CPU 上直接
解引用。`size()` 按元素计数，`bytes()` 按底层显存字节计数。
`ShapeStringShort()` 只格式化元数据，不访问 GPU。

以下片段需要 CUDA 构建和可用 GPU：

```cpp
mini_llama::Tensor host({2, 3}, 1.0f);
auto device = mini_llama::CudaTensorFromHost(host, 0);
mini_llama::Tensor restored = device.Download();
```

### 移动与失败状态

移动构造和移动赋值一起转移 shape、元素数、设备编号及 Buffer。
源对象恢复为空 Tensor，可以执行零长度传输或重新分配；自移动保持状态。
这修复了默认移动只转移 Buffer、却保留源对象标量元数据的问题。

`Reset(shape, id)` 先校验并复制新 shape，再分配替换 Buffer，最后发布元数据。
形状校验、形状复制或显存分配失败时，旧显存和旧元数据保持有效。
上传下载前还会检查主机实际存储大小，避免逻辑 shape 正确但数据不足的拷贝。

## 模型权重上传与快照

### 支持范围与校验

`UploadModelWeightsToCuda(model, device_id)` 要求模型已加载，校验正维度、
层数、GQA/RoPE 约束及所有 Linear 的 shape、存储长度和量化块数。
当前上传仅接受 F32/Q8_0 Linear；Q4_0/Q4_1 上传明确拒绝。
Embedding、Norm 与实际存在的 Q/K/V bias 以 F32 上传。

量化 Linear 沿输出行保存块，每行独立补齐到 32 个元素：

```text
blocks_per_row = ceil(in_features / 32)
weight_blocks = out_features * blocks_per_row
```

Q8_0 上传保留原始块数据，不先展开整张 F32 权重矩阵。
CPU 权重继续保留，设备快照另外持有显存分配。

### 一次性发布与共享生命周期

上传先构建临时 `CudaModelWeights`，逐项准备设备 Buffer 并拷贝权重。
全部成功后才替换 `model.cuda_weights`；校验、分配或上传失败时，临时资源
由 RAII 回收，模型原有快照保持不变。

每项 `CudaUploadedWeight` 保存名称、类型、shape、量化块数、Linear 标记
和独占显存 Buffer。模型通过 `shared_ptr<CudaModelWeights>` 共享快照：

- 复制模型会共享同一份设备快照，CPU 权重仍按原有成员复制。
- `ClearModelCudaWeights()` 只解除当前模型引用，其他副本仍可持有快照。
- 手工修改 CPU 权重不会自动更新设备副本，需要重新上传或清除旧快照。
- `ModelHasCudaWeights()` 只判断引用非空，不校验快照完整性或推理可用性。

### 上传统计与预留运行统计

上传总数与显存字节数包括 Linear、Embedding、Norm 和可选 bias。
按 F32/Q8_0/Q4_0/Q4_1 分类的数量仅统计 Linear。
当前正常上传结果的 Q4_0/Q4_1 数量为零。

本阶段另外提供 Linear、激活、Attention、CPU 回退、KV 读写和主机/设备
传输的运行统计查询接口。当前 CPU 前向不更新这些字段，上传操作也不计入
推理传输统计。字段存在不代表对应 CUDA 计算路径已经实现。

`ResetModelCudaRuntimeStats()` 清零运行字段，保留权重和上传统计。
共享同一快照的模型副本也会看到统计清零；这些字段没有并发更新保护。

## 模型重新量化与 CPU 前向

### 整模型重新量化

新增 `QuantizeModelToQ80()` 与 `QuantizeModelToQ40()`，转换 LM Head 及
每层 Q/K/V/O、Gate/Up/Down Linear。支持从 F32/Q8_0/Q4_0/Q4_1 转换，
保持 Embedding、Norm 和 bias 不变。

转换先校验并准备全部替换权重，全部成功后统一写回；失败保留原模型。
已有量化格式会先反量化为 F32 临时 Tensor，再使用现有逐行量化接口。
转换过程中会产生额外主机内存，不是流式转换。

实际改变权重后清除当前模型的旧 CUDA 快照，避免设备副本过期。
已是目标格式的权重只校验；全模型无需转换时保留已有快照。
Q4_0 转换沿用原编码与舍入规则，没有新增 Q4_1 量化入口。

### 前向与 CLI 接入边界

`src/forward.cc` 整理了 Linear、QKV 投影、RMSNorm、SwiGLU、残差、
RoPE 与最终 logits 的包装层。配置维度关系使用 64 位乘法校验，
避免非法 head 配置导致有符号整数乘法溢出。

`ForwardToken()` 始终调用当前 CPU 算子，不根据上传快照切换后端。
CPU Linear 保留 F32/Q8_0/Q4_0/Q4_1 分发，Attention 和 KV Cache 仍在 CPU。
`ForwardBatch()` 继续按 token 顺序完成 prefill/decode。

CLI 没有调用上传或整模型重新量化接口。`generate/run/bench --backend cuda`
仍在后端校验中报错，`inspect --backend cuda` 仍只读取模型元数据。

## 构建与阶段验证

### 构建方式

核心库加入 `src/cuda_tensor.cc`，测试注册新增 `cuda_tensor.` 前缀。
移除引用不存在文件的 GPU 前向测试目标，保留真实设备存储与上传验证。

`MINI_LLAMA_CUDA` 默认关闭。开启后寻找 CUDAToolkit 头文件与 Runtime 库，
公开 `MINI_LLAMA_USE_CUDA=1` 并链接 `CUDA::cudart`。
当前仍以 C++17 编译，没有 `.cu` 源码或 CUDA 语言编译需求。

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4

cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure -j4
ctest --test-dir build-cuda \
  -R 'cuda_runtime|cuda_tensor|ForwardTest' --output-on-failure
```

非系统 SDK 可通过 `-DCUDAToolkit_ROOT=/path/to/sdk` 指定；复用本机 GTest
源码可设置 `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`。

### 测试覆盖

设备 Tensor 新增 5 项测试：空对象传输与移动、非法形状/溢出、CPU stub
分配失败、真实 GPU 上传下载与移动，以及失败 Reset 保留 shape 和数据。

该阶段前向专项共 12 项：原有 6 项加上 F32/Q8_0 参数化的 6 项扩展测试。
扩展测试覆盖两层非零 FFN、GQA、可选 bias、RoPE 布局与量化尾块：

- 批量 prefill/decode 与逐 token CPU 执行的 logits、KV 和序列状态一致。
- 手工快照不切换 CPU 前向，也不清零既有运行统计。
- 真实 GPU 上传后 CPU logits 不变，上传失败保留原快照。

真实上传用例的两层小模型包含 15 个 Linear 与 9 个稠密权重，共 24 项。
测试校验上传数量、显存字节与类型统计，并下载 LM Head 对照原始字节。
这些用例验证设备存储与 CPU 前向兼容，没有执行 GPU 前向。
整模型重新量化已有库实现，但本阶段没有新增针对该接口的独立专项测试。

### 回归结果

以下为该历史阶段已完成的回归记录；本次文档整理未重复运行推理测试。

| 配置 | 注册 | 执行通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: | ---: |
| CPU 全量 | 169 | 157 | 12 | 0 |
| CUDA 单 GPU 全量 | 169 | 166 | 3 | 0 |
| CPU Runtime 专项 | 13 | 5 | 8 | 0 |
| CUDA Runtime 专项 | 13 | 11 | 2 | 0 |
| CPU Tensor 专项 | 5 | 3 | 2 | 0 |
| CUDA Tensor 专项 | 5 | 4 | 1 | 0 |

CPU 全量跳过 8 项 Runtime、2 项 Tensor 与 2 项模型上传用例。
CUDA 单 GPU 全量跳过 2 项 CPU stub 和 1 项多 GPU 用例。
GPU 查询成功后的显存或传输失败仍报告为测试失败。

验证设备为 NVIDIA GeForce RTX 4060 Laptop GPU，SDK 为 CUDA 12.9。
CUDA 结果来自允许访问 GPU 的沙箱外运行；沙箱内 Runtime 查询失败产生的
跳过结果不能代替真实 GPU 验证。临时 SDK 与构建路径见
[CLI 同步说明](cli_integration_status.md#回归验证)。

多线程 CLI 专项已验证线程配置 1/4/0 的 9 份 logits、生成 ID 与 bench
输出一致。同期修复 `docs` 中 55 处 Markdown 警告，保留默认 lint 规则。

## 后续实现边界

- CUDA Linear、RMSNorm、SwiGLU、RoPE 等计算算子尚未接入。
- Attention、KV Cache、采样和 token 编排仍使用 CPU，尚无设备驻留 decode。
- GPU 上传只支持 F32/Q8_0，Q4_0/Q4_1 设备存储与计算需另行设计。
- 模型上传和重新量化仍为库接口，尚无对应 CLI 选项。
- 运行统计没有计算路径更新，也没有并发保护；多 GPU 与性能基准未验证。
- 后续接入 GPU 前向时需明确后端选择，并新增真实 CPU/GPU logits 对齐测试。

[tensor-h]: ../include/mini_llama/cuda_tensor.h
[tensor-cc]: ../src/cuda_tensor.cc
[runtime-h]: ../include/mini_llama/cuda_runtime.h
[model-h]: ../include/mini_llama/model.h
[model-cc]: ../src/model.cc
[forward-cc]: ../src/forward.cc
[tensor-test]: ../tests/tensor_test.cc
[forward-test]: ../tests/forward_test.cc
[cmake]: ../CMakeLists.txt
[test-cmake]: ../tests/CMakeLists.txt
