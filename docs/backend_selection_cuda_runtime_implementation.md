# 后端选择与 CUDA Runtime 阶段实现汇总

本阶段完成统一的后端配置、CLI 参数接入以及 CUDA Runtime 基础封装。
CPU 构建无需 CUDA Toolkit，CUDA 构建支持真实设备查询、设备选择、显存分配与拷贝。

CUDA 推理算子、模型权重上传、GPU Attention、GPU KV Cache 和设备驻留 Decode
属于后续模块，当前 `generate`、`run`、`bench` 的实际计算路径仍为 CPU。
Runtime 可用时请求 CUDA 推理会明确报错，不静默落入 CPU 路径。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| [backend.h](../include/mini_llama/backend.h) | 后端枚举、统一配置、解析、校验与执行说明接口 |
| [backend.cc](../src/backend.cc) | 后端名称解析、Runtime 查询转发和运行前校验 |
| [cuda_runtime.h](../include/mini_llama/cuda_runtime.h) | 设备信息、拷贝方向、Runtime 接口及显存 Buffer |
| [cuda_runtime.cc](../src/cuda_runtime.cc) | CUDA API 封装、CPU Stub、版本格式化和 RAII 所有权管理 |
| [main.cc](../src/main.cc) | 参数解析、命令分发、设备诊断及推理接入边界 |
| [cuda_runtime_test.cc](../tests/cuda_runtime_test.cc) | 13 项 Runtime 测试与编译期接口约束检查 |
| [test_backend_cli.py](../scripts/test_backend_cli.py) | 标准库实现的 CLI 参数和兼容性测试 |
| [CMakeLists.txt](../CMakeLists.txt)、[tests/CMakeLists.txt](../tests/CMakeLists.txt) | 可选 CUDA 依赖、编译宏及 CTest 注册 |

## 后端配置与 CLI 参数

```cpp
enum class BackendKind {
  kCpu,
  kCuda,
};

struct BackendConfig {
  BackendKind kind = BackendKind::kCpu;
  int device_id = 0;
  bool device_id_set = false;
};
```

`ParseBackendKind(text, kind)` 仅接受小写 `cpu` 和 `cuda`。
成功返回 `true` 并更新 `kind`，失败返回 `false` 并保留原值。
`BackendKindName()` 将已知枚举转换为名称，未知值返回 `unknown`。
`ValidateBackend()` 会拒绝未知后端，名称转换的结果不能替代校验。

`generate`、`run`、`bench`、`inspect` 统一接受 `--backend cpu|cuda` 和 `--device ID`。
默认使用 CPU，设备编号接受 `0..INT_MAX`，拒绝负数、加号、空白、尾随字符和溢出。
两个选项均禁止重复，包括重复相同值，参数顺序不影响最终配置。
CLI 的 `ParseBackendArg()`、`ParseDeviceArg()` 位于入口内部，复用公共解析和严格整数转换。

`device_id_set` 表示是否显式传入 `--device`，不能依据编号是否为零判断。
因此 `--backend cpu --device 0` 同样报错，CPU 配置未显式指定设备时直接通过校验。

`inspect` 只读取元数据，校验参数语法和 CPU/设备参数冲突，不查询 GPU 或上传权重。
即使在 CPU 构建中，也允许 `inspect --backend cuda --device ID` 检查计划用于 CUDA 的模型。
输出明确说明 `compute: none (metadata only; ...)`。
`inspect-gguf` 保留原有独立参数，`--cuda-info` 是独立顶层诊断命令。

```bash
./build/mini-llama generate models/tiny --backend cpu --prompt hello --n-predict 4
./build/mini-llama bench models/tiny --backend cpu --n-predict 4 --verbose
./build/mini-llama run models/tiny --backend cpu
./build/mini-llama inspect models/tiny --backend cuda --device 0
```

## 校验与执行路径

`CudaBackendBuilt()` 复用 `CudaRuntimeBuilt()`，只表示二进制启用了 CUDA Runtime。
`CudaBackendAvailable()` 查询设备数量并检查编号范围，查询异常转为 `false`。
`CudaDeviceSummary()` 复用属性查询和格式化，失败时返回具体错误原因。
可见设备编号有效仍不保证后续上下文初始化、分配或拷贝成功，这些操作独立检查错误。

```text
推理命令
  ParseOptions -> BackendConfig -> ValidateInferenceBackend
    CPU  -> ValidateBackend -> 模型与分词器加载 -> 原有 CPU 推理链路
    CUDA -> ValidateBackend -> 构建与设备检查
         -> 当前阶段报 CUDA inference is not integrated yet

设备诊断
  --cuda-info -> RunCudaInfo -> CudaDeviceCount
              -> CudaGetDeviceInfo -> CudaFormatDeviceInfo

元数据检查
  inspect -> 参数校验 -> 元数据读取 -> compute: none
```

`BackendExecutionNote()` 只描述路径，不查询或初始化设备。
CPU 返回 `compute: cpu`，CUDA 当前返回推理路径尚未接入的说明。
CLI 另行输出 `backend: cpu` 等后端名称，请求追踪复用配置中的后端名称。

后续 CUDA 推理模块接入时，需要替换 `ValidateInferenceBackend()` 中的阶段性报错，
完成权重上传和实际算子分发，并依据真实迁移情况更新执行说明。
本阶段没有新增 CUDA 算子或修改既有 CPU 算子实现。

## Runtime 查询、拷贝与错误处理

公共头文件只包含标准库类型，CUDA SDK 头文件和类型仅出现在实现文件的
`MINI_LLAMA_USE_CUDA` 条件编译分支中。

| 接口 | 行为 |
| --- | --- |
| `CudaRuntimeBuilt()` | 检查编译宏，不查询 GPU，不抛异常 |
| `CudaDeviceCount()` | 返回真实设备数，无设备返回 0，其他 CUDA 查询错误抛异常 |
| `CudaSetDevice(device_id)` | 校验编号并选择调用线程的当前 CUDA 设备 |
| `CudaGetDeviceInfo(device_id)` | 校验编号，查询名称、Compute Capability、显存及版本 |
| `CudaFormatDeviceInfo(info)` | 纯文本转换，不访问 GPU，在 CPU 构建中也可使用 |
| `CudaMallocBytes(bytes)` | 在当前设备分配显存，零字节返回 `nullptr` |
| `CudaFreeBytes(ptr)` | 释放显存，空指针直接返回 |
| `CudaMemcpyBytes(dst, src, bytes, kind)` | 检查非空指针和方向，调用 CUDA 拷贝接口，零长度直接返回 |

`CudaMemcpyKind` 支持 Host → Device、Device → Host 和 Device → Device。
内部的 `ToCudaMemcpyKind()` 将其映射到 SDK 枚举，未知方向抛异常。
原始显存接口使用当前设备，调用方负责设备选择和分配结果的释放。

CPU Stub 的设备查询、设备选择和非零显存操作统一抛出 `std::runtime_error`。
零字节分配、零长度拷贝和空指针释放没有 CUDA 操作，在两种构建中均安全。

```text
CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.
```

`CudaNotBuiltError()` 统一构造 CPU Stub 错误。
CUDA 分支通过 `CheckCuda()` 检查 API 返回码，错误包含被调用 API 和底层原因。
只有 `cudaErrorNoDevice` 被转换为零设备，驱动或初始化错误不会被隐去。

```text
CUDA runtime error in cudaMalloc: <cudaGetErrorString 返回的原因>
```

显存按 bytes 和 GiB 同时显示，GiB 使用 `1024 * 1024 * 1024` 换算。
版本按 `major * 1000 + minor * 10` 解码，例如 `12090` 显示为 `12.9 (12090)`。
`driver_version` 表示驱动支持的 CUDA API 版本，来自 `cudaDriverGetVersion()`。
版本编码依据见 [NVIDIA Runtime 版本接口说明](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html)。

## CudaDeviceBuffer 所有权与异常安全

Buffer 保存显存指针、字节容量及所属设备，禁止拷贝，支持 `noexcept` 移动构造和移动赋值。
`data()` 提供可变和 const 重载，`bytes()`、`device_id()`、`empty()` 提供状态访问。
默认对象的指针为空、容量为 0、设备编号为 0。

| 操作 | 所有权与状态 |
| --- | --- |
| 构造 | 复用 `Reset(bytes, device_id)` 分配，失败时不发布新资源 |
| 移动构造 | 转移指针、容量和设备编号，源对象清空并恢复设备编号 0 |
| 移动赋值 | 自移动保持状态，普通移动释放目标旧资源后接管源对象 |
| `Reset()` | 在所属设备释放，成功后清空状态，释放失败抛异常并保留所有权 |
| `Reset(bytes, device_id)` | 先分配替换对象再释放旧资源，分配失败保留旧 Buffer |
| `Reset(0, device_id)` | 释放旧资源，保留传入设备编号，不创建新的 GPU 分配 |
| `Upload()` / `Download()` | 检查容量，零长度返回，在所属设备执行拷贝 |
| 析构 / `ReleaseNoexcept()` | 捕获释放异常并记录 stderr 诊断，不向外抛异常 |

移动后的源对象仍可重新分配，自移动和接收空对象均有测试覆盖。
替换对象通过 RAII 负责分配后的临时所有权，避免原资源释放失败时遗失新分配。
私有 `DeviceGuard` 保存调用线程的原设备，按需要切换并在操作结束后恢复。
恢复失败记录诊断，避免在异常展开期间再次抛异常。

## 构建与复现

`MINI_LLAMA_CUDA` 默认关闭，仅 CUDA 构建调用 `find_package(CUDAToolkit REQUIRED)`。
核心目标链接 `CUDA::cudart`，并向关联目标传递 `MINI_LLAMA_USE_CUDA=1`。
本阶段实现是 C++17 Runtime API 封装，没有 `.cu` 文件或 CUDA kernel 编译。

CPU 构建及专项测试：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j4
ctest --test-dir build -R cuda_runtime --output-on-failure
ctest --test-dir build -R backend-cli-smoke --output-on-failure
ctest --test-dir build --output-on-failure -j4
```

已安装 CUDA Toolkit 的环境：

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON
cmake --build build-cuda -j4
ctest --test-dir build-cuda -R cuda_runtime --output-on-failure
ctest --test-dir build-cuda --output-on-failure -j4
./build-cuda/mini-llama --cuda-info
```

本机没有系统 CUDA Toolkit，验证时将 NVIDIA 官方 Runtime 12.9 与配套 CRT 头文件
放在 `/tmp/mini-llama-cuda-sdk`，通过标准 `CUDAToolkit_ROOT` 参数完成 CUDA 构建。
googletest 复用现有 CPU 构建缓存，未新增测试框架。

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON \
  -DCUDAToolkit_ROOT=/tmp/mini-llama-cuda-sdk \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$PWD/build/_deps/googletest-src"
```

该路径是本机临时验证配置，没有写入项目 CMake。
临时目录清理后应重新配置正式 SDK 路径，当前 CUDA 可执行文件也依赖该目录下的 libcudart。
沙箱内驱动查询曾报不可用，实际 GPU 验证在允许访问 GPU 的沙箱外执行。

## 验证结果与限制

本阶段验证日期为 2026-10-08，实际设备为 NVIDIA GeForce RTX 4060 Laptop GPU。
Runtime 专项固定注册 `cuda_runtime.` 前缀，便于使用 CTest 的 `-R cuda_runtime` 筛选。

| 验证项目 | 执行通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: |
| CPU Runtime 专项，共 13 项 | 5 | 8 | 0 |
| CUDA Runtime 专项，真实单 GPU，共 13 项 | 11 | 2 | 0 |
| CUDA Runtime 专项，隐藏 GPU，共 13 项 | 5 | 8 | 0 |
| CPU 全量 CTest，共 158 项 | 150 | 8 | 0 |
| CUDA 全量 CTest，真实单 GPU，共 158 项 | 156 | 2 | 0 |

CPU 专项跳过需要 CUDA 构建的参数校验和 7 项 GPU 测试。
CUDA 单 GPU 专项跳过 CPU Stub 专项和要求两张 GPU 的测试。
GPU 不可用时，前置检查输出实际跳过原因，查询成功后的显存操作失败仍报告为测试失败。
格式化测试使用给定结构体数据，只验证文本转换，不计作实际设备查询验证。

真实 GPU 测试已执行设备查询与选择、Host/Device 和 Device/Device 拷贝、容量边界、
移动所有权、Reset 失败后原数据保持可读，以及异常展开中的 Buffer 生命周期。
隐藏设备场景通过以下命令验证零设备诊断和跳过逻辑：

```bash
CUDA_VISIBLE_DEVICES=-1 ctest --test-dir build-cuda -R cuda_runtime --output-on-failure
```

CLI smoke 使用 Python 标准库，共 13 项测试，CPU 与 CUDA 单 GPU 构建均为 9 项通过、4 项跳过。
CUDA 构建中的推理测试因推理模块尚未接入而明确跳过，不能据此报告 GPU 推理通过。
既有线程 CLI 脚本也已验证线程数 1/4/0、9 份 logits 及生成 ID 一致。

CPU 构建的 `--cuda-info` 返回退出码 1，并输出规定错误：

```text
CUDA backend built: no
CUDA info failed: CUDA backend was not built. Reconfigure with -DMINI_LLAMA_CUDA=ON on a NVIDIA CUDA machine.
```

CUDA 构建的实际诊断返回退出码 0：

```text
CUDA backend built: yes
CUDA GPU count: 1
GPU 0: NVIDIA GeForce RTX 4060 Laptop GPU; Compute Capability: 8.9; memory bytes: 8585216000; memory GiB: 7.996; Driver Version (CUDA API): 13.0 (13000); CUDA Runtime Version: 12.9 (12090)
```

多 GPU 场景仍未验证，驱动失效导致的释放和设备恢复失败尚未做故障注入。
CUDA 推理算子、模型权重上传以及真实 CPU/CUDA logits 对齐属于后续阶段。
后续实现应复用本阶段的 Runtime 和 Buffer 接口，并在实际算子分发接入后更新路径说明与测试。
