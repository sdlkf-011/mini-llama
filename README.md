# mini-llama.cpp

一个使用 C++17 实现的 LLaMA-style 自回归推理项目，支持 CPU 和可选 NVIDIA CUDA
后端。项目自行实现张量、算子、模型加载、分词、Transformer 前向、KV 缓存和采样，
通过 `mini-llama` 提供单次生成、终端聊天、模型检查和基准测试。

核心推理在 C++ 中执行，不依赖 PyTorch 或 llama.cpp 作为运行时。Python 脚本用于
准备模型、导出分词资源、生成数值参考和执行回归验证。以下说明对应 2026-10-12
的实现与全项目检查结果。

## 打开终端聊天

打开系统终端，或在 VS Code 中选择「终端 → 新建终端」。当前机器的项目路径是
`/home/robin/projects/mini-llama`；在其他机器上，将下面的路径换成实际项目目录。

已经完成 CPU 构建，并在 `models/chat/` 放入兼容的 GGUF 聊天模型后，执行：

```bash
cd /home/robin/projects/mini-llama
./build/mini-llama run models/chat -n 64
```

已经完成 CUDA 构建时，使用 GPU 执行：

```bash
cd /home/robin/projects/mini-llama
./build-cuda/mini-llama run models/chat --backend cuda -n 64
```

看到 `mini-llama >` 后输入消息，按回车发送。`-n 64` 表示每轮最多生成 64 个 token，
模型遇到 EOS 或用完剩余上下文窗口时会提前结束本轮。

| 聊天命令 | 作用 |
| --- | --- |
| `/help` | 查看内置命令 |
| `/clear` | 清空消息、前缀索引和会话统计，重建推理上下文 |
| `/stats` | 查看会话累计 token 数、耗时和吞吐 |
| `/params` | 查看当前采样参数 |
| `/exit` | 退出聊天 |

首次使用请先完成[构建与测试](#构建与测试)和[模型准备](#模型准备)。
`models/tiny` 是未经训练的教学模型，可用于验证 CLI 和计算链路：

```bash
./build/mini-llama run models/tiny -n 8
```

它的生成结果不代表真实聊天能力。

## 构建与测试

项目通过 CMake 构建 `mini_llama_core` 核心库、`mini-llama` 可执行程序和各模块测试。
C++ 标准为 C++17，CMakeLists.txt 声明最低 CMake 版本为 3.17。

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `MINI_LLAMA_CUDA` | `OFF` | 编译 CUDA 包装、kernel 并链接 CUDA Runtime/cuBLAS |
| `MINI_LLAMA_LIBEDIT` | `AUTO` | 自动检测可选的 libedit 行编辑器；`ON` 要求可用，`OFF` 禁用 |
| `BUILD_TESTING` | `ON` | 构建 GoogleTest 测试并注册 CTest/Python 回归 |
| `CMAKE_CUDA_ARCHITECTURES` | 由构建环境决定 | 指定目标 GPU 架构，使用此选项需 CMake 3.18 或更新版本 |

启用测试需要 GoogleTest 和 Python 3。CMake 优先查找本机 GoogleTest，找不到时
通过 FetchContent 获取 v1.15.2。NumPy 用于 golden logits 回归，缺少时该项明确跳过。
只构建推理程序可添加 `-DBUILD_TESTING=OFF`。

libedit 是 Linux/macOS 交互聊天的可选依赖，用于 UTF-8 中文退格、光标移动和
跨行重绘。默认 `AUTO` 在找到开发文件时启用；缺少时给出配置警告，仍可正常
构建，交互输入回退到 `std::getline`，中文退格的显示问题仍可能出现。
Ubuntu/Debian 可运行 `sudo apt-get install libedit-dev` 安装开发文件。
`-DMINI_LLAMA_LIBEDIT=ON` 要求 libedit 可用，缺少依赖或平台不支持时配置失败；
`-DMINI_LLAMA_LIBEDIT=OFF` 完全禁用其查找和链接。

管道、重定向输入及注入的流始终使用 `std::getline`。启用行编辑器时终端需设置
有效的 `TERM` 和 UTF-8 locale（如 `LANG=C.UTF-8`）；读取时只调整当前线程的
字符 locale，环境不是 UTF-8 时尝试使用 `C.UTF-8` 或 `en_US.UTF-8`。
`terminal-input` 伪终端回归只在启用 libedit 的构建中注册。

### CPU

需要支持 C++17 的编译器和 CMake，无需 CUDA Toolkit：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4
./build/mini-llama --help
```

### CUDA

需要 NVIDIA GPU、兼容驱动和完整 CUDA Toolkit，包括 CUDA 编译器、头文件、
libdevice、Runtime 和 cuBLAS。下面的 `89` 对应本机 RTX 4060 Laptop GPU，
其他设备请按目标 GPU 设置：

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j4
./build-cuda/mini-llama --cuda-info
ctest --test-dir build-cuda --output-on-failure -j1
```

本机另已验证 CUDA 12.3 与 Clang 18。使用该工具链且 Toolkit 安装在对应用户目录时，
可显式指定路径与编译器：

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="$HOME/.local/share/cuda-12.3" \
  -DCMAKE_CUDA_COMPILER=clang++-18 \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

Toolkit 路径应指向实际安装目录，建议使用系统目录或固定用户目录。更换机器、SDK
或编译器时应重新配置构建目录；CMake 3.24 及更新版本可使用 `--fresh` 清除旧配置。
无需测试的部署构建可添加 `-DBUILD_TESTING=OFF`；运行 CUDA 程序还需要驱动及
CUDA Runtime/cuBLAS 动态库。

CMake 自动生成 `compile_commands.json`。CPU 构建数据库不包含 `.cu` 编译命令；
CPU/CUDA 分别构建时，将编辑器的 clangd 编译数据库目录设为 `build-cuda/`：

```bash
clangd --check=src/cuda_ops.cu --compile-commands-dir=build-cuda
```

更多工具链说明见[后端与 CUDA Runtime](docs/backend_selection_cuda_runtime_implementation.md#clangd-的-cuda-文件诊断)。

## 模型准备

| 模型来源 | 文件与用途 |
| --- | --- |
| `models/tiny/` | `model.json`、`model.bin` 和词表等教学资产，用于计算与 CLI 验证 |
| `models/chat/` | 本地准备的兼容 GGUF 聊天模型；`*.gguf` 不随仓库提交 |
| 指定文件或目录 | `--model` 和 `run/bench` 的位置参数均可使用其他模型路径 |

当前机器使用 `models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf` 完成真实模型验证。
在新环境中，需先将兼容 GGUF 放入 `models/chat/`，再执行前面的聊天命令。
目录中有多个 GGUF 时，请明确指定文件：

```bash
./build/mini-llama run models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf -n 64
```

模型目录中完整的 `model.json + model.bin` 优先使用；缺少其中任一文件时查找
唯一 GGUF 候选。格式按文件魔数判断，多候选不会自动挑选其中一个。

JSON+BIN 加载路径要求 float32 权重，并校验配置、名称、形状、偏移、重叠和文件长度。
GGUF Reader 支持 v3，模型加载支持 F32、F16、Q8_0、Q4_0、Q4_1。F16 转为 F32；
支持的量化 Linear 保留块格式，embedding、norm 和 bias 使用 F32。

CLI 优先采用显式 `--tokenizer`；GGUF 默认读取内嵌分词器，失败时尝试同目录外部资源。
JSON 模型按清单选择 ASCII 或 JSON 词表，清单相对路径以配置目录为基准。
显式外部分词资源加载失败会报错，JSON 清单中的 BOS/EOS/UNK ID 须与分词器一致。

需要重新生成教学模型时，可输出到独立目录：

```bash
python3 scripts/export_tiny_model.py --output-dir /tmp/mini-llama-demo --seed 42
python3 scripts/test_mvp.py --binary build/mini-llama --model /tmp/mini-llama-demo
```

模型下载、词表导出及配置升级工具位于 `scripts/`，使用前可查看各脚本的 `--help`。

## 常用 CLI 命令

以下命令在项目根目录执行；CUDA 推理使用 `build-cuda/mini-llama` 并加
`--backend cuda`。

| 命令 | 当前行为 |
| --- | --- |
| `generate` | 单次 prompt 生成，输出 token、文本与请求追踪；支持逐步 logits 导出 |
| `run <path>` | 终端多轮聊天，维护消息与当前上下文的 KV 历史 |
| `bench <path>` | 预热、重复测量 prefill/decode，报告吞吐、权重大小及 CUDA 指标 |
| `inspect <path>` | 检查 JSON 清单或 GGUF 元数据，不加载权重或查询 GPU |
| `inspect-gguf <file>` | 查看 GGUF 头、元数据和张量目录 |
| `--cuda-info` | 查询 CUDA 构建状态和设备信息 |
| `--help` / `<command> --help` | 查看全局或命令帮助 |

```bash
./build/mini-llama generate --model models/tiny -p hello -n 8 --threads 4
./build/mini-llama bench models/tiny -p hello -n 8 --threads 4 --warmup 1 --repeat 3
./build/mini-llama inspect models/tiny
./build/mini-llama inspect-gguf models/tiny/model.gguf
./build/mini-llama run --help
```

常用参数：

| 参数 | 适用命令 | 含义 |
| --- | --- | --- |
| `--backend cpu\|cuda` | generate/run/bench/inspect | 默认 CPU；CUDA 推理需 CUDA 构建及可用设备 |
| `--device ID` | generate/run/bench/inspect | CUDA 设备编号，默认 0，仅与 CUDA 后端配合 |
| `--threads N` | generate/run/bench | CPU 并行配置，0 为自动模式 |
| `-p` / `--prompt` | generate/run/bench | prompt；run 中作为首轮用户消息 |
| `-n` / `--n-predict` | generate/run/bench | 生成预算；run 默认 64，其他两者默认 16 |
| `--temperature`、`--top-k`、`--seed` | generate/run | 控制采样，默认温度 0 为 greedy；seed 控制随机采样 |
| `--seed` | bench | 保留采样种子接口，当前 benchmark 使用默认 greedy |
| `--quant q8_0\|q4_0` | generate/run/bench | 重量化所有 Linear，embedding/norm/bias 保持 F32 |
| `--dump-logits DIR` | generate | 导出逐位置 logits 和 `generation_tokens.txt` |
| `--warmup N`、`--repeat N`、`--verbose` | bench | 预热次数、测量次数和详细输出；默认 1 次预热、3 次测量 |

`generate/bench -n 0` 只执行 prefill；聊天要求正数响应预算。单次 generate 会检查
完整预算是否超过上下文窗口，run 和 bench 则按剩余窗口截断。Q4_1 可从 GGUF
加载并推理，但不是 `--quant` 的重量化目标。

## 当前实现

模型、上下文与 CLI 编排分工如下：

```text
输入文本 / CLI 参数
        ↓
模型加载 + Tokenizer + PromptBuilder（聊天）
        ↓
MiniBatch → ForwardBatch → ForwardToken
        ↓
Embedding → Transformer 各层（读写 KV）→ Final RMSNorm → LM Head
        ↓
CPU MiniSampler → Tokenizer 解码 → 终端输出
```

`ForwardBatch()` 顺序处理各 token，每步设置 position、调用原有 `ForwardToken()`
并记录实际计算历史，返回最后一份 logits。prefill 和 decode 共用这条路径，
每个 prefill token 也执行最终投影；CUDA 模式每步下载一份 logits。
生成 N 个 token 通常只需 N−1 次 decode，首个 token 直接从 prefill logits 采样。
聊天为下一轮复用，会将每个非 EOS 的生成 token 继续送入前向；EOS 不写入 KV。

| 模块 | 已实现能力 | 主要源码 |
| --- | --- | --- |
| 模型与加载 | 配置、权重、字节统计；JSON+BIN、GGUF；Qwen2 NeoX RoPE、可选 Q/K/V bias | [model.cc](src/model.cc)、[loader.cc](src/loader.cc)、[gguf_loader.cc](src/gguf_loader.cc) |
| 分词与模板 | ASCII、JSON 最长匹配、外部/内嵌 GPT-2 byte-level BPE；Plain/Qwen2/Jinja 子集 | [tokenizer.h](include/mini_llama/tokenizer.h)、[gguf_tokenizer.cc](src/gguf_tokenizer.cc)、[prompt_builder.cc](src/prompt_builder.cc) |
| 推理与状态 | GQA、因果 attention、RMSNorm、RoPE、残差、SwiGLU；CPU/GPU F32 KV 缓存 | [forward.cc](src/forward.cc)、[context.h](include/mini_llama/context.h)、[kv_cache.cc](src/kv_cache.cc) |
| CPU 算子 | F32/Q8_0/Q4_0/Q4_1 Linear、基础算子、线程分块和按编译目标选择的 SIMD | [ops.cc](src/ops.cc)、[quant.cc](src/quant.cc)、[matmul_optimized.cc](src/matmul_optimized.cc)、[thread_pool.cc](src/thread_pool.cc) |
| CUDA 后端 | 权重上传、cuBLAS F32 Linear、量化 Linear kernel、激活、RoPE、GPU attention/KV | [cuda_matmul.cc](src/cuda_matmul.cc)、[cuda_quant.cu](src/cuda_quant.cu)、[cuda_ops.cu](src/cuda_ops.cu)、[cuda_attention.cu](src/cuda_attention.cu) |
| 会话与输出 | greedy/temperature/top-k；消息历史、流式解码和当前 KV 中的前缀复用 | [sampler.cc](src/sampler.cc)、[chat.cc](src/chat.cc)、[radix_tree.cc](src/radix_tree.cc)、[terminal.cc](src/terminal.cc) |
| 应用与观测 | 命令解析、生成/聊天/检查/benchmark、请求 trace、耗时与 token 统计 | [main.cc](src/main.cc)、[request_context.cc](src/request_context.cc)、[debug.cc](src/debug.cc) |

### CPU 与 CUDA 执行方式

CPU 的 F32 Linear 点积按编译目标使用 AVX2+FMA、ARM NEON 或 scalar；通用 Matmul
的 SIMD 模式仍使用行并行的标量实现。量化 Linear 直接计算量化块，Q8_0/Q4_0
支持 NEON 点积，Q4_1 点积为 scalar。常驻线程池按需增长，小任务与工作线程中的
嵌套调用串行执行，回调异常在本批任务全部完成后传播。

CUDA 模式上传权重后，由 `model.cuda_weights` 决定设备前向分支。权重、层间激活
及 K/V 在设备侧计算，最终 logits 回到 CPU 采样，文本解码与终端输出也在 CPU。
正常主路径不上传激活、不回退到 CPU attention；缺失或损坏设备权重明确报错。
CPU 权重仍保留，Context 仍分配 CPU KV，GPU KV 在首次设备前向时分配且不镜像
更新 CPU KV。

CUDA Tensor/Buffer 独占显存。cuBLAS 句柄、RMSNorm 标量和 attention 分数工作区
在现有算子内按线程、设备复用；超过 64 MiB 的分数工作区使用临时分配。
内置算子使用默认 stream，共享模型快照的运行计数采用单调用线程约定。

### 状态与校验

上下文须绑定传入的同一个模型对象；更换模型、实际权重或后端后，应新建上下文并
重放历史。模型生命周期须覆盖引用它的上下文，手工修改 CPU 权重后还需重新上传
CUDA 权重。批次在计算前检查所有 token/position 范围，连续位置构造拒绝整数溢出。
CPU KV 访问检查缓存形状和存储大小，写入前核对 K/V 及输入数据长度；F32 权重
转换遇到错误数据长度会报错。底层 Tensor 的 shape/data 为公开字段，调用方须
维持二者一致。

RadixTree 只保存 token 前缀索引，实际 KV 位于当前 Context。聊天复用长度还要与
当前计算历史的公共前缀求交集，重放时覆盖旧尾部；完整命中时重算最后一个 token
以取得 logits。

### Benchmark 与统计口径

CLI 默认预热一次、测量三次，每次调用创建独立上下文；预热不计入结果。当前
`RunBenchmark()` 的 prefill 时间包含批次构造与前向，decode 时间包含循环内的
前向、采样和 verbose 输出，首次采样不计时。总时间为 prefill + decode，报告
均值及 min/median/max；正式性能比较应关闭 `--verbose`。

权重压缩比例为 `F32 等价字节数 / 实际字节数`；`bench --quant` 比较 prompt
最后位置的 logits 与模型原始格式，不等同于全 F32 golden 或生成质量评估。
CUDA 摘要中权重显存不包括 KV/临时激活，KV 读写是逻辑字节数而非硬件带宽测量。
generate 的运行计数覆盖当前请求，bench 对应最后一次正式测量，run 跨轮累计；
`/clear` 不清零模型 CUDA 计数。

## 验证结果与架构边界

2026-10-12 在本机 Release 构建上执行全量 CTest，环境包括 GCC 13、CUDA 12.3 /
Clang 18 和 RTX 4060 Laptop GPU：

| 配置 | 注册 | 实际通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 236 | 180 | 56 | 0 |
| CUDA，真实单 GPU，串行测试 | 236 | 228 | 8 | 0 |

CPU 跳过 CUDA 用例；CUDA 跳过 7 项 CPU stub 用例和 1 项多 GPU 用例。
回归覆盖模块单元测试、CLI、量化、线程池、CPU/CUDA 数值对照、真实模型聊天和
NumPy golden；跳过项不计为执行通过。最新范围与修复记录见
[全项目检查](docs/cli_integration_status.md#2026-10-12-全项目检查)。

当前架构以单进程、单条上下文的教学推理和终端交互为主：

- `MiniBatch` 仍逐 token 执行，不提供多序列并行或矩阵化 prompt prefill。
- F32 Linear 支持 `[in]` / `[1, in]`；Q8_0/Q4_0/Q4_1 Linear 支持 `[in]` / `[batch, in]`。
- GPU 后端支持单机单卡；没有多 GPU 推理、服务端队列、continuous batching、FlashAttention 或计算图编译器。
- KV 使用 F32，权重量化不会同时量化 KV。
- GGUF 模型适配面向现有 Llama/Qwen2，不支持 K-quants/IQ-quants 等扩展格式。
- 分词不是通用 SentencePiece/unigram 实现，聊天模板也不是完整 Jinja2 引擎。
- 当前实际验证为单 GPU；ARM NEON 运行验证和 compute-sanitizer 设备检查尚未完成。

## 目录与模块文档

```text
include/mini_llama/   模块类型与函数声明
src/                 C++ 实现和 CUDA kernel
tests/               GoogleTest 模块测试及 CMake 注册
scripts/             模型准备、数值参考、CLI/CUDA 回归与报告
models/tiny/         教学模型资产
models/chat/         本地 GGUF 聊天模型
docs/                模块说明、阶段记录和验证结果
build/               CPU 构建产物
build-cuda/          CUDA 构建产物
```

下表按主题组织模块说明。阶段文档中的旧测试数字、性能和接入状态保留历史含义；
当前入口与最新验证以本 README 和 CLI 同步说明为准。

| 主题 | 文档 |
| --- | --- |
| 当前接入与检查 | [CLI 同步说明](docs/cli_integration_status.md)、[实现调整与验证](docs/inference_improvements.md) |
| 整体链路阅读 | [模型文件到生成](docs/first_major_module_review.md)、[真实 GGUF 接入](docs/second_major_module_review.md) |
| CPU 计算与状态 | [Tensor/算子](docs/tensor_ops_implementation.md)、[线程与 SIMD](docs/cpu_threaded_inference_implementation.md)、[量化 Linear](docs/simd_quantized_linear_implementation.md)、[KV/Context](docs/kv_cache_context_implementation.md) |
| 模型文件 | [JSON Loader](docs/model_formats_and_loader_branch_summary.md)、[GGUF Reader](docs/gguf_implementation.md)、[GGUF Loader](docs/gguf_loader_branch_summary.md) |
| 分词、采样与聊天 | [Tokenizer](docs/tokenizer_implementation.md)、[GGUF Tokenizer](docs/gguf_tokenizer_integration_implementation.md)、[Sampler](docs/sampler_sampling_strategy_implementation.md)、[模板与聊天](docs/chat_prompt_run_implementation.md) |
| CUDA 路径 | [后端/Runtime](docs/backend_selection_cuda_runtime_implementation.md)、[设备前向](docs/device_resident_cuda_decode.md)、[KV/Attention](docs/cuda_kv_cache_implementation.md) |
| CUDA 阶段记录 | [Tensor/上传](docs/cuda_tensor_model_weights_implementation.md)、[cuBLAS](docs/cublas_f32_linear_implementation.md)、[Ops/量化](docs/cuda_ops_quant_implementation.md) |
| 数值与验收 | [Golden logits](docs/golden_logits_testing.md)、[调试/基准](docs/benchmark_debug_implementation.md)、[Benchmark 报告](docs/cuda_benchmark_validation.md)、[CUDA 正确性](docs/cuda_correctness_validation.md)、[Qwen2 smoke](docs/qwen2_gguf_end_to_end_smoke_test.md) |

常用验证工具：`test_mvp.py`、`test_quant_cli.py`、`test_golden.py`、
`run_cuda_benchmarks.py`、`test_cuda_asan.sh`。`compare_logs.py` 比较 C++ 与 Python
logits 导出，参数为 `--cpp-dir` 和 `--python-dir`。
CUDA ASan/UBSan 脚本检查主机侧内存；`--device-memcheck` 使用 compute-sanitizer，
缺少工具或必需 GPU 用例未执行时返回 77，失败返回非零；历史 sanitizer 记录见
[实现调整与验证](docs/inference_improvements.md#验证与实测)。
