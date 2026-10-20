# mini-llama.cpp

一个使用 C++17 实现的 LLaMA-style 自回归推理项目，支持 CPU 和可选 NVIDIA CUDA
后端。项目自行实现张量、算子、模型加载、分词、Transformer 前向、KV 缓存和采样，
通过 `mini-llama` 提供单次生成、终端聊天、模型检查和基准测试。

核心推理在 C++ 中执行，不依赖 PyTorch 或 llama.cpp 作为运行时。Python 脚本用于
准备模型、导出分词资源、生成数值参考和执行回归验证。项目主要用于学习和研究
推理引擎的实现，支持的模型与运行方式见[功能与限制](#功能与限制)。

## 快速开始

以下步骤使用支持 C++17 的编译器、Git 和 CMake 3.24 或更新版本
（CUDA 示例中的 `native` 架构选项需要该版本）。默认构建还需要 Python 3 和
GoogleTest；未找到系统中的 GoogleTest 时，CMake 会联网下载。

Ubuntu/Debian 可先安装 CPU 构建和测试依赖：

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git python3 libgtest-dev libedit-dev
cmake --version
```

若发行版提供的 CMake 版本较旧，请先升级 CMake。其他系统请使用相应包管理器
安装编译器、CMake 和 Python 3；libedit 为可选依赖。

```bash
git clone https://github.com/sdlkf-011/mini-llama.git
cd mini-llama
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/mini-llama --help
./build/mini-llama generate --model models/tiny -p hello -n 8
```

仓库自带的 `models/tiny/` 是未经训练的教学模型，用于检查程序能否加载模型并
完成推理，输出不代表真实聊天能力。要进行自然语言对话，请继续阅读
[模型准备与终端聊天](#模型准备与终端聊天)。以下命令均在仓库根目录执行。

## 构建与测试

项目通过 CMake 构建 `mini_llama_core` 核心库、`mini-llama` 可执行程序和各模块测试。

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `MINI_LLAMA_CUDA` | `OFF` | 编译 CUDA 包装、kernel 并链接 CUDA Runtime/cuBLAS |
| `MINI_LLAMA_LIBEDIT` | `AUTO` | 自动检测可选的 libedit 行编辑器；`ON` 要求可用，`OFF` 禁用 |
| `BUILD_TESTING` | `ON` | 构建 GoogleTest 测试并注册 CTest/Python 回归 |
| `CMAKE_CUDA_ARCHITECTURES` | 由构建环境决定 | 指定目标 GPU 架构，使用此选项需 CMake 3.18 或更新版本 |

启用测试需要 GoogleTest 和 Python 3。CMake 优先查找系统中的 GoogleTest，找不到时
通过 FetchContent 获取 v1.15.2。NumPy 用于 golden logits 回归，缺少时该项明确跳过。
可选安装 NumPy，例如 Ubuntu/Debian 的 `python3-numpy` 包，以运行数值参考测试。

libedit 是 Linux/macOS 交互聊天的可选依赖，用于 UTF-8 中文退格、光标移动和
跨行重绘。默认 `AUTO` 在找到开发文件时启用；缺少时给出配置警告，仍可正常
构建，交互输入回退到 `std::getline`，中文退格的显示问题仍可能出现。
Ubuntu/Debian 可运行 `sudo apt-get install libedit-dev` 安装开发文件。
`-DMINI_LLAMA_LIBEDIT=ON` 要求 libedit 可用，缺少依赖或平台不支持时配置失败；
`-DMINI_LLAMA_LIBEDIT=OFF` 完全禁用其查找和链接。

启用行编辑器时终端需设置有效的 `TERM` 和 UTF-8 locale（如 `LANG=C.UTF-8`）。
管道和重定向输入不使用行编辑器。

### CPU

快速开始中的构建无需 CUDA Toolkit，默认包含测试。构建后运行：

```bash
ctest --test-dir build --output-on-failure -j4
```

如果只需推理程序，可使用独立目录关闭测试构建，此时不需要 GoogleTest 或 Python：

```bash
cmake -S . -B build-cpu -DMINI_LLAMA_CUDA=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j4
./build-cpu/mini-llama generate --model models/tiny -p hello -n 8
```

### CUDA

需要 NVIDIA GPU、兼容驱动和完整 CUDA Toolkit，包括 CUDA 编译器、头文件、
libdevice、Runtime 和 cuBLAS。下面使用 Toolkit 自带的 `nvcc`，请先确认其在
`PATH` 中，并且驱动可以访问 GPU：

```bash
nvidia-smi
nvcc --version
```

在将要运行程序的 GPU 机器上构建：

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j4
./build-cuda/mini-llama --cuda-info
./build-cuda/mini-llama generate --model models/tiny --backend cuda -p hello -n 8
ctest --test-dir build-cuda --output-on-failure -j1
```

`native` 为构建机器可见的 GPU 生成代码；在无 GPU 的机器上编译或为其他 GPU
构建时，应将其换成目标 GPU 的架构编号。Toolkit 必须支持所选架构，主机编译器
版本也须与 Toolkit 兼容。

Toolkit 未安装在标准目录时，配置命令还需添加
`-DCUDAToolkit_ROOT=/实际的/cuda/安装目录` 和
`-DCMAKE_CUDA_COMPILER=/实际的/cuda/安装目录/bin/nvcc`，将路径替换为实际值。
使用其他 CUDA 编译器时，通过 `CMAKE_CUDA_COMPILER` 指定兼容版本。
更换工具链时请使用新的构建目录，避免沿用旧配置。
运行时还需要驱动及 CUDA Runtime/cuBLAS 动态库。

部分测试需要可用 GPU、真实聊天模型或 NumPy，缺少前提时会跳过；CPU 构建中的
CUDA 设备测试也会跳过。请结合 CTest 的跳过信息判断实际执行范围。
工具链与编辑器配置见[后端与 CUDA Runtime](docs/backend_selection_cuda_runtime_implementation.md)。

## 模型准备与终端聊天

| 模型来源 | 文件与用途 |
| --- | --- |
| `models/tiny/` | `model.json`、`model.bin` 和词表等教学资产，用于计算与 CLI 验证 |
| `models/chat/` | 可自行创建，用于存放兼容 GGUF 聊天模型；其中的 `*.gguf` 不随仓库提交 |
| 指定文件或目录 | `--model` 和 `run/bench` 的位置参数均可使用其他模型路径 |

聊天模型需自行获取。以 `Qwen2-0.5B-Instruct-Q8_0.gguf` 为例，先创建存放目录：

```bash
mkdir -p models/chat
```

将下载的 GGUF 文件放入该目录后启动聊天：

```bash
./build/mini-llama run models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf -n 64
```

已完成 CUDA 构建时，使用：

```bash
./build-cuda/mini-llama run models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf --backend cuda -n 64
```

看到 `mini-llama >` 后输入消息，按回车发送。`-n 64` 表示每轮最多生成 64 个 token，
遇到 EOS 或用完剩余上下文窗口时会提前结束。`-p` 用于指定首轮提问文本，不是
生成长度参数。

| 聊天命令 | 作用 |
| --- | --- |
| `/help` | 查看内置命令 |
| `/clear` | 清空对话历史、缓存和会话统计 |
| `/stats` | 查看会话累计 token 数、耗时和吞吐 |
| `/params` | 查看当前采样参数 |
| `/exit` | 退出聊天 |

也可将模型目录作为路径；目录中有多个 GGUF 时，请明确指定文件。
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

## 功能与限制

主要推理流程如下：

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

| 模块 | 已实现能力 | 主要源码 |
| --- | --- | --- |
| 模型与加载 | 配置、权重、字节统计；JSON+BIN、GGUF；Qwen2 NeoX RoPE、可选 Q/K/V bias | [model.cc](src/model.cc)、[loader.cc](src/loader.cc)、[gguf_loader.cc](src/gguf_loader.cc) |
| 分词与模板 | ASCII、JSON 最长匹配、外部/内嵌 GPT-2 byte-level BPE；Plain/Qwen2/Jinja 子集 | [tokenizer.h](include/mini_llama/tokenizer.h)、[gguf_tokenizer.cc](src/gguf_tokenizer.cc)、[prompt_builder.cc](src/prompt_builder.cc) |
| 推理与状态 | GQA、因果 attention、RMSNorm、RoPE、残差、SwiGLU；CPU/GPU F32 KV 缓存 | [forward.cc](src/forward.cc)、[context.h](include/mini_llama/context.h)、[kv_cache.cc](src/kv_cache.cc) |
| CPU 算子 | F32/Q8_0/Q4_0/Q4_1 Linear、基础算子、线程分块和按编译目标选择的 SIMD | [ops.cc](src/ops.cc)、[quant.cc](src/quant.cc)、[matmul_optimized.cc](src/matmul_optimized.cc)、[thread_pool.cc](src/thread_pool.cc) |
| CUDA 后端 | 权重上传、cuBLAS F32 Linear、量化 Linear kernel、激活、RoPE、GPU attention/KV | [cuda_matmul.cc](src/cuda_matmul.cc)、[cuda_quant.cu](src/cuda_quant.cu)、[cuda_ops.cu](src/cuda_ops.cu)、[cuda_attention.cu](src/cuda_attention.cu) |
| 会话与输出 | greedy/temperature/top-k；消息历史、流式解码和当前 KV 中的前缀复用 | [sampler.cc](src/sampler.cc)、[chat.cc](src/chat.cc)、[radix_tree.cc](src/radix_tree.cc)、[terminal.cc](src/terminal.cc) |
| 应用与观测 | 命令解析、生成/聊天/检查/benchmark、请求 trace、耗时与 token 统计 | [main.cc](src/main.cc)、[request_context.cc](src/request_context.cc)、[debug.cc](src/debug.cc) |

CPU 使用线程池执行并行计算，部分算子按编译目标使用 AVX2+FMA 或 ARM NEON。
CUDA 后端在 GPU 上执行权重计算、层间激活和 attention，并保留 GPU KV 缓存；
最终 logits 传回 CPU 进行采样和文本解码。CUDA 模式仍保留 CPU 权重和 CPU KV
分配，评估资源需求时需同时考虑主机内存和显存。

项目以单进程、单条上下文的教学推理和终端交互为主，使用时需注意：

- `MiniBatch` 逐 token 执行，不提供多序列并行或矩阵化 prompt prefill。
- GPU 后端支持单机单卡；没有多 GPU 推理、服务端队列、continuous batching、FlashAttention 或计算图编译器。
- KV 使用 F32，权重量化不会同时量化 KV。
- GGUF 模型适配面向 Llama/Qwen2 结构，不保证任意 GGUF 模型均兼容；不支持 K-quants/IQ-quants 等扩展格式。
- 分词不是通用 SentencePiece/unigram 实现，聊天模板也不是完整 Jinja2 引擎。
- 采样支持 greedy、temperature 和 top-k，尚未实现 top-p 与重复惩罚；生成内容可能出现重复。

### Benchmark 与统计口径

`bench` 默认预热一次、测量三次，每次使用独立上下文，预热不计入结果。
报告 prefill/decode 耗时、吞吐量及总耗时的最小值、中位数、最大值。
比较性能时应保持模型、输入、生成预算和线程配置一致，并关闭 `--verbose`。

权重压缩比例为 `F32 等价字节数 / 实际字节数`；`bench --quant` 比较 prompt
最后位置的 logits 与模型原始格式，不等同于全 F32 golden 或生成质量评估。
CUDA 摘要中权重显存不包括 KV/临时激活，KV 读写是逻辑字节数而非硬件带宽测量。
详细计时与指标定义见[调试与基准测试](docs/benchmark_debug_implementation.md)。

## 目录与模块文档

```text
include/mini_llama/   模块类型与函数声明
src/                 C++ 实现和 CUDA kernel
tests/               GoogleTest 模块测试及 CMake 注册
scripts/             模型准备、数值参考、CLI/CUDA 回归与报告
models/tiny/         教学模型资产
models/chat/         自行准备的 GGUF 聊天模型（目录可能需手动创建）
docs/                模块说明、阶段记录和验证结果
```

模块设计与开发记录按主题列在下表中。阶段文档中的测试数字、性能和实测环境
属于对应阶段的记录，不代表所有平台的运行结果。

| 主题 | 文档 |
| --- | --- |
| CLI 设计与开发记录 | [CLI 同步说明](docs/cli_integration_status.md)、[实现调整与验证](docs/inference_improvements.md) |
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
缺少工具或必需 GPU 用例未执行时返回 77，失败返回非零。具体用法见各脚本的
`--help` 和相关模块文档。
