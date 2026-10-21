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
sudo apt-get install build-essential cmake git python3 libgtest-dev
cmake --version
```

若发行版提供的 CMake 版本较旧，请先升级 CMake。其他系统请使用相应包管理器
安装编译器、CMake 和 Python 3。可选的 libedit 安装与配置见
[终端行编辑](#终端行编辑)。

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

CMake 构建 `mini_llama_core` 核心库和 `mini-llama` 程序，默认同时构建测试。
以下命令均在仓库根目录执行。CPU 和 CUDA 分别使用 `build/`、`build-cuda/`，
运行和测试时请使用对应目录中的程序。配置选项保存在各自的 CMake 缓存中；
修改选项后，需要重新编译才能更新可执行程序。

### 配置项与依赖

| 配置项 | 默认值 | 作用 |
| --- | --- | --- |
| `MINI_LLAMA_CUDA` | `OFF` | 启用 CUDA 后端，编译 kernel 并链接 CUDA Runtime/cuBLAS |
| `BUILD_TESTING` | `ON` | 构建 GoogleTest 测试，并注册 CTest/Python 回归 |
| `MINI_LLAMA_LIBEDIT` | `AUTO` | 选择交互行编辑器，支持 `AUTO`、`ON`、`OFF` |
| `CMAKE_CUDA_ARCHITECTURES` | 由工具链决定 | CUDA 目标架构；下文使用 `native`，需要 CMake 3.24 或更新版本 |

基础构建需要支持 C++17 的编译器和 CMake。默认启用测试，还需要 Python 3 和
GoogleTest。CMake 优先查找系统中的 GoogleTest；未找到时联网下载 v1.15.2。
离线构建测试时，可在配置命令中添加
`-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/已有的/googletest/源码目录` 复用本地源码。

### CPU 构建

CPU 构建无需 CUDA Toolkit：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
./build/mini-llama generate --model models/tiny -p hello -n 8
```

如果只需推理程序，在配置命令中追加 `-DBUILD_TESTING=OFF`，此时不需要
GoogleTest 或 Python，也不会下载测试依赖。以后需要运行测试时，将其改回
`-DBUILD_TESTING=ON`，重新配置并编译。

### CUDA 构建

CUDA 构建需要完整的 CUDA Toolkit，包括 CUDA 编译器、头文件、libdevice、
Runtime 和 cuBLAS；运行时还需要可访问的 NVIDIA GPU 和兼容驱动。
先用 `nvidia-smi` 检查设备；使用 `nvcc` 时，再用 `nvcc --version` 检查编译器。
仅能运行 `nvidia-smi` 不代表已安装 Toolkit。

在配置时能够访问 GPU 的环境中，使用自动架构探测：

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-cuda -j4
./build-cuda/mini-llama --cuda-info
./build-cuda/mini-llama generate --model models/tiny --backend cuda -p hello -n 8
```

`native` 需要在配置阶段运行 GPU 探测程序。无 GPU、设备访问受限或为其他 GPU
构建时，应显式指定目标架构，例如用 `-DCMAKE_CUDA_ARCHITECTURES=89` 为计算能力
8.9 的 GPU 构建。所选架构必须受 Toolkit 和 CUDA 编译器支持。
CUDA 构建的程序默认仍使用 CPU，推理时需添加 `--backend cuda` 才会使用 GPU。

Toolkit 安装在非标准位置时，可在当前终端设置路径，再执行上面的配置命令：

```bash
export CUDAToolkit_ROOT=/实际的/cuda/安装目录
export PATH="$CUDAToolkit_ROOT/bin:$PATH"
```

CMake 可使用 `nvcc` 或兼容的 Clang。首次配置可通过 `CUDACXX` 环境变量选择
CUDA 编译器，例如 `export CUDACXX=/usr/bin/clang++-18`；也可在配置命令中显式添加
`-DCMAKE_CUDA_COMPILER=/实际的/编译器路径`。路径和版本须与实际安装的 Toolkit
匹配。使用 `nvcc` 时，其主机编译器也须受 Toolkit 支持，可通过
`-DCMAKE_CUDA_HOST_COMPILER=/兼容的/g++路径` 指定。

已有构建目录会优先沿用缓存中的编译器，修改 `PATH` 或 `CUDACXX` 不会自动切换
工具链。更换编译器时使用新构建目录，或在完整的配置命令中添加 `--fresh`
刷新缓存；重新提供原先需要保留的配置选项。

### 终端行编辑

libedit 是 Linux/macOS 的可选依赖，Ubuntu/Debian 可通过
`sudo apt-get install libedit-dev` 安装。

| `MINI_LLAMA_LIBEDIT` | 交互输入行为 |
| --- | --- |
| `AUTO` | 找到 libedit 时启用；缺少时给出配置警告，使用内置 UTF-8 编辑器 |
| `ON` | 要求使用 libedit；缺少开发文件或平台不支持时配置失败 |
| `OFF` | 禁用 libedit 的查找和链接；Linux/macOS 使用内置 UTF-8 编辑器 |

内置编辑器支持常见中文的退格、左右移动、Home/End、Delete 和跨行重绘，
无需新增依赖。验证缺少 libedit 时的行为，可临时禁用它并重新编译：

```bash
cmake -S . -B build -DMINI_LLAMA_LIBEDIT=OFF
cmake --build build -j4
```

随后正常启动交互聊天即可测试。恢复自动检测时，把上面的 `OFF` 改为 `AUTO`，
再次配置并编译。测试 CUDA 程序时，将这两条命令的目录换成 `build-cuda`。

行编辑需要支持 UTF-8 的交互终端和有效的 `TERM`，建议使用 UTF-8 locale
（如 `LANG=C.UTF-8`）。管道、重定向输入及不支持行编辑的终端使用 `std::getline`。
`TERM=dumb`、未设置或清空 `TERM` 都会禁用两种行编辑器，仍可能出现中文退格
显示问题；测试内置编辑器时应保留正常的终端设置。

### 运行测试

测试需要 `BUILD_TESTING=ON`。CTest 只运行测试，不会编译程序；修改源码或配置后，
先完成构建再运行测试。CPU 构建：

```bash
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4
```

CUDA 测试共用 GPU，建议串行执行：

```bash
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure -j1
```

只运行终端输入回归（Linux/macOS）：

```bash
ctest --test-dir build --output-on-failure -R '^terminal-input$'
```

该测试通过真实 PTY 检查中文退格、光标移动、跨行重绘、终端状态恢复和重定向
输入；构建启用了哪种编辑器，就验证哪种编辑器。可先按上一节禁用 libedit，
再运行此测试验证内置编辑器。

部分测试有额外前提：

| 测试类型 | 缺少前提时的行为 |
| --- | --- |
| CUDA 设备测试 | 未启用 CUDA 后端或无可用 GPU 时跳过 |
| 真实模型相关测试 | 缺少相应模型时跳过；聊天模型准备见下文 |
| `golden-logits` | 缺少 NumPy 时跳过；Ubuntu/Debian 可安装 `python3-numpy` |

请检查测试输出中的跳过信息，确认实际执行范围；CTest 通过不代表所有 GPU 或
真实模型用例均已执行。更多 CUDA 验证说明见
[后端与 CUDA Runtime](docs/backend_selection_cuda_runtime_implementation.md)。

### 常见配置问题

| 现象 | 处理方式 |
| --- | --- |
| `Failed to find nvcc` | 确认 Toolkit 已安装，设置其 `bin` 目录到 `PATH`，必要时指定 `CUDAToolkit_ROOT` 和 CUDA 编译器 |
| `unsupported GNU version` | 为 `nvcc` 指定受支持的主机编译器，或选择与 Toolkit 兼容的 Clang；例如 CUDA 12.3 的 `nvcc` 不支持 GCC 13 |
| `No CUDA devices found`、`sm_No` | 检查配置环境的 GPU 访问权限；无法使用 `native` 时显式指定正确的架构编号 |
| 修改编译器后仍使用旧配置 | 使用新构建目录，或用 `--fresh` 加上完整配置重新生成缓存 |

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
