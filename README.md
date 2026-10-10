# mini-llama.cpp

一个 C++17 教学推理项目，实现模型加载、Transformer 前向、采样、单轮生成、
多轮聊天与基准测试。支持自定义 JSON+BIN 模型和 Llama/Qwen2 风格 GGUF v3。
默认计算后端为 CPU。

## 当前实现与接入状态

本说明按当前工作区源码同步，包含尚未提交的 CUDA 改动。模块接口存在、CLI 已接入
和通过运行验证是不同状态；阶段文档中的测试数字保留为历史记录。

| 模块 | 已实现的功能 | 源码入口 |
| --- | --- | --- |
| 张量与算子 | 连续 F32 Tensor、索引/reshape/shape 校验；Matmul、Linear、RMSNorm、Softmax、SiLU、SwiGLU、RoPE、ArgMax | [tensor.h](include/mini_llama/tensor.h)、[ops.h](include/mini_llama/ops.h) |
| CPU 调度与 SIMD | naive/threaded/simd/threaded-simd 分发；F32 点积按编译目标使用 AVX2+FMA、NEON 或 scalar；同步 ParallelFor，小任务串行 | [matmul_dispatch.h](include/mini_llama/matmul_dispatch.h)、[thread_pool.h](include/mini_llama/thread_pool.h) |
| 量化 | F32/Q8_0/Q4_0/Q4_1 权重表示；Q8_0/Q4_0 量化与三种格式反量化；量化 Linear 支持 batch 和外层并行，Q8_0/Q4_0 支持 NEON 点积 | [quantized_tensor.h](include/mini_llama/quantized_tensor.h)、[quant.h](include/mini_llama/quant.h) |
| 模型管理 | 配置、权重与字节统计；整模型 Linear 重量化到 Q8_0/Q4_0，CLI `--quant` 已接入；成功改变权重后清除旧 CUDA 快照 | [model.h](include/mini_llama/model.h)、[model.cc](src/model.cc) |
| JSON Loader | 解析清单与 tokenizer 描述，校验名称、形状、偏移和长度，加载 F32 权重 | [loader.h](include/mini_llama/loader.h)、[loader.cc](src/loader.cc) |
| GGUF Reader/Loader | GGUF v3 结构与元数据检查；F32/F16/Q8_0/Q4_0/Q4_1 加载；线性层保留量化块，embedding/norm/bias 使用 F32；Qwen2 NeoX RoPE | [gguf.h](include/mini_llama/gguf.h)、[gguf_loader.h](include/mini_llama/gguf_loader.h) |
| Tokenizer | ASCII、JSON 最长匹配、外部 GPT-2 byte-level BPE、GGUF 内嵌 byte-level BPE；特殊 token 和整体解码 | [tokenizer.h](include/mini_llama/tokenizer.h)、[gguf_tokenizer.h](include/mini_llama/gguf_tokenizer.h) |
| 上下文与前向 | CPU/GPU F32 KV Cache、GQA、可选 Q/K/V bias、因果 Attention、Pre-Norm SwiGLU；ForwardBatch 按 token 顺序执行 prefill/decode | [context.h](include/mini_llama/context.h)、[kv_cache.h](include/mini_llama/kv_cache.h)、[forward.h](include/mini_llama/forward.h) |
| 采样 | 贪心、温度、Top-k、种子控制、有限值与参数校验 | [sampler.h](include/mini_llama/sampler.h) |
| 对话与终端 | Plain/Qwen2/轻量 Jinja 模板、消息历史、当前上下文中的前缀复用、流式文本、/help、/clear、/stats、/params、/exit | [prompt_builder.h](include/mini_llama/prompt_builder.h)、[chat.h](include/mini_llama/chat.h)、[terminal.h](include/mini_llama/terminal.h)、[radix_tree.h](include/mini_llama/radix_tree.h) |
| CLI、追踪与调试 | 模型/词表/模板装配；generate/run/bench/inspect/inspect-gguf；阶段耗时、请求摘要、logits 导出、Top-k 与 KV 状态、吞吐统计 | [main.cc](src/main.cc)、[request_context.h](include/mini_llama/request_context.h)、[debug.h](include/mini_llama/debug.h) |
| Backend/CUDA Runtime | CPU/CUDA 参数解析、设备诊断、显存 Buffer、拷贝及 CPU stub；generate/run/bench 已接入 CUDA，无设备或执行失败明确报错 | [backend.h](include/mini_llama/backend.h)、[cuda_runtime.h](include/mini_llama/cuda_runtime.h) |
| CUDA 设备前向 | F32 激活；F32/Q8_0/Q4_0/Q4_1 权重上传与 Linear；Embedding、Norm、RoPE、FFN、Attention、KV 全程设备计算，运行统计已接入 | [cuda_tensor.h](include/mini_llama/cuda_tensor.h)、[cuda_ops.h](include/mini_llama/cuda_ops.h)、[cuda_quant.h](include/mini_llama/cuda_quant.h)、[cuda_attention.h](include/mini_llama/cuda_attention.h)、[cuda_kv_cache.h](include/mini_llama/cuda_kv_cache.h) |

### 当前执行路径

CLI 在加载与可选重量化后，按 `--backend cuda` 上传权重。库层
`ForwardToken()` 在无快照时走 CPU；存在快照时校验完整权重并走设备前向，
缺失或损坏的快照报错。GPU 前向每 token 只下载最终 F32 logits，采样在 CPU；
激活和 KV 历史保留在 GPU，正常链路 H2D 为 0、CPU Attention fallback 为 0。
切换权重或后端时应新建上下文并重放历史。细节见
[后端与 CUDA 说明](docs/backend_selection_cuda_runtime_implementation.md)和
[设备驻留 decode](docs/device_resident_cuda_decode.md)。

2026-10-10 重新构建并验证 CPU/CUDA；当前测试注册与实测结果见
[CLI 同步说明](docs/cli_integration_status.md#回归验证)。全量 CTest 包含后端 CLI、
多线程 CLI 与 NumPy golden logits；GPU 用例的跳过不计为实际执行通过。

## 构建与使用

CPU 构建无需 CUDA Toolkit；启用测试需要
GTest（本机安装或 CMake 下载）与 Python 3，golden logits 另需 NumPy。

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4

./build/mini-llama generate models/tiny --prompt hello --n-predict 4 --threads 4
./build/mini-llama bench models/tiny --prompt hello --n-predict 4 --threads 4
./build/mini-llama run models/tiny
./build/mini-llama inspect models/tiny
./build/mini-llama inspect-gguf models/tiny/model.gguf
```

CUDA 配置使用 `-DMINI_LLAMA_CUDA=ON`，需要完整的 CUDA Toolkit，包括编译器、
头文件、libdevice、CUDA Runtime 和 cuBLAS。当前使用 C++17/CUDA 17，编译
`src/cuda_ops.cu`、`src/cuda_quant.cu` 和 `src/cuda_attention.cu`，链接
`CUDA::cudart` 与 `CUDA::cublas`。
只安装运行库不足以构建或让 Clangd 检查 CUDA 源码。

### CUDA 工具链与 Clangd

将 Toolkit 安装在系统目录或固定用户目录，例如 `/usr/local/cuda-12.3` 或
`$HOME/.local/share/cuda-12.3`，不要使用 `/tmp`。安装方法见
[NVIDIA 安装指南](https://docs.nvidia.com/cuda/archive/12.3.2/cuda-installation-guide-linux/index.html)。
本机验证组合为 CUDA 12.3 与 Clang/Clangd 18；其他设备可以使用 NVCC 或与
Toolkit 版本兼容的 Clang。NVCC 的主机编译器也须与 Toolkit 版本兼容，例如
CUDA 12.3 可通过 `-DCMAKE_CUDA_HOST_COMPILER=g++-12` 选择 GCC 12。

以下命令适用于安装在固定用户目录的 CUDA 12.3 与 Clang 18：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCUDAToolkit_ROOT="$HOME/.local/share/cuda-12.3" \
  -DCMAKE_CUDA_COMPILER=clang++-18 \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4
./build/mini-llama --cuda-info
```

`89` 对应本机 GPU；其他设备应按目标 GPU 选择计算能力。需要部署到多个 GPU
架构时可传入列表，例如 `-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"`，前提是
安装的 Toolkit 支持这些架构。使用架构选项需要 CMake 3.18 或更新版本。
系统已安装 NVCC 时可省略 Clang 编译器参数，让 CMake 使用 NVCC。

CMake 自动生成 `build/compile_commands.json`，包含本机 Toolkit 路径和每个
CUDA 源文件的编译命令。Clangd 默认会搜索源码上层目录及 `build/`，因此新设备
配置到 `build/` 即可，无需新增项目 `.clangd` 文件或复制其他机器的编译数据库。
如果 CPU 与 CUDA 分别配置到 `build/` 和 `build-cuda/`，应在编辑器已有的
Clangd 设置中将 CUDA 编译数据库目录设为 `build-cuda/`。详见
[Clangd 编译配置](https://clangd.llvm.org/config.html#compilationdatabase)。

更换编译器或迁移项目后，应重新配置构建目录；CMake 3.24 及更新版本可在
上面的配置命令中添加 `--fresh`，清除旧机器或旧 SDK 的缓存。首次修正已有
编辑器会话后重启 Clangd，之后重新打开项目使用同一固定 Toolkit 和编译数据库。

### 在其他设备构建

无 NVIDIA GPU 或无需 CUDA 的设备使用前面的 CPU 构建命令，不需要安装
Toolkit。CUDA 设备安装完整 Toolkit 后在本机重新运行配置和构建命令；SDK
路径通过 `CUDAToolkit_ROOT` 指定，项目不写入个人安装路径，也不拷贝构建缓存。
无需测试时可加入 `-DBUILD_TESTING=OFF`，避免部署环境依赖 GTest、Python 和
NumPy。CUDA 可执行程序还需要目标机器提供兼容的 NVIDIA 驱动及 CUDA Runtime、
cuBLAS 动态库；跨 CPU 架构或操作系统部署时应在目标平台重新构建。

`generate`、`run`、`bench`、`inspect` 接受 `--backend cpu|cuda`；`--device ID`
仅与 CUDA 配置配合。`inspect` 只检查元数据，不查询 GPU 或上传权重。
`--threads 0` 使用自动 CPU 线程配置；显式正数固定配置。
`generate/run/bench --quant q8_0|q4_0` 重量化所有 Linear；不指定时保留加载格式。
CUDA 权重上传由推理命令自动完成，`inspect` 不接受 `--quant`。

```bash
./build-cuda/mini-llama generate models/tiny --backend cuda -p hello -n 4
./build-cuda/mini-llama run models/chat --backend cuda -n 16
./build-cuda/mini-llama bench models/tiny --backend cuda --quant q4_0 -n 4
```

`bench --quant` 还报告 prompt 最后一份 logits 相对模型原始格式的最大/平均
绝对误差和权重字节数；原模型不等于全 F32 模型，该指标也不代表生成质量评估。
generate/bench 的 CUDA 运行摘要累计当前请求的 prefill + decode；run 的设备
计数自权重准备后跨轮累计。上传不计入运行拷贝统计。

## 功能边界与文档

F32 Linear 仅支持 `[in]` / `[1, in]`，量化 Linear 支持 `[in]` / `[batch, in]`；
该 batch 算子能力不代表多序列并行推理。通用 Matmul 的 SIMD 模式仍复用标量
行并行实现。线程调度每次调用创建和回收线程，尚无常驻线程池。

GGUF 支持上述五种类型，不支持 Q4_K/Q5_K/Q6_K/IQ 等扩展类型；模型映射面向
现有 Llama/Qwen2 结构，不保证任意 architecture 兼容。Tokenizer 不是通用
SentencePiece/unigram 实现；聊天模板仅支持项目所需的 Jinja 子集。聊天前缀树
记录 token 前缀，不持有跨上下文 KV 快照。

CUDA Prefill 仍逐 token 计算；Context 保留 CPU Cache 的分配，GPU 路径不镜像
更新其内容。设备临时 Tensor 和 cuBLAS 句柄仍按调用分配；共享权重快照的运行
计数采用单调用线程约定。本轮只验证单 GPU，未运行 compute-sanitizer。

- [CLI 接入与参数](docs/cli_integration_status.md)
- [主推理链路复习](docs/first_major_module_review.md)
- [真实 GGUF 接入复习](docs/second_major_module_review.md)
- [CPU 调度与 SIMD](docs/cpu_threaded_inference_implementation.md)
- [量化 Linear 与 batch](docs/simd_quantized_linear_implementation.md)
- [Golden logits 回归](docs/golden_logits_testing.md)
- [调试与基准](docs/benchmark_debug_implementation.md)
- [GGUF Tokenizer](docs/gguf_tokenizer_integration_implementation.md)
- [Chat Template 与 Run](docs/chat_prompt_run_implementation.md)
- [后端与 CUDA Runtime 当前状态](docs/backend_selection_cuda_runtime_implementation.md)
- [设备驻留 CUDA Decode 与验证](docs/device_resident_cuda_decode.md)
- [CUDA KV Cache 与 Attention](docs/cuda_kv_cache_implementation.md)
- [cuBLAS F32 Matmul / Linear 阶段记录](docs/cublas_f32_linear_implementation.md)
- [CUDA Ops / Quant 阶段记录](docs/cuda_ops_quant_implementation.md)
- [CUDA Tensor 与权重上传阶段记录](docs/cuda_tensor_model_weights_implementation.md)
- [Tensor 与基础算子](docs/tensor_ops_implementation.md)
- [Tokenizer](docs/tokenizer_implementation.md)
- [KV Cache 与 Context](docs/kv_cache_context_implementation.md)
- [Sampler 与采样策略](docs/sampler_sampling_strategy_implementation.md)
- [JSON 模型格式与 Loader](docs/model_formats_and_loader_branch_summary.md)
- [GGUF 文件格式](docs/gguf_implementation.md)
- [GGUF Loader](docs/gguf_loader_branch_summary.md)
- [Qwen2 端到端 smoke 阶段记录](docs/qwen2_gguf_end_to_end_smoke_test.md)
