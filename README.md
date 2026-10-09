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
| 模型管理 | 配置、权重与字节统计；整模型 Linear 重量化到 Q8_0/Q4_0，成功改变权重后清除旧 CUDA 快照 | [model.h](include/mini_llama/model.h)、[model.cc](src/model.cc) |
| JSON Loader | 解析清单与 tokenizer 描述，校验名称、形状、偏移和长度，加载 F32 权重 | [loader.h](include/mini_llama/loader.h)、[loader.cc](src/loader.cc) |
| GGUF Reader/Loader | GGUF v3 结构与元数据检查；F32/F16/Q8_0/Q4_0/Q4_1 加载；线性层保留量化块，embedding/norm/bias 使用 F32；Qwen2 NeoX RoPE | [gguf.h](include/mini_llama/gguf.h)、[gguf_loader.h](include/mini_llama/gguf_loader.h) |
| Tokenizer | ASCII、JSON 最长匹配、外部 GPT-2 byte-level BPE、GGUF 内嵌 byte-level BPE；特殊 token 和整体解码 | [tokenizer.h](include/mini_llama/tokenizer.h)、[gguf_tokenizer.h](include/mini_llama/gguf_tokenizer.h) |
| 上下文与前向 | CPU F32 KV Cache、GQA、可选 Q/K/V bias、因果 Attention、Pre-Norm SwiGLU；ForwardBatch 按 token 顺序执行 prefill/decode | [context.h](include/mini_llama/context.h)、[kv_cache.h](include/mini_llama/kv_cache.h)、[forward.h](include/mini_llama/forward.h) |
| 采样 | 贪心、温度、Top-k、种子控制、有限值与参数校验 | [sampler.h](include/mini_llama/sampler.h) |
| 对话与终端 | Plain/Qwen2/轻量 Jinja 模板、消息历史、当前上下文中的前缀复用、流式文本、/help、/clear、/stats、/params、/exit | [prompt_builder.h](include/mini_llama/prompt_builder.h)、[chat.h](include/mini_llama/chat.h)、[terminal.h](include/mini_llama/terminal.h)、[radix_tree.h](include/mini_llama/radix_tree.h) |
| CLI、追踪与调试 | 模型/词表/模板装配；generate/run/bench/inspect/inspect-gguf；阶段耗时、请求摘要、logits 导出、Top-k 与 KV 状态、吞吐统计 | [main.cc](src/main.cc)、[request_context.h](include/mini_llama/request_context.h)、[debug.h](include/mini_llama/debug.h) |
| Backend/CUDA Runtime | CPU/CUDA 参数解析、设备诊断、显存 Buffer、拷贝及 CPU stub；CLI CUDA 推理仍明确拒绝 | [backend.h](include/mini_llama/backend.h)、[cuda_runtime.h](include/mini_llama/cuda_runtime.h) |
| CUDA 库扩展 | F32 CudaTensor、F32/Q8_0 模型权重上传快照与统计接口；上传不改变 CPU 前向路径，运行统计为预留字段 | [cuda_tensor.h](include/mini_llama/cuda_tensor.h)、[cuda_tensor.cc](src/cuda_tensor.cc)、[model.cc](src/model.cc) |

### 当前执行路径

CUDA 权重上传不切换库层或 CLI 的推理后端；`ForwardToken()` 继续使用 CPU
算子，`ValidateInferenceBackend()` 仍拒绝
`generate/run/bench --backend cuda`。当前上下文只有 CPU KV Cache，Attention
仍在 CPU 上计算，尚无完整 GPU Attention、GPU KV Cache 或设备驻留 decode。
CUDA 细节与边界见[后端与 CUDA 说明](docs/backend_selection_cuda_runtime_implementation.md)。

2026-10-08 对当前源码重新构建并运行全量 CTest：CPU **157 项通过、12 项跳过**，
CUDA 单 GPU **166 项通过、3 项跳过**，两种构建均注册 169 项，失败为 0。
CUDA 验证覆盖 Runtime、设备 Tensor 和权重上传；生成仍使用 CPU。跳过原因与
复现配置见 [CLI 同步说明](docs/cli_integration_status.md#回归验证)。

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

CUDA 配置使用 `-DMINI_LLAMA_CUDA=ON`，需要 CUDAToolkit 的头文件与 Runtime 库。
当前模块以 C++17 编译并链接 `CUDA::cudart`，没有 CUDA kernel，也不启用 CUDA
语言编译。`./build-cuda/mini-llama --cuda-info` 可诊断设备；临时 SDK 可通过
`-DCUDAToolkit_ROOT=/path/to/sdk` 指定。

`generate`、`run`、`bench`、`inspect` 接受 `--backend cpu|cuda`；`--device ID`
仅与 CUDA 配置配合。`inspect` 只检查元数据，不查询 GPU 或上传权重。
`--threads 0` 使用自动 CPU 线程配置；显式正数固定配置。整模型重新量化和
CUDA 权重上传目前是库接口，没有对应 CLI 选项。

## 功能边界与文档

F32 Linear 仅支持 `[in]` / `[1, in]`，量化 Linear 支持 `[in]` / `[batch, in]`；
该 batch 算子能力不代表多序列并行推理。通用 Matmul 的 SIMD 模式仍复用标量
行并行实现。线程调度每次调用创建和回收线程，尚无常驻线程池。

GGUF 支持上述五种类型，不支持 Q4_K/Q5_K/Q6_K/IQ 等扩展类型；模型映射面向
现有 Llama/Qwen2 结构，不保证任意 architecture 兼容。Tokenizer 不是通用
SentencePiece/unigram 实现；聊天模板仅支持项目所需的 Jinja 子集。聊天前缀树
记录 token 前缀，不持有跨上下文 KV 快照。

- [CLI 接入与参数](docs/cli_integration_status.md)
- [主推理链路复习](docs/first_major_module_review.md)
- [真实 GGUF 接入复习](docs/second_major_module_review.md)
- [CPU 调度与 SIMD](docs/cpu_threaded_inference_implementation.md)
- [量化 Linear 与 batch](docs/simd_quantized_linear_implementation.md)
- [Golden logits 回归](docs/golden_logits_testing.md)
- [调试与基准](docs/benchmark_debug_implementation.md)
- [GGUF Tokenizer](docs/gguf_tokenizer_integration_implementation.md)
- [Chat Template 与 Run](docs/chat_prompt_run_implementation.md)
