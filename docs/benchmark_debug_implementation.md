# 调试与基准测试阶段实现汇总

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。
>
> 本文记录 `feature/benchmark-design` 阶段完成时的实现状态。命令行的整体装配
> 约定见[命令行与模块同步说明](cli_integration_status.md)。
> 本文的 CLI 示例已同步线程配置与输出标签；CPU 并行路径见
> [CPU 多线程与 F32 SIMD 实现说明](cpu_threaded_inference_implementation.md)。

本阶段在现有模型加载、Tokenizer、`ForwardBatch()` 和 `MiniSampler` 之上增加了
独立的调试与基准测试模块，并把它接入 `mini-llama bench` 命令。目标是用同一套
CPU 推理链路分别测量预填充和逐 token 解码，同时提供定位张量、logits 与 KV
Cache 状态的文本输出，不改变普通 `generate` 和交互式 `run` 的生成语义。

## 阶段定位

```text
模型文件 + Tokenizer
        ↓
   CLI bench
        ↓ 编码 prompt
 RunBenchmark()
   ├─ ForwardBatch(prompt) ──→ prefill_ms
   ├─ 首 token 采样
   └─ ForwardBatch(single) ──→ decode_ms
        ↓
 BenchmarkResult + RequestContext + 可选 verbose 输出
```

本阶段涉及的主要文件：

- `include/mini_llama/debug.h`：公共调试接口和基准结果结构。
- `src/debug.cc`：调试输出与基准执行实现。
- `src/main.cc`：`bench` 参数解析、模型装配、结果展示和请求追踪。
- `tests/debug_test.cc`：调试模块单元测试。
- `tests/main_test.cc`：`bench` 命令集成测试。
- `CMakeLists.txt` 与 `tests/CMakeLists.txt`：生产代码和测试目标注册。

## 公共接口参考

### BenchmarkResult

`BenchmarkResult` 保存一次基准运行的计数与耗时：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `n_prompt_tokens` | `int` | 本次预填充的 prompt token 数量 |
| `n_generated_tokens` | `int` | 实际完成采样的生成 token 数量 |
| `n_decode_tokens` | `int` | 实际执行单 token 前向的次数 |
| `prefill_ms` | `double` | 批量预填充耗时，单位为毫秒 |
| `decode_ms` | `double` | 逐 token 解码阶段累计耗时，单位为毫秒 |

结构体提供两个吞吐量计算函数：

```text
tokens_per_sec
  = n_generated_tokens × 1000 / (prefill_ms + decode_ms)

decode_tokens_per_sec
  = n_decode_tokens × 1000 / decode_ms
```

当分母不大于零时返回 `0.0`；解码次数为零时，解码吞吐量同样返回 `0.0`。

### 调试输出函数

| 接口 | 行为 |
| --- | --- |
| `DumpTensorShape(tensor, name)` | 输出名称、完整 shape、固定的 F32 类型和元素数量 |
| `DumpLogitsTopK(logits, k)` | 对一维 logits 做降序部分排序并输出前 k 个 token ID 与数值 |
| `DumpKvCacheInfo(cache, current_pos)` | 输出层数、最大序列长度、KV 头数、头维度及键值张量 shape |

`DumpLogitsTopK()` 的输入不是一维张量时只输出错误提示。`k <= 0` 或 `k` 大于
词表大小时会输出整个词表。详细浮点输出临时使用固定六位小数，函数返回前会恢复
调用方输出流原来的格式状态。

### RunBenchmark

公共接口接受已经加载的模型、已经编码的 prompt token、生成预算、随机种子和
详细输出开关：

```cpp
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens,
                             int n_predict,
                             unsigned int seed,
                             bool verbose);
```

另一个重载额外接受 `std::ostream&`。CLI 使用该重载，将详细输出写入
`RunMain()` 的可注入输出流；五参数接口保持直接写入 `std::cout` 的便捷行为。

## 基准执行流程

### 预填充

每次运行创建独立的 `MiniLlamaContext`，因此 KV Cache、位置和 token 历史不会在
不同基准之间复用。prompt 通过 `MiniBatch::FromTokens(prompt_tokens, 0)` 一次提交，
`steady_clock` 只测量这次 `ForwardBatch()` 调用。

预填充结束后，最后一个 prompt token 的 logits 直接用于首个生成 token 的采样。
首 token 不需要额外前向，因此它会计入 `n_generated_tokens`，但不会计入
`n_decode_tokens`。

### 逐 token 解码

从第二个生成 token 开始，运行器把上一次采样结果封装为 `MiniBatch::Single()`，
执行一次前向并再次采样。因而在没有触及上下文上限时：

```text
n_generated_tokens = N
n_decode_tokens    = max(N - 1, 0)
```

循环同时受 `n_predict` 和 `model.config.max_seq_len` 限制。prompt 已占满上下文时
只执行预填充；`n_predict <= 0` 时不生成 token，`decode_ms` 保持为零。

### 详细模式

`verbose == true` 时输出：

1. 预填充 token 数量。
2. 预填充 logits 的 shape 和 Top-5；词表少于五项时自动缩小。
3. 预填充后的 KV Cache 信息和当前位置。
4. 每个生成 token ID，以及后续解码步骤的 logits Top-5。
5. 运行结束时的 KV Cache 信息。

详细输出用于定位状态，不用于获得稳定性能数字。预填充调试输出位于预填充计时
之外，但解码循环中的 token 与 Top-k 输出位于 `decode_ms` 的计时区间内，启用
`--verbose` 会降低测得的解码吞吐量。

## CLI 接入

### 命令与选项

```bash
./build/mini-llama bench <model-path|dir> [options]
```

| 选项 | 默认值 | 行为 |
| --- | --- | --- |
| `-m, --model <path\|dir>` | 无 | 指定二进制、GGUF 或模型目录；也可使用位置参数 |
| `--config <path>` | 自动推导 | 指定 JSON+BIN 模型清单 |
| `--weights <path>` | 无 | `--model` 的旧版二进制权重别名 |
| `--tokenizer <path>` | 自动选择 | 指定 tokenizer 目录或 `vocab.json` |
| `--vocab <path>` | 无 | `--tokenizer` 的旧版别名 |
| `-p, --prompt <text>` | `hello` | 指定需要编码和预填充的文本 |
| `-n, --n-predict <n>` | `16` | 指定生成预算；`0` 表示只测预填充 |
| `--max-tokens <n>` | `16` | `--n-predict` 的旧版别名 |
| `--threads <N>` | 自动 | 正数设置 CPU 线程数；`0` 恢复自动模式 |
| `--backend cpu\|cuda` | `cpu` | 选择 CPU 或完整设备前向 |
| `--device <id>` | `0` | CUDA 设备编号，仅与 CUDA 后端配合 |
| `--quant q8_0\|q4_0` | 保留模型格式 | 重量化所有 Linear，输出相对模型原始格式的 logits 误差 |
| `--seed <n>` | `0` | 设置采样器种子 |
| `--verbose` | 关闭 | 输出 logits 和 KV Cache 调试信息 |

`bench` 复用 `generate` 的模型路径解析、模型加载、Tokenizer 选择和特殊 token
校验，但不接受 `--temperature`、`--top-k` 或 `--dump-logits`。当前基准运行器使用
默认温度 `0.0`，因此实际采用贪心采样；固定 seed 已保留在接口和追踪中，但不会
改变当前贪心结果。

CUDA 模式自动上传权重，基准阶段执行完整设备前向和 CPU 采样，报告上传权重
及运行统计。`bench --quant` 保留原模型，对同一 prompt 最后位置的 logits
报告最大/平均绝对误差；两次误差对照前向不计入 prefill/decode 性能区间。
原始格式可能已经量化，不等于全 F32 参考。运行摘要在误差计算前输出，
只包含基准实际工作量。权重大小为数据字节，不包括 KV Cache 和临时激活。

### 运行性能基准

```bash
./build/mini-llama bench models/tiny \
  --prompt hello \
  --n-predict 16 \
  --seed 42 \
  --threads 4
```

输出结构如下，实际耗时取决于硬件、模型和系统负载：

```text
mini-llama benchmark
====================
threads: 4
prompt tokens: 6
generated tokens: 16
decode tokens: 15
prefill: <毫秒> ms
decode: <毫秒> ms
throughput: <每秒 token 数> tokens/s
decode throughput: <每秒 token 数> tokens/s
tokens/s (total): <整体生成吞吐量>
tokens/s (Decode): <逐 token 解码吞吐量>
```

`threads` 是已解析的配置线程数。每个算子通过 `ParallelFor()` 按任务量决定
是否创建线程，小任务仍直接执行。新进程使用自动模式，`--threads 0` 从硬件
并发数选择配置，获取失败时回退为 4；比较性能时应显式固定线程数。
新旧两组吞吐标签使用相同的 `BenchmarkResult` 计算函数。

### 只测预填充

```bash
./build/mini-llama bench models/tiny -p hello -n 0
```

此时仍会执行完整 prompt 前向，但生成和解码计数均为零。当前
`tokens_per_sec()` 按生成 token 定义，因此整体吞吐量也显示为零；`prefill_ms`
仍可用于不同 prompt 长度或模型之间的直接比较。

### 查看详细状态

```bash
./build/mini-llama bench models/tiny -p hello -n 4 --threads 4 --verbose
```

使用该模式检查 logits 排名、KV Cache shape 和位置推进。需要记录性能数字时应
关闭 `--verbose`，并在相同机器、相同模型和相同 prompt 下比较。

## CLI 数据流与请求追踪

`RunBench()` 先使用现有 `EncodePrompt()` 编码输入，再调用带输出流参数的
`RunBenchmark()`。结果随后写回 `RequestContext`：

| 基准结果 | 请求追踪字段 |
| --- | --- |
| `n_prompt_tokens` | `prompt_tokens`、`prefill_tokens` |
| `n_generated_tokens` | `generated_tokens` |
| `n_decode_tokens` | `decode_tokens` |
| `prefill_ms` | `prefill_ms` |
| `decode_ms` | `decode_ms` |

标准输出承载基准结果和可选调试信息；标准错误输出承载 `model_load`、`tokenize`、
`prefill`、`decode` 事件及最终 summary。模型路径解析、加载或推理失败时沿用现有
CLI 异常处理，输出 `status=error` 的请求摘要。

## 设计取舍

### 基准路径不替换普通生成路径

`GenerateTokens()` 仍负责温度/Top-k、EOS、文本解码、逐步 logits 文件导出和
生成请求语义。`RunBenchmark()` 只负责隔离上下文、执行固定推理工作量和统计
耗时。两条路径共享底层 `ForwardBatch()` 和 `MiniSampler`，但不为了减少代码行数
而混合不同目的。

### 首 token 与 decode 分开计数

首 token 直接来自预填充 logits，没有单 token 前向成本。把它算进生成数、但不
算进解码数，可以让 `decode_tokens_per_sec()` 的分子与真正执行的解码前向次数
一致。整体吞吐量则继续反映从预填充到获得全部生成 token 的总推理成本。

### 输出流可注入

原始便捷接口继续支持 `std::cout`，CLI 使用显式 `std::ostream&`。这样
`RunMain()` 的调用方和单元测试能完整捕获 `--verbose` 输出，调试模块也不需要依赖
命令行全局状态。

## 测试覆盖

`tests/debug_test.cc` 包含 8 项单元测试：

- 两种吞吐量公式及零耗时保护。
- 张量 shape 元数据输出。
- logits Top-k 的数量、降序和 token ID。
- 非一维 logits 的拒绝提示。
- KV Cache 配置、shape 和当前位置输出。
- 预填充、首 token 和逐 token 解码计数。
- 最大上下文长度截断。
- verbose 模式的完整调试信息。

`tests/main_test.cc` 另有 1 项 CLI 集成测试，覆盖 `bench --help`、资源装配、
`--verbose` 输出流、统计结果和 `mode=bench` 请求追踪。

本阶段验证命令：

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

基准模块阶段完成时，完整构建成功，CTest **104/104** 通过。手工执行
`bench models/tiny -p hello -n 2 --seed 42` 可得到 2 个生成 token、1 次 decode
及对应的 prefill/decode 吞吐量。

CPU 多线程阶段的历史完整回归为 113/113，`mini-llama-tests` 前缀测试为
22/22。`python3 scripts/test_threaded_cli.py` 检查 1/4/自动线程配置、logits
和生成结果一致性，以及本节示例中的输出标签；测试入口使用当前文件名
`tests/threaded_matmul_test.cc`。完整说明见 CPU 多线程文档。
该脚本已以 `threaded-cli-smoke` 接入全量 CTest。

## 当前边界

- 库级 `RunBenchmark()` 接受已经编码且非空的 prompt；空 prompt 会被
  `ForwardBatch()` 拒绝。CLI 会通过 `EncodePrompt()` 为合法空文本补入 BOS。
- 基准路径不识别 EOS，也不解码生成文本；它按预算或上下文上限完成固定工作量。
- 当前只使用贪心采样，`seed` 不影响结果；随机采样性能不属于本阶段范围。
- 当前没有 warm-up、重复轮次、均值、分位数或方差统计，单次结果容易受系统负载
  和首次运行开销影响。
- `prefill_ms` 与 `decode_ms` 不包含模型加载和 prompt 编码；这些阶段单独记录在
  `RequestContext` 中。
- `decode_ms` 覆盖逐 token 前向、采样以及 verbose 模式下的循环内输出，因此
  性能比较必须关闭 `--verbose`。
- CLI 已有 `--backend cpu|cuda`、`--device`、`--threads` 和 `--quant`；CUDA
  benchmark 已连接完整设备前向。CPU F32 Linear 使用 SIMD/线程分发，Attention 按 head 并行；
  三种量化 Linear 支持 batch 与外层并行，Q8_0/Q4_0 可使用 NEON，Q4_1 点积
  为 scalar。不能概括为全部 CPU 算子都已 SIMD 化；GPU 链路与验证见
  [设备驻留 Decode](device_resident_cuda_decode.md)。
- `DumpKvCacheInfo()` 面向正常构造的四维 `KvCache`，不负责诊断损坏或未初始化的
  cache shape。verbose 中展示的是保留的 CPU Cache 配置，不是 GPU 历史回读；
  GPU Cache 实际读写由 CUDA 运行摘要记录。

## 相关文档

- [命令行与模块同步说明](cli_integration_status.md)
- [CPU 多线程与 F32 SIMD 实现说明](cpu_threaded_inference_implementation.md)
- [KV Cache、Context 与 Forward 实现说明](kv_cache_context_implementation.md)
- [Sampler、采样策略与生成入口阶段实现汇总](sampler_sampling_strategy_implementation.md)

## 一句话总结

本阶段把可复用的张量、logits、KV Cache 调试输出和 prefill/decode 计时封装为
独立模块，并通过 `mini-llama bench` 接入真实模型加载、请求追踪和完整回归测试，
形成了可运行、可诊断、计时口径明确的 CPU 推理基准入口。
