# 命令行与模块同步说明

本文描述当前 `src/main.cc` 与已实现模块的接入约定。分支阶段文档中的实现范围、
真实模型输出和测试快照保留历史含义；当前入口行为以本文和源码为准。
调试接口、计时口径和 `bench` 的详细设计见
[调试与基准测试阶段实现汇总](benchmark_debug_implementation.md)。
CPU 调度、F32 SIMD 和测试细节见
[CPU 多线程与 F32 SIMD 实现说明](cpu_threaded_inference_implementation.md)。

## 构建与入口

所有 `src/*.cc` 已纳入 `mini_llama_core`。`mini-llama` 可执行目标定义
`MINI_LLAMA_BUILD_EXECUTABLE`，启用同一个 `src/main.cc` 中的全局 `main()`。
测试可直接使用流注入的 `RunMain()`；库入口 `Generate()`、`Run()` 继续接受已加载
的模型与分词器。

| 命令 | 行为 |
| --- | --- |
| `generate` | 单轮生成，默认教学模型、`hello`、16 个生成 token |
| `run` | 多轮聊天，需要模型路径，默认每轮最多生成 64 个 token |
| `bench` | 测量预填充与逐 token 解码性能，可用 `--verbose` 输出调试信息 |
| `inspect` | 检查 JSON 清单或 GGUF 元数据，不加载模型权重或分词器 |
| `inspect-gguf` | 直接检查 GGUF 文件头、元数据和张量目录 |
| `--cuda-info` | 查询 CUDA Runtime 构建状态和真实 GPU 信息，失败时返回非零 |

CPU 线程配置已通过 `--threads` 接入推理命令，后端选项和 CUDA Runtime 诊断也已接入。
库层已有 CUDA Tensor、F32/Q8_0 权重上传快照和统计接口；上传不改变库层
前向后端，当前算子、Attention 与 KV Cache 均在 CPU。CLI 未调用上传接口，
CUDA 推理请求仍明确报错。CUDA 算子、GPU Attention、GPU KV Cache 和完整
设备驻留 decode 尚未实现，运行统计为预留字段。
后端参数、Runtime 当前行为与验证结果见
[后端选择与 CUDA Runtime 阶段实现汇总](backend_selection_cuda_runtime_implementation.md)。
整模型 `QuantizeModelToQ80/Q40()` 已有库实现，尚未接入 CLI；转换所有 Linear，
保持 embedding/norm/bias，成功改变权重后清除旧 CUDA 快照。已有 CPU 量化
Linear 可执行 GGUF Q8_0/Q4_0/Q4_1 权重。单张量量化、权重字节统计与上传也是
库能力，没有对应命令行开关。

## 参数与模型选择

`ParseOptions()` 统一消费选项及其值，并生成 `CommandOptions`。选项值中的
`--help`、`--model` 按文本处理。`RunMain()` 允许资源选项，库入口 `Run()` 禁止
重新指定资源。查看帮助不读取模型。

`ResolveModelPaths()` 支持二进制权重、GGUF 文件和目录：

1. `--model` 与旧版 `--weights` 二选一；JSON 权重可配合 `--config` 使用。
2. 模型目录中的 `model.json + model.bin` 完整时优先采用 JSON+BIN，否则寻找
   唯一 GGUF 候选。多个候选要求显式指定文件，格式按魔数判断。
3. 单个二进制权重未指定配置时读取同目录 `model.json`；只指定配置时推导同目录
   `model.bin`。`generate` 未指定资源时使用 `models/tiny/model.bin`。
4. `inspect` 可只读取 JSON 清单，无需权重和词表文件。
5. GGUF 不与 `--config`、`--weights` 混用；`--tokenizer` 与旧版 `--vocab` 二选一。

推理长度选项包括 `--n-predict/-n` 和旧版 `--max-tokens`。CLI `generate` 拒绝
超过上下文的完整生成预算，`-n 0` 只执行 prefill；`run` 要求正数响应预算并根据
当前剩余窗口截断。`bench` 接受 `-n 0` 测量纯预填充，并在达到上下文上限时停止。
既有库调用保留按窗口剩余容量截断的行为。

### CPU 线程数

`generate`、`bench`、`run` 支持 `--threads N`。参数解析后在推理前调用
`SetThreadCount(N)`：正数设置线程数，`0` 恢复自动模式；自动模式使用硬件
并发数，获取失败时回退为 4。CLI 接受 `0..1000000`，拒绝负数、非法文本和溢出。
新进程默认自动模式；库入口 `Run()` 未指定选项时保留当前线程配置。

```bash
./build/mini-llama generate models/tiny --prompt hello --n-predict 4 --threads 4
./build/mini-llama bench models/tiny --prompt hello --n-predict 4 --threads 4 --verbose
```

`generate` 和 `bench` 输出实际解析的配置线程数，例如 `threads: 4`。
Linear 按输出通道、Matmul 按输出行、CPU Attention 按 query head 复用
`ParallelFor()`。小任务和单线程配置直接在调用线程执行，因此该输出不等于
每个算子创建的线程数。Q8_0/Q4_0/Q4_1 Linear 已按 batch 行与输出通道展平并行；
Q8_0/Q4_0 使用 NEON 或 scalar 点积，Q4_1 使用 scalar 点积。F32 Linear 仍限于
单向量或 `[1, in]`；量化 Linear 支持 `[batch, in]`，ForwardBatch 仍逐 token 执行。

## 分词器与特殊 token 校验

`LoadModelAndTokenizer()` 优先使用显式 `--tokenizer/--vocab`。未指定时，GGUF
读取内嵌分词器元数据，缺失时尝试同目录外部词表；JSON 模型按清单选择 ASCII
或 JSON 词表。外部词表目录有 `merges.txt` 时使用 BPE。

JSON 模型通过 `ValidateTokenizerSpecialIds()` 核对清单与实际分词器的三个 ID：

| 字段 | 校验依据 |
| --- | --- |
| `bos_id` | `manifest.tokenizer.bos_id == tokenizer.bos_id()` |
| `eos_id` | `manifest.tokenizer.eos_id == tokenizer.eos_id()` |
| `unk_id` | `manifest.tokenizer.unk_id == tokenizer.unk_id()` |

自动选择、显式词表和旧版 `--vocab` 均执行校验。发生冲突时在推理前失败，例如：

```text
tokenizer eos_id mismatch: manifest=11, actual=2
```

校验不把 ID 固定为教学模型的 1/2/0。JSON 词表采用其他 ID 时，只要清单与词表
一致即可运行；ASCII 分词器自身采用固定 ID。GGUF 按其分词器元数据恢复 ID，
不套用 JSON 清单默认值。`ValidateGeneration()` 另行检查词表容量与模型兼容性。

## 推理编排与终端帮助

`RunGenerate()` 调用 `GenerateTokens()`，后者执行 tokenize、prefill、采样和
decode，并更新上下文与请求指标。首个 token 也检查 EOS；最后一个生成 token
无需计算下一份 logits。原始 ID 和导出文件保留 EOS，展示文本过滤结尾 EOS。
`--dump-logits` 导出逐位置 F32 logits 和 `generation_tokens.txt`。

`RunBench()` 调用独立的 `RunBenchmark()`，分别统计批量 prefill 和逐 token
decode。首个生成 token 直接从 prefill logits 采样，因此生成 N 个 token 通常
只有 N-1 次 decode。`--verbose` 输出 logits Top-k 和 KV Cache 状态，诊断时可用，
正式测量吞吐量时应关闭。
结果同时输出 `tokens/s (total):`、`tokens/s (Decode):`，保留原有
`throughput:` 和 `decode throughput:` 标签。

`RunChat()` 驱动交互，`RunChatTurn()` 根据 RadixTree 命中与当前物理上下文的
公共前缀确定复用范围，只对缺失后缀 prefill。正常生成结束后提交消息与缓存
前缀；推理失败时重建上下文并清除当前 token 跟踪，避免复用半轮数据。

`Terminal::PrintHelp()` 统一维护 `/help`、`/clear`、`/stats`、`/params` 和 `/exit`
的帮助。入口调用该函数即可，不再追加 `/params` 文案。`/params` 使用
`Terminal::PrintParams()`，`/clear` 同时清理会话消息、缓存与累计统计。

## 请求与会话统计

| 路径 | 请求生命周期 | 追踪输出 |
| --- | --- | --- |
| CLI `generate` | 一份 RequestContext 覆盖资源加载和生成 | stderr |
| CLI `bench` | 一份 RequestContext 覆盖资源加载、编码和基准运行 | stderr |
| CLI `run` 加载 | 资源加载完成时结束 | stderr |
| CLI `run` 每轮 | 读取输入后启动，成功或失败时结束 | 终端输出流 |
| 库 `Generate/Run` | 复用相同推理编排 | 默认不展示追踪日志 |

每轮聊天分别记录 `prompt_build`、`tokenize`、`prefill`、`decode` 和 `sample`。
等待用户输入不计入请求耗时。RequestContext 的 `prompt_tokens` 是完整输入
长度，`prefill_tokens` 是实际新增的后缀长度；ChatSession 累加实际 prefill
工作量、生成 token 和轮次耗时供 `/stats` 展示。

EOS 计入生成统计但不展示；聊天模式只把非 EOS 的生成 token 写入上下文。
`generate` 通常生成 N 个 token 时有 N-1 次 decode，聊天为了下一轮复用会对
每个非 EOS 的生成 token 执行前向。成功与失败请求均输出最终 summary。

## 回归验证

已移除不存在的 GPU 前向测试注册，保留 CPU 前向并注册实际存在的 Runtime
与 CUDA Tensor 测试。Runtime 的零长度操作、设备信息格式、设备恢复及 Reset
失败状态保持已与既有测试约定对齐；CUDA Tensor 移动后源对象恢复为空。

2026-10-08 对当前源码重新配置、构建并运行全量 CTest：

| 配置 | 注册 | 执行通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: | ---: |
| CPU，MINI_LLAMA_CUDA=OFF | 169 | 157 | 12 | 0 |
| CUDA，真实单 GPU | 169 | 166 | 3 | 0 |

CPU 跳过 8 项 CUDA Runtime、2 项设备 Tensor 和 2 项模型上传用例。
CUDA 单 GPU 跳过 2 项 CPU stub 和 1 项多 GPU 用例。CPU 前向专项 12 项中，
10 项验证 CPU 前向与快照契约，另外 2 项在 GPU 上验证 F32/Q8_0 上传后 CPU
logits 不变；这不是 GPU 前向验证。

多线程 CLI 专项也已通过：线程配置 1/4/0 的 9 份 logits、生成 ID 与 bench
输出一致。19 份 Markdown 本地链接检查和 `git diff --check` 通过。

验证构建目录分别为 `/tmp/mini-llama-sync-audit-20261008` 与
`/tmp/mini-llama-sync-cuda-20261008`，复用现有 googletest 源码。CUDA 使用
`-DCUDAToolkit_ROOT=/tmp/mini-llama-cuda-sdk`，SDK 版本为 12.9，设备为
NVIDIA GeForce RTX 4060 Laptop GPU。沙箱内 CUDA 查询失败会导致 GPU 用例
跳过；表中 CUDA 结果来自允许访问 GPU 的沙箱外运行。

可运行以下命令：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
ctest --test-dir build --output-on-failure -j4
python3 scripts/test_threaded_cli.py
```

CPU 多线程阶段的回归在模型加载、Tokenizer、生成、聊天、检查命令和基准测试之外，新增
`tests/threaded_matmul_test.cc` 当时的 9 项多线程测试（现为 14 项），CMake 引用与当前文件名一致。
该阶段构建成功，全量 CTest **113/113 通过**，`mini-llama-tests` 前缀
筛选结果为 **22/22 通过**，这些数字与历史文档中的 82、95、104 项均为阶段快照。
此前 Backend/CUDA Runtime 阶段记录过 CPU 150 项通过、8 项跳过，CUDA 单 GPU
156 项通过、2 项跳过。这是 CUDA Tensor 与扩展前向测试加入之前的阶段快照，
不是当前测试总数。
