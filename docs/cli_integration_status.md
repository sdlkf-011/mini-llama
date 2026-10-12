# 命令行与模块同步说明

本文描述当前 `src/main.cc` 与已实现模块的接入约定。分支阶段文档中的实现范围、
真实模型输出和测试快照保留历史含义；当前入口行为以本文和源码为准。
调试接口、计时口径和 `bench` 的详细设计见
[调试与基准测试阶段实现汇总](benchmark_debug_implementation.md)。
CPU 调度、F32 SIMD 和测试细节见
[CPU 多线程与 F32 SIMD 实现说明](cpu_threaded_inference_implementation.md)。

## 构建与入口

所有 `src/*.cc` 已纳入 `mini_llama_core`；`cuda_ops.cu`、`cuda_quant.cu`、
`cuda_attention.cu` 仅在
CUDA 构建中编译，链接 Runtime 和 cuBLAS，C++ / CUDA 标准均为 C++17。
`mini-llama` 可执行目标定义
`MINI_LLAMA_BUILD_EXECUTABLE`，启用同一个 `src/main.cc` 中的全局 `main()`。
测试可直接使用流注入的 `RunMain()`；库入口 `Generate()`、`Run()` 继续接受已加载
的模型与分词器。

| 命令 | 行为 |
| --- | --- |
| `generate` | 单轮生成，默认教学模型、`hello`、16 个生成 token |
| `run` | 多轮聊天，需要模型路径，默认每轮最多生成 64 个 token |
| `bench` | 默认生成预算 16、预热 1 次、测量 3 次，统计前向与采样，可用 `--verbose` 输出调试信息 |
| `inspect` | 检查 JSON 清单或 GGUF 元数据，不加载模型权重或分词器 |
| `inspect-gguf` | 直接检查 GGUF 文件头、元数据和张量目录 |
| `--cuda-info` | 查询 CUDA Runtime 构建状态和真实 GPU 信息，失败时返回非零 |

CPU 线程配置已通过 `--threads` 接入推理命令，后端选项和 CUDA Runtime 诊断也已接入。
库层已有 CUDA Tensor、F32/Q8_0/Q4_0/Q4_1 上传快照、cuBLAS F32 Linear、
Ops/Quant Kernel、GPU Attention 与 KV Cache。`ForwardToken()` 无快照时执行
CPU 前向，存在快照时执行完整设备前向；缺失或损坏权重明确报错。
CLI `generate/run/bench --backend cuda` 已调用上传入口，激活和 KV 历史在 GPU，
只下载最终 logits 供 CPU 采样。正常路径 H2D 和 CPU Attention fallback 均为 0。
运行统计记录实际成功的算子调用、KV 逻辑字节和边界拷贝；上传统计独立。
后端参数、Runtime 当前行为与验证结果见
[后端选择与 CUDA Runtime 阶段实现汇总](backend_selection_cuda_runtime_implementation.md)。
整模型 `QuantizeModelToQ80/Q40()` 通过 `--quant q8_0|q4_0` 接入三个推理命令；转换所有 Linear，
保持 embedding/norm/bias，成功改变权重后清除旧 CUDA 快照。已有 CPU 量化
Linear 可执行 GGUF Q8_0/Q4_0/Q4_1 权重，CUDA 上传与前向也支持这三种格式。
`bench` 输出权重实际大小/F32 等价大小；CUDA 自动上传，无需额外上传参数。

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

### 后端与量化

`generate/run/bench` 默认 `--backend cpu`。CUDA 请求先检查构建与设备，再
加载资源、可选重量化、上传权重并重置运行计数；初始化或推理失败直接报错。
`--device` 仅与 CUDA 配合，默认设备 0。`inspect --backend cuda` 只检查参数
和元数据，即使未构建 CUDA 也可使用；检查命令不接受 `--quant`。

`--quant q8_0|q4_0` 只改变 Linear，Embedding、Norm 和 bias 保持 F32。
不传则保持模型加载后的格式；Q4_1 可直接从 GGUF 加载并执行，但不是 CLI
重量化目标。已加载模型的库入口 `Run()` 也接受后端与量化选项，通过模型副本
准备所需权重，保持调用方模型。直接 `Generate()` / `ForwardBatch()` 则使用
传入模型的快照状态。

```bash
./build-cuda/mini-llama generate models/tiny --backend cuda -p hello -n 4
./build-cuda/mini-llama run models/chat --backend cuda -n 16
./build-cuda/mini-llama bench models/tiny --backend cuda --quant q4_0 -n 4
```

CUDA 摘要的 `gpu memory used` 仅指上传权重的显存，不含 KV Cache 和临时激活。
generate 的运行摘要累计当前请求的 Prefill + Decode；bench 对应最后一次正式
测量；run 的设备计数自权重准备后跨轮累计，`/clear` 重建会话与 Cache，
不重置模型设备计数。ForwardBatch 逐 token 调用原有 ForwardToken，每个
token 都计算、下载一份 logits，单次 D2H 为 `vocab_size * 4` 字节。整批只
返回最后一份 logits，不保留新增内部前向接口。共享快照计数采用单调用线程约定。
设备数据流、计数公式与上下文切换约定见
[设备驻留 CUDA Decode](device_resident_cuda_decode.md)。

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
每个算子实际参与的线程数。达到并行阈值时向常驻工作线程提交任务，嵌套调用
在当前工作线程串行执行，避免等待自身。Q8_0/Q4_0/Q4_1 Linear 已按 batch 行与输出通道展平并行；
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

`RunBench()` 通过 `MeasureBenchmark()` 重复调用 `debug.h` 已有的 `RunBenchmark()`，
默认 `--warmup 1 --repeat 3`；预热次数可为零，测量次数必须为正数。每次创建
独立上下文并从零重放，预热不计入结果。prefill 包括批次构造与前向，decode
包括后续前向、采样和 verbose 输出，首次采样不计时。报告每次测量的均值及
prefill + decode 的最小值、中位数、最大值。`debug.cc` 按提供的原始结构实现，
不保留跨文件的内部测量入口，也不单列 sample/output 时间；`debug.h` 未扩展。
首个生成 token 直接从 prefill logits 采样，因此生成 N 个 token 通常
只有 N-1 次 decode。`--verbose` 输出 logits Top-k 和 KV Cache 状态，诊断时可用，
正式测量吞吐量时应关闭。
结果同时输出 `tokens/s (total):`、`tokens/s (Decode):`，保留原有
`throughput:` 和 `decode throughput:` 标签。

`bench --quant` 在保留原模型的基础上输出实际权重字节、F32 等价字节，以及
同一 prompt 最后位置 logits 相对模型原始格式的最大/平均绝对误差。原始格式
可能含量化权重，不是全 F32 Golden；该误差不评估长文本质量。额外两次前向
位于性能计时之外；CUDA 运行摘要在误差计算之前打印，仅覆盖基准工作量。
`--verbose` 的 KV shape 来自上下文保留的 CPU Cache 配置，GPU 历史的实际
读写应以 CUDA KV 统计为准。

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

2026-10-12 在恢复原有 `context.h`、`cuda_runtime.h`、`debug.h`、`cuda_tensor.h`
接口后重新构建并运行全量 CTest，环境为 Release、GCC 13、CUDA 12.3 /
Clang 18、RTX 4060 Laptop（CC 8.9）：

| 配置 | 注册 | 实际通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: | ---: |
| CPU，MINI_LLAMA_CUDA=OFF | 236 | 180 | 56 | 0 |
| CUDA，真实单 GPU，串行执行 | 236 | 228 | 8 | 0 |

CPU 跳过 56 项 CUDA 用例；GPU 跳过 7 项 CPU Stub 用例和 1 项多 GPU 用例。
本轮覆盖模型对象身份、整批边界预检、逐 token logits 及传输计数、重复测量、线程池复用、
嵌套与并发任务、算子工作区和句柄复用，以及 Python CLI/报告解析和 NumPy golden。
调试测试按恢复后的语义验证独立上下文、decode 输出计时与输出流适配恢复。
没有新增 workspace 模块；CudaTensor 仍使用原有独占 Buffer，不包含通用资源池。
完整说明与性能历史记录见[五项推理改进与验证](inference_improvements.md)。
此前恢复原 Tensor 的日志与 JUnit 为 `/tmp/mini-llama-original-tensor-{cpu,cuda}.{log,xml}`。
恢复提供的 debug.cc 实现、移除新增内部前向入口后重新构建与回归，当前日志/JUnit 为
`/tmp/mini-llama-original-batch-{cpu,cuda}.{log,xml}`。

### 2026-10-12 全项目检查

此次检查覆盖源码构建注册、模块接口、模型加载与分词、CPU/CUDA 前向与缓存、
采样与聊天复用、CLI/benchmark、测试脚本和全部 27 份 Markdown 的本地链接。
37 个源码文件均纳入相应构建，测试源码均有注册；阶段历史数字保留并明确标记。

检查修正了近期改动之外的两处实现问题：CPU valuePtr 对空缓存缺少校验，以及
ToTensor 在 F32 数据长度错误时静默返回全零。现在 KV 访问检查秩、维度与存储；
写入先核对两个缓存及输入长度，失败不部分写入。F32 权重转换和对应 Linear
明确拒绝错误存储。新增 3 个回归用例，全量结果为上表的 236 项。

同步修正 KV 指针的单位置/单头说明、旧线程池描述、线程用例数量及 GGUF
Tokenizer 测试筛选规则。当前 Tensor 仍公开 shape/data，调用方须保持一致；
模板仍为 Jinja 子集，CUDA 统计仍按单调用线程约定，未扩展为服务端多请求调度。

本次 CPU/CUDA Release 全量回归日志及 JUnit：
`/tmp/mini-llama-project-audit-{cpu,cuda}.{log,xml}`。Python 脚本语法、Shell 语法、
Markdown 本地文件及锚点链接、git diff --check 均通过。

### 2026-10-10 历史回归

2026-10-10 同步后重新配置、构建并运行全量 CTest，环境为 Release、
GCC 13、CUDA 12.3 / Clang 18、RTX 4060 Laptop（CC 8.9）：

| 配置 | 注册 | 实际通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: | ---: |
| CPU，MINI_LLAMA_CUDA=OFF | 212 | 164 | 48 | 0 |
| CUDA，真实单 GPU | 212 | 204 | 8 | 0 |

CPU 跳过依赖 CUDA 构建或 GPU 的 48 项；GPU 跳过 7 项 CPU Stub 专用用例和
1 项多 GPU 用例。GPU 结果来自可访问真实设备的环境，未将沙箱内的跳过计作通过。
`backend-cli-smoke`、`threaded-cli-smoke`、`golden-logits` 均计入全量；Python
脚本内部子用例不另加到 CTest 注册数。

后端 CLI 覆盖默认 CPU、非法参数、无设备、元数据检查、CPU/CUDA logits 对照、
generate/bench/run 与 Q8_0/Q4_0 重量化；聊天逐轮检查 `status=ok` 和实际生成，
覆盖连续会话及 `/clear` 后重放。线程 CLI 比较 1/4/自动配置及 9 份 logits。
CUDA Forward 测试覆盖四种 Linear、GQA、两种 RoPE、bias、缓存历史、快照校验
与 0 H2D / 1 D2H；NumPy golden 验证 CPU logits。

真实 Qwen2 GGUF 的 inspect、CPU/CUDA 8 步生成 logits 对照、CUDA bench 和
三轮聊天验证见[设备驻留 Decode 本次同步复验](device_resident_cuda_decode.md#项目链路与文档同步复验)。
该阶段构建日志、CTest/JUnit 和真实模型输出保存在 `.cache/project-sync-20261010/`，
缓存不属于提交内容。只验证单 GPU，未运行 compute-sanitizer。

以下为此前 Ops/Quant 阶段记录，保留当时的接入范围，不能作为当前总数。
当时 CPU 161 项通过、28 项跳过，CUDA 单 GPU
182 项通过、7 项跳过；均注册 189 项，失败为 0。
独立 Ops/Quant 专项匹配 8 项：CPU 3 项通过、5 项跳过，GPU 5 项通过、
3 项 CPU Stub 跳过。原始日志见[本阶段验证记录](cuda_ops_quant_implementation.md)。
当时完整设备前向和 CUDA CLI benchmark 尚未接入，未计入上述历史结果。

以下为较早的 CUDA Tensor 阶段记录，保留当时的接入范围，不能作为当前总数。

本阶段接口、所有权与测试覆盖见
[CUDA Tensor 与模型权重上传阶段总结](cuda_tensor_model_weights_implementation.md)。

该阶段移除不存在的 GPU 前向测试注册，保留 CPU 前向并注册实际存在的 Runtime
与 CUDA Tensor 测试。Runtime 的零长度操作、设备信息格式、设备恢复及 Reset
失败状态保持已与既有测试约定对齐；CUDA Tensor 移动后源对象恢复为空。

设备存储阶段对当时源码重新配置、构建并运行全量 CTest：

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

历史验证分别使用独立的 CPU/CUDA 构建目录，复用现有 googletest 源码。CUDA 使用
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
`tests/threaded_matmul_test.cc` 当时的 9 项多线程测试（现为 17 项），CMake 引用与当前文件名一致。
该阶段构建成功，全量 CTest **113/113 通过**，`mini-llama-tests` 前缀
筛选结果为 **22/22 通过**，这些数字与历史文档中的 82、95、104 项均为阶段快照。
此前 Backend/CUDA Runtime 阶段记录过 CPU 150 项通过、8 项跳过，CUDA 单 GPU
156 项通过、2 项跳过。这是 CUDA Tensor 与扩展前向测试加入之前的阶段快照，
不是当前测试总数。
