# 真实 Qwen2 GGUF 端到端 Smoke Test 实现总结

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。
>
> 本文的真实模型输出、计时和 82 项测试结果保留原阶段快照；参数装配和当前边界已按现有实现更新。
> 最新回归覆盖与入口约定见[命令行与模块同步说明](cli_integration_status.md)。

## 阶段目标

本阶段使用仓库内的真实模型
`models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf`，验证现有 GGUF Reader、GGUF
Loader、`GgufTokenizer`、`PromptBuilder`、Forward、KV Cache 和 Sampler 能否通过
进程级 CLI 组成完整推理链路。

工作范围限定为：

- 先运行已有构建、测试和 CLI，复现真实模型路径上的问题。
- 检查 GGUF metadata、模型配置、内嵌 tokenizer 和 chat template。
- 修复 CLI 编排、追踪和终端统计中的明确集成缺口。
- 不重新设计或重写已经实现的模型和推理模块。

最终验证链路为：

```text
Qwen2 GGUF
  ├─ metadata ───────────────→ ModelConfig
  ├─ tensor info/data ───────→ MiniLlamaModel
  ├─ tokenizer.ggml.* ───────→ GgufTokenizer
  └─ tokenizer.chat_template → PromptBuilder
                                      ↓
prompt / chat messages → token ids → ForwardBatch
                                      ↓
                                  KV Cache
                                      ↓
                                   Sampler
                                      ↓
                              streaming decode / exit
```

## 初始复现结果

构建和已有 `mini-llama-tests` 集成测试初始即通过，真实 GGUF 的只读检查和
`generate` 数值链路也能运行。复现发现的问题位于应用编排层，而不是模型计算
模块：

1. `./build/mini-llama run models/chat ...` 中的裸位置参数没有被识别为模型目录，
   最终报错要求显式提供 `--model`。
2. 仓库已有 `RequestContext`、事件格式化和 summary 格式化实现，但进程级 CLI
   没有接入，因此成功生成后没有 `trace summary`。
3. `/stats` 已显示累计 prompt token、生成 token、耗时和吞吐，但缺少消息数和
   当前上下文 token 数。

以下真实链路在修复前已经确认正常，因此本阶段没有修改其内部实现：

- Qwen2 GGUF metadata 到 `ModelConfig` 的动态字段映射。
- Q8_0 权重装配和量化 Linear 分发。
- `GgufTokenizer` 从 GGUF metadata 直接加载 tokens、merges 和特殊 token。
- Qwen2 的 NeoX RoPE 分支。
- Forward、KV Cache、Sampler 和流式 token 解码。
- GGUF chat template 到 `PromptBuilder` 的注入。

## 最小修改

### 位置模型参数

`ParseOptions()` 现在同时支持：

```bash
mini-llama run models/chat -n 10 --seed 42
mini-llama run --model models/chat -n 10 --seed 42
```

当前 `ParseOptions()` 统一消费路径和生成选项，不再原样转发剩余参数。命令后
紧跟的位置参数可以作为模型路径；`ResolveModelPaths()` 优先选择目录中的
完整 JSON+BIN 配对，否则要求唯一 GGUF 候选。

帮助文本同步展示位置参数形式，同时保留 `--model` 写法。

### 请求追踪接入

本阶段复用已有 `RequestContext`，没有引入新的日志或计时框架。进程级流程为：

```text
ParseOptions
    ↓
StartRequest(mode, "cpu", model_path)
    ↓
LoadModelAndTokenizer ──记录 model_load
    ↓
GenerateTokens
    ├─ EncodePrompt  ──记录 tokenize / prompt_tokens
    ├─ ForwardBatch ──记录 prefill / prefill_tokens
    ├─ Sample       ──累计 sample / generated_tokens
    └─ decode step  ──累计 decode / decode_tokens
    ↓
Finish
    ↓
trace events
trace summary（最后一行）
```

`generate` 的 trace 输出到 stderr，结果输出到 stdout。`run` 的加载 trace 输出到
stderr，每轮 trace 输出到交互终端输出流。成功请求以 `status=ok` 收尾；资源加载或生成异常会写入
`RequestContext::error`，并以 `status=error` 收尾。

原有无追踪 `Generate()` 接口保留，并转发到内部 `GenerateTokens()`；现有单元
测试和嵌入调用接口未被替换。

`generate` 的 `decode_tokens` 表示为了取得下一轮 logits 而执行的 decode
Forward 次数。最后一个已输出 token 不需要再次前向，因此 `-n 3` 正常情况下可
出现 `generated_tokens=3`、`decode_tokens=2`。

### 交互统计

`Terminal::PrintStats()` 在既有统计上补充：

- `Messages`：当前 `ChatSession.messages` 数量。
- `Context tokens`：当前已评估并保存在会话 token history 中的 token 数量。

完整输出包含：

```text
Messages: ...
Context tokens: ...
Prompt tokens: ...
Generated tokens: ...
Elapsed: ... ms
Speed: ... tokens/s
```

`/clear` 仍使用既有逻辑同步清理消息、token history、前缀树、统计和 KV Cache
上下文。

## 真实 GGUF 检查

模型文件：

```text
models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
size: 531065536 bytes
```

执行：

```bash
./build/mini-llama inspect-gguf \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

结果为 exit 0，并确认：

| 项目 | 实测值 |
| --- | --- |
| GGUF version | 3 |
| metadata 数量 | 26 |
| tensor 数量 | 290 |
| `general.architecture` | `qwen2` |
| `qwen2.embedding_length` | 896 |
| `qwen2.feed_forward_length` | 4864 |
| `qwen2.block_count` | 24 |
| `qwen2.context_length` | 32768 |
| query / KV heads | 14 / 2 |
| RoPE frequency base | 1000000 |
| `tokenizer.ggml.tokens` | 151936 个 string |
| `tokenizer.ggml.token_type` | 151936 个 int32 |
| `tokenizer.ggml.merges` | 151387 个 string |
| BOS token | 151643 |
| EOS token | 151645 |
| `tokenizer.chat_template` | 存在且非空 |

tensor 列表从 Q8_0 `token_embd.weight` 开始，包含 24 层 Attention/FFN 权重、
Q/K/V bias 和 norm，最后包含 `output_norm.weight`。这同时验证了 metadata 数组和
完整 tensor directory 均可由现有 GGUF Reader 读取。

## 真实 Generate 验证

执行：

```bash
./build/mini-llama generate \
  --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  --prompt "The capital of France is" \
  --n-predict 3 \
  --seed 42
```

最终执行结果：

- exit code 为 0。
- prompt 编码为 5 个 token。
- prefill 处理 5 个 token。
- Sampler 生成 3 个 token。
- 为前两个生成 token 执行 2 次 decode Forward。
- 最后一行为 `trace summary`，其中 `mode=generate status=ok`。

一次实测 summary 为：

```text
trace summary: ... mode=generate status=ok backend=cpu \
model=models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
prompt_tokens=5 prefill_tokens=5 decode_tokens=2 generated_tokens=3 \
model_load_ms=1772.050 tokenize_ms=0.099 prefill_ms=17627.957 \
decode_ms=6864.502 sample_ms=3.984 total_ms=26834.826
```

时间数据只记录本次 CPU 环境，不作为跨机器性能基准。3-token smoke test 只用于
验证模型加载、编码、prefill、decode、KV Cache、采样、解码和正常退出，不使用
短输出文本质量判断模型正确性。

## 真实交互模式验证

执行：

```bash
./build/mini-llama run models/chat -n 10 --seed 42
```

交互步骤：

```text
Hello
/stats
/exit
```

实测行为：

- `models/chat` 自动找到唯一的 Qwen2 GGUF。
- GGUF 中的 Jinja chat template 经 `PromptBuilder` 生成 system、user 和
  assistant generation prompt。
- assistant 内容逐 token 刷新，实测回复为
  `Hello! How can I assist you today?`。
- 模型在上限 10 之前生成 EOS，本轮实际输出 9 个 token，属于正常行为。
- `/stats` 输出 Messages=2、Context tokens=29、Prompt tokens=20、Generated
  tokens=9、Elapsed=103072.22 ms、Speed=0.09 tokens/s。
- `/exit` 输出 `Bye.`，进程 exit code 为 0，最终 summary 包含
  `mode=run status=ok`。

交互首轮包含默认 system prompt 和 Qwen2 控制 token，prefill 长于纯文本
`generate`；当前标量 CPU 路径下耗时较长不影响本阶段稳定性结论。

## 测试覆盖与结果

按要求重新执行：

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
```

结果：`mini-llama-tests` 13/13 通过。

新增和相关测试还执行了：

```bash
./build/tests/mini-llama-chat-tests
./build/tests/mini-llama-request-context-tests
ctest --test-dir build --output-on-failure
```

结果：

- Chat 单元测试 11/11 通过。
- RequestContext 单元测试 2/2 通过。
- 全量 CTest 82/82 通过。
- `git diff --check` 通过。

新增断言覆盖：

- `/stats` 输出消息数和上下文 token 数。
- `run MODEL_OR_DIRECTORY` 的位置参数进入模型资源解析，而不是误传给生成参数。
- 模型路径加载失败时 trace 以 `mode=run status=error` 收尾。

真实推理没有出现崩溃、越界或非法内存访问迹象，因此本阶段没有额外启动 ASan
构建。若后续模型或输入复现崩溃，再按问题定位范围启用 ASan。

## 涉及文件

| 文件 | 修改内容 |
| --- | --- |
| `src/main.cc` | 位置模型参数、RequestContext 接入、阶段计时和 CLI 帮助 |
| `src/terminal.cc` | `/stats` 增加消息数和上下文 token 数 |
| `tests/unit/chat_test.cc` | stats、位置模型参数和错误 trace 回归测试 |
| `docs/qwen2_gguf_end_to_end_smoke_test.md` | 本阶段实现与实测总结 |

没有修改 GGUF Reader、GGUF Loader、`GgufTokenizer`、`PromptBuilder`、Forward、
KV Cache、Sampler 或模型权重格式。

## 当前边界

- `generate` 使用一次 RequestContext 追踪加载与推理；`run` 先追踪资源加载，
  再为每轮输入建立独立 RequestContext，记录阶段耗时、token 数和错误状态。
  ChatSession 与 `/stats` 维护累计会话统计，等待用户输入不计入逐轮耗时。
- 模型目录优先选择完整的 JSON+BIN 配对，否则要求唯一 GGUF 候选；多个候选
  会要求明确指定文件，不依赖遍历顺序选择模型。
- JSON 清单的 BOS/EOS/UNK 必须与实际分词器一致，显式词表覆盖也执行校验。
- `/help` 由 Terminal 统一展示所有命令，包含 `/params`。
- trace 时间受 CPU、构建类型和系统负载影响，不作为模型数值正确性条件。
- 本阶段验证的是仓库内指定 Qwen2 Q8_0 模型；其他架构和未支持量化类型仍受现有
  GGUF Reader/Loader 能力边界约束。

## 一句话总结

本阶段没有重写已有推理模块，而是补齐真实 Qwen2 GGUF 链路最后的 CLI、trace
和终端统计接线，并通过指定模型完成 inspect、单轮 generate、交互 run 以及
82 项全量测试验证，最终请求均能稳定生成 token 并以 `status=ok` 正常退出。
