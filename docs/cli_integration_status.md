# 命令行与模块同步说明

本文描述当前 `src/main.cc` 与已实现模块的接入约定。分支阶段文档中的实现范围、
真实模型输出和测试快照保留历史含义；当前入口行为以本文和源码为准。

## 构建与入口

所有 `src/*.cc` 已纳入 `mini_llama_core`。`mini-llama` 可执行目标定义
`MINI_LLAMA_BUILD_EXECUTABLE`，启用同一个 `src/main.cc` 中的全局 `main()`。
测试可直接使用流注入的 `RunMain()`；库入口 `Generate()`、`Run()` 继续接受已加载
的模型与分词器。

| 命令 | 行为 |
| --- | --- |
| `generate` | 单轮生成，默认教学模型、`hello`、16 个生成 token |
| `run` | 多轮聊天，需要模型路径，默认每轮最多生成 64 个 token |
| `inspect` | 检查 JSON 清单或 GGUF 元数据，不加载模型权重或分词器 |
| `inspect-gguf` | 直接检查 GGUF 文件头、元数据和张量目录 |

CUDA、线程配置、benchmark 和整模型重新量化尚未接入 CLI。已有量化 Linear 可
运行 GGUF 中的 Q8_0/Q4_0/Q4_1 权重；单张量量化与模型权重字节统计仍是库能力。

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

生成长度选项包括 `--n-predict/-n` 和旧版 `--max-tokens`。CLI `generate` 拒绝
超过上下文的完整生成预算，`-n 0` 只执行 prefill；`run` 要求正数响应预算并根据
当前剩余窗口截断。既有库调用保留按窗口剩余容量截断的行为。

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

运行以下命令验证当前源码：

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure -j2
```

新增回归覆盖三个特殊 ID 的逐项冲突、自动与显式词表来源、ASCII 清单冲突、
非默认特殊 ID 的一致配置，以及 Terminal 与聊天入口的完整帮助一致性。
本次同步修复后，构建成功，全量 CTest **95/95 通过**。
历史文档中的 82 项测试是当时的快照，不代表当前测试数量。
