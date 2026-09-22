# Chat、PromptBuilder 与 Run 模式实现汇总

本分支在已有模型、Tokenizer、KV Cache、前向计算和采样器基础上，完成交互式 `run` 模式、聊天会话管理与轻量级 Jinja2 Chat Template 执行器。`src/main.cc` 仍作为核心库编排入口，不定义进程级 `main` 函数，也不负责模型加载。

## Run 模式

- `Run` 支持 `generate` 与 `run`；`run` 使用可注入的输入/输出流，便于终端运行和单元测试。
- 支持 `/help`、`/clear`、`/stats`、`/exit`；命令不会进入模型推理。
- 每轮执行“构建 prompt → 编码 → prefill → 逐 token decode → 更新会话”。解码 token 会立即写入并刷新输出流。
- `--prompt` 可作为 `run` 的首轮输入；采样参数继续复用现有 `SamplingParams` 与 `MiniSampler`。

## 会话与缓存

- `ChatSession` 保存消息历史、已计算 token、RadixTree 前缀、采样参数和累计统计。
- `/clear` 同时清理会话、前缀树、统计和 `MiniLlamaContext`，避免旧 KV Cache 参与新会话。
- `main.cc` 通过 `LongestCachedPrefix` 与当前上下文 token 公共前缀共同确定可复用长度；回退上下文位置后只 prefill 未命中的后缀。该策略保证仅复用当前 KV Cache 中真实存在的数据。

## PromptBuilder

- 空模板使用 Plain 格式，`qwen2` 使用内置 Qwen2 fallback；未知模板回退 Plain 格式。
- Jinja2 子集支持 `{{ ... }}`、`for message in messages`、`if`、`message`/`messages[i]` 访问、`loop.first`、`add_generation_prompt`、字符串拼接、比较与逻辑运算，以及 trim marker。
- 执行器分为 `EvalContext`、表达式/条件求值、指令编译和指令执行四层；异常或不完整模板按安全的空结果/模板末尾处理，不引入第三方依赖。

## 扩展边界

- 当前未实现 GGUF 聊天模板读取或模型加载；后续模块可将模板文本注入 `PromptBuilder`，交互循环已通过参数接收它。
- 当前前缀复用依赖单个 `MiniLlamaContext` 的 KV Cache；若后续增加 KV 快照或多请求缓存，只需替换 `FindReusablePrefix`/上下文准备策略。
- 未添加 `main.h`，保持现有库入口与调用方式。

## 测试

本阶段只维护 `tests/unit/chat_test.cc` 与 `tests/unit/gguf_tokenizer_test.cc`：前者覆盖会话、模板、命令和流式输出，后者覆盖现有 GGUF Tokenizer 的加载与损坏文件处理。

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

当前完整测试矩阵为 45 项，全部通过。
