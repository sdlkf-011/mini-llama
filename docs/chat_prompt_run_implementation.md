# Chat Template、PromptBuilder 与 Run 模式实现总结

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

## 阶段目标

本阶段在已有 Chat、Tokenizer、GGUF Reader 和交互式 `run` 模式基础上，审查并
完善 Chat Template / PromptBuilder 模块。工作重点是确认既有设计、修复明确的
行为问题、补齐边界测试，而不是重新实现模板引擎。

模块边界保持为：

```text
GGUF metadata
    ↓
chat template string
    ↓
ChatMessage[] ──PromptBuilder──→ prompt string
                                      ↓
                                  Tokenizer
                                      ↓
                                  token ids
```

`PromptBuilder` 不读取词表、不编码 token，也不依赖具体 Tokenizer。Tokenizer
只消费最终 prompt，不解释 ChatMessage 或 Jinja2 模板。

## 涉及文件

| 文件 | 职责 |
| --- | --- |
| `include/mini_llama/prompt_builder.h` | 保留 `SetChatTemplate()`、`Build()` 接口 |
| `src/prompt_builder.cc` | Plain、内置 Qwen2 和轻量 Jinja2 模板执行 |
| `src/main.cc` | 从 GGUF 装配 chat template，并将其注入 `PromptBuilder` |
| `tests/chat_test.cc` | PromptBuilder、会话和 `run` 模式单元测试 |
| `tests/gguf_tokenizer_test.cc` | GGUF chat template metadata 边界测试 |
| `tests/chat_test.cc` | GGUF → template → PromptBuilder 集成测试 |
| `tests/CMakeLists.txt` | 注册单元测试与 `mini-llama-tests` 集成测试 |

公共 `PromptBuilder` 接口未改变，也没有引入第三方 Jinja2 库。

## GGUF Chat Template 加载

`src/main.cc` 中的 `LoadChatTemplateFromGguf()` 使用一次 `GgufReader::Load()`
取得并校验 metadata。读取优先级为：

```text
tokenizer.chat_template 存在
    └─ 原样返回模板字符串，包括合法空字符串

tokenizer.chat_template 缺失
    ├─ general.architecture == "qwen2" → 返回 "qwen2"
    └─ 其他或缺失                         → 返回空字符串
```

文件缺失、GGUF header 损坏、metadata 截断、类型或数据范围非法时，读取器返回
失败，模板加载函数安全返回空字符串。该过程复用已有 GGUF Reader，不在
PromptBuilder 中增加文件解析逻辑，也不会读取张量数据内容。

`LoadModelAndTokenizer()` 在最终 GGUF 路径确定后加载模板。模板来源独立于
Tokenizer 来源，因此即使调用方显式提供 tokenizer，聊天格式仍与所选模型
GGUF 保持一致。

## `PromptBuilder::Build()` 分发

模板分发顺序为：

1. 模板为空：调用 `BuildPlain()`。
2. 模板等于 `qwen2`：调用 `BuildQwen2()`。
3. 模板包含 `{%` 或 `{{`：编译并执行轻量 Jinja2 子集。
4. 其他无法识别的模板：安全回退 `BuildPlain()`。

Jinja2 路径固定调用：

```cpp
ExecuteTemplate(CompileTemplate(chat_template_), messages, true);
```

这里的 `true` 表示 `add_generation_prompt = true`。模板中相应条件为真时，会在
最终 prompt 末尾生成 assistant generation prefix。当前公共 `Build()` 接口
没有暴露关闭该开关的选项。

模板判断不要求 Jinja 标签必须位于字符串第一个字节，因此带普通文本前缀的
模板也能进入模板执行路径。

## Plain 格式

空模板或未知模板使用以下格式：

```text
System: <content>\n
User: <content>\n
Assistant: <content>\n
Assistant: <space>
```

Plain 模式只输出 `system`、`user` 和 `assistant`，忽略未知角色。空消息历史仍
返回 `Assistant:`，保证生成入口始终存在。

## 内置 Qwen2 格式

每条标准消息使用：

```text
<|im_start|>{role}\n
{content}<|im_end|>\n
```

规则如下：

- 第一条消息不是 `system`，或消息历史为空时，插入默认 system。
- 第一条已经是 `system` 时，保留调用方内容，不重复插入默认 system。
- 未知角色不输出。
- 每条已输出消息均以 `<|im_end|>\n` 结束。
- 最后始终追加 `<|im_start|>assistant\n`。

例如输入：

```cpp
std::vector<ChatMessage> messages = {{"user", "你好"}};
```

输出按字节精确等于：

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

原实现曾检查“任意位置是否存在 system”。当第一条为 user、后续才出现 system
时，这会漏掉默认 system。当前实现改为只检查第一条消息，与 Qwen2 模板规则
一致。

## 轻量 Jinja2 子集

### 编译与执行

模板被编译为以下顺序指令：

- `kText`
- `kOutput`
- `kIf` / `kEndIf`
- `kFor` / `kEndFor`

编译阶段使用栈匹配 `if/endif` 与 `for/endfor`，并计算跳转位置；执行阶段只按
指令和跳转地址运行。未闭合控制块统一跳到模板末尾，多余的结束标签被忽略，
不会产生无界扫描、非法跳转或死循环。

循环目前面向 Qwen2 所需的：

```jinja2
{% for message in messages %}
...
{% endfor %}
```

空 `messages` 会直接跳过循环体。`messages[i]` 在负下标、非数字下标或越界时
求值为空字符串，不抛出异常。

### 表达式

当前支持：

- `message['role']`、`message["role"]`
- `message['content']`、`message["content"]`
- `messages[i]['role']`、`messages[i]["role"]`
- `messages[i]['content']`、`messages[i]["content"]`
- `loop.first`
- `add_generation_prompt`
- 单引号和双引号字符串字面量
- 字符串 `+` 拼接
- `\n`、`\r`、`\t`、`\\` 和引号转义

未知变量和不支持的字段求值为空字符串。

### 条件

当前支持：

- `==`
- `!=`
- `and`
- `or`

`and` 优先级高于 `or`。条件运算符只在字符串字面量之外查找，避免把
`"a and b"` 之类的内容错误拆分。条件递归求值使用 C++ 短路语义。

### Trim marker

`StripJinjaTrimMarkers()` 兼容：

```jinja2
{%- ... -%}
{{- ... -}}
```

当前行为仅移除标签内部两端的 `-` 并继续解析。它不是完整 Jinja2 whitespace
control，不会删除标签外部相邻的空格或换行。Qwen2 模板及当前测试不依赖完整
空白裁剪语义，因此本阶段没有扩大实现范围。

## 边界与安全行为

| 场景 | 当前行为 |
| --- | --- |
| 空消息历史、Plain | 返回 `Assistant:` |
| 空消息历史、内置 Qwen2 | 默认 system 加 assistant prefix |
| 第一条非 system | 插入默认 Qwen2 system |
| 第一条是 system | 保留调用方 system，不重复插入 |
| 多轮 user/assistant | 保持原消息顺序 |
| 未知角色、Plain/内置 Qwen2 | 忽略 |
| 空 content | 保留格式与结束 token，不增加额外空白 |
| 空模板 | Plain fallback |
| 非 Jinja 未知模板 | Plain fallback |
| 未闭合 `if` / `for` | 执行可安全确定的部分后结束 |
| 多余 `endif` / `endfor` | 忽略 |
| 空数组访问 `messages[0]` | 返回空字符串 |
| 损坏或缺失 GGUF | 模板加载返回空字符串 |
| 非 qwen2 architecture 且无模板 | Plain fallback |

模板执行只保证安全和当前 Qwen2 子集语义，不尝试为任意 malformed Jinja2
推导完整 Jinja2 错误恢复结果。

## 与 Run 模式的关系

交互式路径保持为：

```text
ChatSession.messages
    ↓
PromptBuilder::Build()
    ↓
ITokenizer::Encode()
    ↓
KV Cache 前缀复用 / prefill / decode
    ↓
ITokenizer::Decode(完整生成 ID)，再输出新增文本后缀
```

`run` 模式支持 `/help`、`/clear`、`/stats`、`/params` 和 `/exit`。帮助列表由
`Terminal::PrintHelp()` 统一维护；`/params` 展示当前采样参数。斜杠命令不会作为用户
消息送入模型；`/clear` 同时清理消息、token 历史、RadixTree、统计和 KV Cache
上下文。

PromptBuilder 与 Tokenizer 的职责在该路径中保持分离：模板层只生成字符串，
Tokenizer 只处理字符串与 token ID 的转换。

## 测试覆盖

### 单元测试

`tests/chat_test.cc` 使用完整字符串断言覆盖：

- 空历史、空 content、多轮对话与未知角色。
- Qwen2 默认 system 注入、已有 system 保留、后置 system 边界。
- Qwen2 特殊 token、空格和换行的字节级输出。
- `for`、`if`、消息访问、`loop.first`、`add_generation_prompt`。
- 单/双引号、字符串拼接、转义换行、`and/or` 优先级。
- trim marker、非法下标、空数组和未闭合模板。
- `run` 命令、流式刷新和上下文长度拒绝。

`tests/gguf_tokenizer_test.cc` 覆盖：

- 显式 `tokenizer.chat_template` 优先于 architecture fallback。
- 显式空模板保持为空。
- 缺少模板时只有 qwen2 返回 `qwen2`。
- 非 qwen2、缺失文件和 malformed GGUF 安全返回空字符串。

### 集成测试

`tests/chat_test.cc` 中每个测试均带“测试目的”注释，覆盖：

- 合成 GGUF 中的 Qwen2 Jinja 模板完整加载与执行。
- 已有 system 消息。
- qwen2 architecture fallback 与非 qwen2 Plain fallback。
- malformed template 和空历史下标访问。
- Jinja trim marker。
- Plain 多轮格式与未知角色过滤。
- 本地存在真实 Qwen2 GGUF 时，验证真实模板的精确输出。

该目标使用 `mini-llama-tests.` CTest 前缀，因此阶段验收命令会同时执行 GGUF
Tokenizer 与 Chat Template 集成测试。

## 构建与验收

```bash
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
```

该阶段历史结果（不代表当前工作区重新验证）：

- 构建成功。
- `mini-llama-tests`：13/13 通过。
- 全量测试：80/80 通过。
- `git diff --check` 通过。

## 当前未支持的 Jinja2 功能

本模块不是完整 Jinja2，当前明确不支持：

- `else`、`elif`、`set`、`macro`、`include`。
- filter、函数调用、测试表达式和对象通用属性访问。
- 括号表达式、算术运算和完整类型系统。
- 通用循环目标、完整嵌套循环上下文和 `loop` 的其他属性。
- 完整 whitespace control。
- Jinja2 的异常诊断、行列号和严格未定义变量模式。

这些功能不是当前 Qwen2 Chat Template 所需能力。本阶段保持轻量实现，没有为
追求语法完整度引入额外复杂度。

## 一句话总结

本阶段在不改变 PromptBuilder 公共接口、不混入 Tokenizer 职责的前提下，打通
GGUF Chat Template 加载、Qwen2 fallback、轻量 Jinja2 执行与真实模型回归测试，
并通过字节级断言固定了系统消息注入和 assistant generation prefix 行为。
