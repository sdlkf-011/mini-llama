# GGUF Tokenizer 集成实现总结

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

## 阶段目标

本阶段打通 GGUF 模型从 tokenizer metadata 到实际生成入口的完整链路，并保留
两条可互相校验的 tokenizer 路径：

```text
路线 A：GGUF ──导出脚本──→ vocab.json + merges.txt
                              + special_tokens.json
                              ↓
                         BpeTokenizer

路线 B：GGUF metadata ──→ GgufTokenizer
```

路线 A 便于人工检查和问题定位，路线 B 是运行
`mini-llama generate --model MODEL.gguf` 时的默认路径。两条路径使用相同的
GPT-2 byte-level BPE 语义，并通过测试确认相同文本得到相同 token ID。

## 涉及文件

| 文件 | 本阶段职责 |
| --- | --- |
| `scripts/export_gguf_tokenizer.py` | 从 GGUF 导出可检查、可由 `BpeTokenizer` 加载的文件 |
| `src/gguf_tokenizer.cc` | 直接读取 GGUF tokenizer metadata，完成编码和解码 |
| `src/bpe_tokenizer.cc` | 修正流式单 token 解码，使导出路径可直接用于生成 |
| `src/main.cc` | 选择模型、tokenizer、回退文件和 chat template |
| `tests/gguf_tokenizer_test.cc` | tiny fixture 与真实 Qwen2 的集成测试 |
| `tests/CMakeLists.txt` | 注册本阶段的 `mini-llama-tests` 测试前缀 |
| `CMakeLists.txt` | 从现有 `src/main.cc` 构建 `mini-llama` 命令行目标 |

本阶段没有为 tokenizer 增加新的公共头文件接口。词表提示路径和 chat template
读取辅助函数都保留在 `src/main.cc` 装配层内部。

## GGUF tokenizer 导出工具

### 用法

在项目根目录指定输入 GGUF 和输出目录：

```bash
python3 scripts/export_gguf_tokenizer.py \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  /tmp/qwen2-tokenizer
```

省略输出目录时，默认写入 GGUF 同目录下的 `<模型名>-tokenizer/`：

```bash
python3 scripts/export_gguf_tokenizer.py \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

### 输出文件

```text
qwen2-tokenizer/
├── vocab.json
├── merges.txt
├── special_tokens.json
├── chat_template.txt
└── tokenizer_metadata.json
```

- `vocab.json`：token 字符串到 token ID 的 JSON 对象。
- `merges.txt`：按 GGUF 数组顺序保存 BPE merge rank，首行为
  `#version: 0.2`。
- `special_tokens.json`：保存 BOS、EOS、UNK、PAD 等控制 ID，以及识别出的
  特殊 token 字符串。
- `chat_template.txt`：原样保存 `tokenizer.chat_template`；metadata 不含该键
  时不生成此文件。
- `tokenizer_metadata.json`：保存全部 `tokenizer.*` metadata，供人工检查。

脚本支持 GGUF v2/v3，检查 magic、metadata 数量、字符串和数组大小、重复键、
UTF-8 合法性以及重复 token。缺少 `tokenizer.ggml.tokens` 时会明确失败。

## `GgufTokenizer` 直接加载

`CreateGgufTokenizer()` 保持既有公共工厂接口。内部 `Load()` 使用项目已有的
GGUF metadata getter 读取：

- `tokenizer.ggml.tokens`
- `tokenizer.ggml.merges`
- `tokenizer.ggml.bos_token_id`
- `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.unknown_token_id` 或 `tokenizer.ggml.unk_token_id`

该实现只扫描 metadata，不会为了创建 tokenizer 将整个 GGUF 权重文件读入
内存。真实 531 MB Qwen2 模型的 tokenizer 加载因此只处理 tokenizer 数据。

控制 token ID 以 GGUF metadata 为准。只有 metadata 没有对应字段时，才从
`<unk>`、`<s>`、`</s>`、`<bos>` 和 `<eos>` 等词面回退推导，避免 Qwen2 词表中
兼容 token 覆盖真实 BOS/EOS：

```text
BOS = 151643
EOS = 151645
```

## 编码流程

`GgufTokenizer::Encode()` 的处理顺序为：

1. 在当前位置优先匹配特殊 token，候选按字符串长度降序排列。
2. 普通文本只处理到下一个特殊 token 之前，避免特殊内容参与 BPE。
3. 使用轻量预分词规则拆分空白、字母数字、UTF-8 文本和标点。
4. 将原始字节转换为 GPT-2 bytes-to-unicode 符号。
5. 反复合并当前 merge rank 最小的相邻符号。
6. 将最终符号映射到 token ID，缺失时回退到 UNK ID。

因此 Qwen2 特殊 token 会整体编码：

```text
<|im_start|> → [151644]
<|im_end|>   → [151645]
```

## 整体解码

`GgufTokenizer::Decode()` 采用整体解码：

1. 按 token 序列顺序拼接所有合法 token 的内部字符串。
2. 按 UTF-8 码点扫描拼接结果。
3. 命中 GPT-2 unicode-to-byte 逆表时恢复原始字节。
4. 非 byte-level 映射字符保持原样。

`DecodeToken()` 复用同一条解码路径。这样完整序列和流式单 token 输出使用一致
规则，同时整体解码不会在 token 边界提前处理内部表示。

`BpeTokenizer::DecodeToken()` 同样改为执行 bytes-to-unicode 逆映射，避免使用
导出词表生成时将空格显示成 `Ġ` 等内部符号。

## 命令行装配流程

`src/main.cc` 中的 `LoadModelAndTokenizer()` 按以下顺序工作：

```text
--model 路径
    ↓
ResolveModelPaths：目录优先 JSON+BIN，否则查找唯一 GGUF 候选
    ↓
以下为选定 GGUF 后的装配分支
    ↓
LoadGgufModel() 加载模型
    ↓
是否显式提供 --tokenizer？
    ├─ 是：从目录或 vocab.json 提示加载 tokenizer
    └─ 否：CreateGgufTokenizer() 读取内嵌 metadata
                ↓ 失败
            查找 GGUF 同目录的 vocab.json / merges.txt
    ↓
从最终 GGUF 读取 tokenizer.chat_template
```

显式 `--tokenizer` 接受导出目录或其中的 `vocab.json`。当同目录存在
`merges.txt` 时调用已有 `CreateBpeTokenizer()`；否则尝试既有 JSON 词表格式。
该辅助选择逻辑属于 `main.cc` 内部实现，没有扩展 `tokenizer.h`。

未选中完整 JSON+BIN 配对时，目录存在多个 GGUF 候选会拒绝自动选择，避免结果
依赖遍历顺序；候选格式按文件魔数判断。旧版 `--config`、`--weights` 和 `--vocab`
路径继续保留。JSON 模型会校验清单的 BOS/EOS/UNK 与实际分词器 ID，包括显式
词表覆盖。完整入口规则见[同步说明](cli_integration_status.md)。

生成参数同时接受项目原有选项和 llama.cpp 风格别名：

```text
--max-tokens N
--n-predict N
-n N
```

## Chat template

运行路径从最终选定的 GGUF 读取 `tokenizer.chat_template`，并在 `run` 模式中
注入 `PromptBuilder`。该键缺失且 `general.architecture == "qwen2"` 时返回
`qwen2` 内置 fallback 标记；其他缺失或读取失败场景保持空模板，由
`PromptBuilder` 使用纯文本回退格式。

导出的 `chat_template.txt` 主要用于检查和外部工具消费；当前显式
`--tokenizer` 路径仍从模型 GGUF 获取运行时 chat template。

## 测试覆盖

`tests/gguf_tokenizer_test.cc` 在单个测试文件内构造 tiny GGUF fixture，不增加
额外 fixture 头文件。覆盖：

- 从 tiny GGUF 加载词表、BPE merges、BOS/EOS/UNK 和 chat template。
- 编码与整体解码 round-trip。
- 特殊 token 前置匹配。
- 缺失 GGUF 文件。
- `GgufTokenizer` 与 `BpeTokenizer` 的 token ID 对齐。
- 导出脚本生成四个运行相关文件。
- 本地存在真实 Qwen2 GGUF 时验证真实 metadata 和两条 tokenizer 路径。

真实模型断言：

| 项目 | 预期值 |
| --- | ---: |
| vocab size | 151936 |
| BOS token ID | 151643 |
| EOS token ID | 151645 |
| `&lt;\|im_start\|&gt;` | 151644 |
| `&lt;\|im_end\|&gt;` | 151645 |

## 构建与验收命令

重新配置和构建：

```bash
cmake -S . -B build
cmake --build build -j2
```

运行本阶段测试：

```bash
ctest --test-dir build/ -R mini-llama-tests --output-on-failure
```

该阶段注册了 6 项 GGUF tokenizer 集成测试和 7 项 Chat Template 集成测试，
当时前缀筛选共 13 项通过。当前 `mini-llama-tests` 前缀还涵盖量化/线程与 GGUF、
调试等集成用例，不能把该前缀的现有全部测试数写成 13。

使用真实 Qwen2 模型生成一个 token：

```bash
./build/mini-llama generate \
  --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  --prompt "Hello, 你好!" \
  --n-predict 1
```

检查控制 token metadata：

```bash
./build/mini-llama inspect-gguf \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  | grep -E "bos|eos"
```

预期关键输出：

```text
metadata tokenizer.ggml.eos_token_id (uint32): 151645
metadata tokenizer.ggml.bos_token_id (uint32): 151643
metadata tokenizer.ggml.add_bos_token (bool): false
```

## 当前边界

- C++ GGUF metadata getter 当前按项目既有 Reader 能力处理 GGUF v3；导出脚本
  支持 v2/v3。
- 当前编码器面向 GPT-2 byte-level BPE metadata，不是通用 SentencePiece 或
  unigram tokenizer 实现。
- 预分词是项目内轻量规则，不包含完整正则表达式引擎；新增模型家族时需要验证
  其 pre-tokenizer 语义。
- 特殊 token 当前依据尖括号形式和常见控制 token 名称识别，没有完整消费
  `tokenizer.ggml.token_type` 的所有模型特定语义。
- 导出的 `special_tokens.json` 可以包含模型用于词表填充的控制项，文件大小取决
  于原始 GGUF metadata。
- 目录形式的 `--model` 只接受唯一 GGUF；多量化文件并存时需要显式指定文件。

## 一句话总结

本阶段完成了 GGUF tokenizer 的“可拆解检查”和“运行时直读”两条路径，并将其
接入模型装配、chat template、命令行生成和真实 Qwen2 回归测试；两条 tokenizer
路径对相同文本产生一致 token ID。
