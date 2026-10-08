# Tokenizer 实现汇总

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

`feature/implement-tokenizer` 分支基于统一的 `ITokenizer` 接口，实现了以下分词器：

- `AsciiTokenizer`：支持 ASCII 字符级编码与解码，非 ASCII 字节回退到 UNK。
- `JsonVocabTokenizer`：加载数组对象格式的 JSON 词表，编码时自动添加 BOS，采用最长优先匹配，并在无匹配时按输入字节回退到 UNK。
- `BpeTokenizer`：加载 `vocab.json`、`merges.txt` 和特殊 token 文件，实现 GPT-2 风格字节映射与 BPE 合并。
- `GgufTokenizer`：从 GGUF 元数据读取词表、合并规则和控制 token ID，实现字节级 BPE 编解码。

JSON 解析覆盖字符串转义、Unicode 代理对、整数、布尔值与未知字段跳过；文件读取失败、字段缺失或语法错误会抛出异常。通用工厂会捕获 JSON 加载异常并回退到 ASCII 分词器，BPE 与 GGUF 工厂则按各自约定返回空指针。

构建系统已将全部 tokenizer 源文件加入 `mini_llama_core`，并提供独立的 `mini-llama-tokenizer-tests` 测试目标。测试覆盖正常编解码、特殊 token、未知 token、非法输入及工厂函数；阶段记录的 tokenizer 测试 8 项、项目测试 61 项通过为历史快照。当前
`tests/unit/tokenizer_test.cc` 有 10 项用例，GGUF tokenizer 与模板另有单元和
集成 target；运行状态需以当前构建验证为准。
