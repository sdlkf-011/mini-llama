# Tokenizer 实现汇总

`feature/implement-tokenizer` 分支基于统一的 `ITokenizer` 接口，实现了以下分词器：

- `AsciiTokenizer`：支持 ASCII 字符级编码与解码，非 ASCII 字节回退到 UNK。
- `JsonVocabTokenizer`：加载数组对象格式的 JSON 词表，编码时自动添加 BOS，采用最长优先匹配，并在无匹配时按输入字节回退到 UNK。
- `BpeTokenizer`：加载 `vocab.json`、`merges.txt` 和特殊 token 文件，实现 GPT-2 风格字节映射与 BPE 合并。
- `GgufTokenizer`：从 GGUF 元数据读取词表、合并规则和控制 token ID，实现字节级 BPE 编解码。

JSON 解析覆盖字符串转义、Unicode 代理对、整数、布尔值与未知字段跳过；文件读取失败、字段缺失或语法错误会抛出异常。通用工厂会捕获 JSON 加载异常并回退到 ASCII 分词器，BPE 与 GGUF 工厂则按各自约定返回空指针。

构建系统已将全部 tokenizer 源文件加入 `mini_llama_core`，并提供独立的 `mini-llama-tokenizer-tests` 测试目标。测试覆盖正常编解码、特殊 token、未知 token、非法输入及工厂函数；当前 tokenizer 测试 8 项、项目测试 61 项全部通过。
