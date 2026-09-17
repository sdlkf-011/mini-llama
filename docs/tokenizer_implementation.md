# Tokenizer 实现汇总

`feature/implement-tokenizer` 分支基于统一的 `ITokenizer` 接口，实现了以下分词器：

- `AsciiTokenizer`：支持 ASCII 字符级编码与解码，非 ASCII 字节回退到 UNK。
- `JsonVocabTokenizer`：加载 JSON 词表，采用最长优先匹配，并支持 UTF-8 字符级 UNK 回退。
- `BpeTokenizer`：加载 `vocab.json`、`merges.txt` 和特殊 token 文件，实现 GPT-2 风格字节映射与 BPE 合并。
- `GgufTokenizer`：从 GGUF 元数据读取词表、合并规则和控制 token ID，实现字节级 BPE 编解码。

分支同时提供 ASCII、JSON、BPE 和 GGUF 分词器工厂函数，并在加载失败时执行约定的回退或返回空指针。

构建系统已将全部 tokenizer 源文件加入 `mini_llama_core`，并新增独立的 `tokenizer-tests` 测试目标。测试覆盖正常编解码、特殊 token、未知 token、非法输入及工厂函数，目前 tokenizer 测试 8 项全部通过，原有测试 13 项全部通过。
