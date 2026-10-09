# GGUF Loader 实现汇总

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。
>
> 本文记录对应分支完成时的阶段状态，文中的未接入功能不代表当前仓库现状。
> 当前 CLI 装配、特殊 token 校验、终端帮助与逐轮追踪见[命令行与模块同步说明](cli_integration_status.md)。

## 分支定位

本分支建立在现有 GGUF Reader、`MiniLlamaModel`、Tokenizer 和 CPU F32
前向链路之上，补充从 GGUF 文件到可执行模型对象的映射与权重加载能力：

```text
GGUF 文件
  ├─ metadata ──→ ModelConfig
  ├─ tensor info ──→ 名称与 shape 映射
  └─ tensor data ──→ F32/F16/Q8_0/Q4_0/Q4_1 解码
                              ↓
                       MiniLlamaModel
                              ↓
                  Context → Forward → Sampler
```

目标是复用现有张量和前向实现完成真实 GGUF 模型加载，使用 `QuantizedTensor`
保存 Linear 并保留量化块；不把具体模型的层数、维度或词表大小写死。

## 主要改动

### 加载接口

新增 `include/mini_llama/gguf_loader.h`：

```cpp
MiniLlamaModel LoadGgufModel(const std::string& gguf_path);
```

接口约定：

- 成功时返回完整模型，`loaded == true` 且 `load_error` 为空。
- 失败时不向调用方传播内部异常，返回 `loaded == false` 并填写
  `load_error`。
- Embedding、norm、bias 使用 F32 `Tensor`；Linear 使用 F32 或保留量化块的 `QuantizedTensor`。

新增 `src/gguf_loader.cc`，并将其加入 `mini_llama_core` 构建目标。

### 配置构建

加载器读取 `general.architecture`，使用架构名动态组成元数据前缀。例如
Qwen2 使用 `qwen2.*`，Llama 使用 `llama.*`，不固定到某一份模型参数。

| `ModelConfig` 字段 | GGUF 元数据来源 |
| --- | --- |
| `dim` | `<architecture>.embedding_length` |
| `hidden_dim` | `<architecture>.feed_forward_length` |
| `n_layers` | `<architecture>.block_count` |
| `n_heads` | `<architecture>.attention.head_count` |
| `n_kv_heads` | `<architecture>.attention.head_count_kv`，缺省时等于 `n_heads` |
| `max_seq_len` | `<architecture>.context_length` |
| `rope_theta` | `<architecture>.rope.freq_base` |
| `rms_norm_eps` | `<architecture>.attention.layer_norm_rms_epsilon` |
| `head_dim` | 由 `dim / n_heads` 推导 |
| `vocab_size` | 优先读取 `<architecture>.vocab_size`，否则由嵌入矩阵推导 |

Qwen2 的 RoPE 采用前后半维配对，因此映射为 `RopeType::kNeoX`；其他架构
保留现有默认模式。

配置阶段会校验：

- 必需整数为正且可由 `int` 表示。
- 浮点参数有限且为正。
- `dim` 可被 `n_heads` 整除。
- `n_heads` 可被 `n_kv_heads` 整除。
- `head_dim` 为偶数，满足当前 RoPE 实现要求。
- 词表大小、嵌入维度与 `token_embd.weight` 一致。

### 张量名称映射

加载器将 llama.cpp 风格的 GGUF 名称转换为项目内部字段名：

| GGUF 名称 | 项目名称 |
| --- | --- |
| `token_embd.weight` | `token_embedding` |
| `output_norm.weight` | `final_norm` |
| `output.weight` | `lm_head` |
| `blk.N.attn_norm.weight` | `layers.N.attention_norm` |
| `blk.N.attn_q/k/v.weight` | `layers.N.wq/wk/wv` |
| `blk.N.attn_q/k/v.bias` | `layers.N.bq/bk/bv` |
| `blk.N.attn_output.weight` | `layers.N.wo` |
| `blk.N.ffn_norm.weight` | `layers.N.ffn_norm` |
| `blk.N.ffn_gate/up/down.weight` | `layers.N.w_gate/w_up/w_down` |

层号必须是十进制数字。当前模型结构不认识的扩展张量不会被误装配，而是留给
后续模块处理；必需张量缺失或两个 GGUF 名称映射到同一内部名称时加载失败。

### Shape 映射

GGUF 使用 `ne0` 作为最快变化维，项目 `Tensor` 使用行主序 shape。因此加载器
反转 GGUF 维度：

```text
GGUF [in_features, out_features]
                 ↓
Tensor [out_features, in_features]
```

原始元素顺序无需转置，只需改变逻辑 shape。转换过程中会拒绝零维度、超过
`int` 范围的维度和元素数量溢出。

### 权重解码

当前支持以下 GGML 类型。稠密张量展开为 F32；Linear 的 F32/F16 转为 F32，三种量化格式保留块数据：

| 类型 | 块布局 | 解码方式 |
| --- | --- | --- |
| `F32` | 每元素 4 字节 | 按 GGUF 小端位模式恢复 |
| `F16` | 每元素 2 字节 | IEEE 754 binary16 转 F32 |
| `Q8_0` | 32 元素/34 字节 | `F16 scale + 32 × int8` |
| `Q4_0` | 32 元素/18 字节 | `F16 scale + 16 × packed uint4`，零点为 8 |
| `Q4_1` | 32 元素/20 字节 | `F16 scale + F16 min + 16 × packed uint4` |

解码前会使用 `GgufTensorDataSize()` 计算理论字节数，并要求实际读取结果精确
匹配。当前 GGUF 量化 Linear 要求每行输入维度是 32 的整数倍。F16 转换覆盖
零、次正规数、普通数以及 Inf/NaN，整数读取显式处理小端序和未对齐访问。

`DecodeDenseTensor()` 为 embedding/norm/bias 反量化；`DecodeLinearWeight()`
为投影权重保留 Q8_0/Q4_0/Q4_1 块。`Linear(x, QuantizedTensor)` 按类型分发并
即时解码点积，不创建完整临时 F32 权重。Q8_0/Q4_0 可用 NEON，三种量化 Linear
均支持 batch 和外层线程调度；更多 SIMD GEMM 优化仍未实现。

## 模型装配流程

```text
GgufReader::Load(path)
    ↓ 校验 GGUF 目录、数据范围和类型大小
BuildConfig(path, reader)
    ↓ 读取架构元数据并推导模型配置
MapTensorName(...)
    ↓ 建立内部名称到 GgufTensorInfo 的索引
DecodeTensor(...)
    ↓ 加载 embedding、每层 Attention/FFN、final norm、lm head
ValidateLoadedShapes(...)
    ↓ 全部成功
model.loaded = true
```

各层按照 `MiniLlamaModel` 的结构顺序装配：

1. `token_embedding`。
2. 每层 Attention norm、Q/K/V 权重与可选偏置、输出投影。
3. 每层 FFN norm、gate/up/down 投影。
4. `final_norm`。
5. `lm_head`。

GGUF 允许省略 `output.weight` 表示输出头与词元嵌入共享权重。遇到这种文件时，
加载器使用已经解码的 `token_embedding` 构造 `lm_head`，没有针对某个模型文件
写死特殊路径。

## 完整性校验

模型发布为 `loaded == true` 前会验证：

- `token_embedding` 为 `[vocab_size, dim]`。
- `final_norm` 为 `[dim]`。
- `lm_head` 为 `[vocab_size, dim]`。
- 实际层数等于 `n_layers`。
- Q 权重为 `[n_heads * head_dim, dim]`。
- K/V 权重为 `[n_kv_heads * head_dim, dim]`。
- Q/K/V 可选偏置与对应投影输出维度一致。
- Attention 输出和 FFN 三个投影符合当前 `[out, in]` 前向约定。

任何解析、读取、分配、类型或 shape 异常都会统一转换为失败模型，避免调用方
拿到部分初始化的权重。

## 与现有模块的关系

```text
GgufReader      → 提供元数据、张量描述和原始字节读取
GgufLoader      → 构建 ModelConfig 并装配 MiniLlamaModel
GgufTokenizer   → 从同一 GGUF 加载词表、BPE 合并和特殊 token
Context/KvCache → 保存生成期间的运行状态
Forward         → 消费 F32/量化权重并生成 logits
Sampler         → 从 logits 选择下一个 token
```

`GgufLoader` 只负责静态模型对象，不负责命令行参数、tokenizer 生命周期或聊天
模板。上层真实生成路径为：

```text
LoadGgufModel()
    + CreateGgufTokenizer()
    → Encode(prompt)
    → ForwardBatch()
    → Sample()
    → DecodeToken()
```

## 构建与验证

基础构建和完整测试：

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

该分支历史结果：56/56 项测试通过；`gguf_loader.cc` 使用 `-Wall -Wextra -Wpedantic`
独立编译无警告。

真实模型使用：

```text
models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
size:   531065536 bytes
sha256: 834f4115ad5a836c9f17716b1577290fda96de3deb881ba45a4d5476fd202e96
```

检查命令：

```bash
./build/mini-llama inspect-gguf \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

验证结果：成功解析 GGUF v3、26 项元数据和 290 个张量，并输出完整张量层级
列表；张量包含 F32 norm/bias 和 Q8_0 embedding/linear 权重。

生成命令：

```bash
./build/mini-llama generate \
  --model models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf \
  -p hello -n 1
```

验证结果：成功加载 Qwen2 配置（24 层、dim 896、14 个查询头、2 个 KV 头、
head_dim 64、词表 151936），创建 GGUF tokenizer，执行真实 24 层 CPU 前向和
采样，生成一个 token，输出文本为 `,`。

上述结果保留阶段验证含义。当前 CMake 已提供正式 `mini-llama` target，
`RunMain()` 已接入 `LoadGgufModel()`、内嵌 tokenizer 和聊天模板；当前工作区的
执行路径与最新回归结果见 [CLI 同步说明](cli_integration_status.md)。

## 当前边界

- 只支持 GGUF Reader 已覆盖的 GGUF v3 和 F32、F16、Q8_0、Q4_0、Q4_1。
- 不支持 Q4_K、Q5_K、Q6_K、IQ 等扩展量化格式。
- Linear 权重保留 Q8_0/Q4_0/Q4_1 块，并在 CPU 前向中逐块即时计算，
  不再分配完整的临时 F32 权重矩阵。
- `token_embedding` 目前仍转换为 F32；省略 `output.weight` 时还会复制该
  embedding 作为 `lm_head`，这部分内存仍需后续改为按行解码与共享存储。
- 当前张量映射面向已有 Llama/Qwen2 风格 `LayerWeights`，未知扩展张量会忽略。
- Qwen2 的 NeoX RoPE 语义已接入；更多架构需要补充对应的 RoPE/模型语义。
- 模型文件必须是真实 GGUF；Git LFS pointer 文本不能用于解析或生成。
- 量化 Linear 已支持 batch 与外层并行，Q8_0/Q4_0 支持 NEON；Q4_1 点积和无 NEON 时使用 scalar。
- Loader 不承担 tokenizer 或 Sampler 职责；当前 CLI 装配已由 `src/main.cc` 完成。

## 一句话总结

本分支完成了从 GGUF Reader 到 `MiniLlamaModel` 的关键桥接：动态读取模型配置，
映射 Llama/Qwen2 风格权重，保留线性层量化块并接入 CPU 前向，已使用真实
Qwen2-0.5B Q8_0 模型验证首 token logits 输出。
