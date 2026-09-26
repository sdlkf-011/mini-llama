# `feature/model-formats-and-loader` 分支汇总

## 分支定位

本分支在已有 Tensor、Tokenizer、KV Cache、Transformer 前向、Sampler、Chat/Run 的基础上，补充模型文件格式和权重加载能力：

```text
model.json + model.bin → Loader → MiniLlamaModel
                                      ↓
                         Tokenizer → Forward → Sampler → 文本输出
```

## 主要改动

### 模型格式

- `model.json`：模型配置、tokenizer 描述、张量名称/shape/dtype/offset/byte_size。
- `model.bin`：按清单偏移保存的连续 F32 权重。
- `models/tiny/`：提供可测试的 tiny 模型、词表及 GGUF 样例文件。

### Loader 接口

新增 `include/mini_llama/loader.h` 和 `src/loader.cc`：

| 接口 | 作用 |
| --- | --- |
| `ParseManifest()` | 解析并校验 `model.json`，返回 `ModelManifest` |
| `LoadModel()` | 按名称和偏移读取权重，生成 `MiniLlamaModel` |
| `InspectModel()` | 只检查并打印元数据，不读取权重 |

新增结构：`TensorInfo` 描述张量布局，`TokenizerInfo` 描述 tokenizer，`ModelManifest` 保存完整清单。

### 模型结构补强

`model.h` 明确了 `ModelConfig`、`LayerWeights` 和 `MiniLlamaModel` 的语义：使用 CPU F32，线性权重布局为 `[out, in]`，`loaded` 表示是否可前向，`load_error` 保存失败原因，`head_dim` 由 `dim / n_heads` 推导。

### 构建与测试

- `src/loader.cc` 加入核心库。
- 新增 `mini-llama-loader-tests` 测试目标。
- 测试使用 `models/tiny` 作为固定样例目录。

## Loader 流程

```text
ParseManifest(model.json)
    ↓ 解析 config / tokenizer / tensors
    ↓ 校验维度、shape、dtype、字节数、offset 和张量完整性
LoadModel(model.json, model.bin)
    ↓ 按名称读取 embedding、各层权重、final_norm、lm_head
    ↓ 检查二进制文件长度
loaded = true
```

主要校验：

- 模型维度有效，`dim` 可被 `n_heads` 整除，`n_heads` 可被 `n_kv_heads` 整除。
- `head_dim` 为正偶数，RoPE/RMSNorm 参数有效。
- tokenizer special ID 非负且互不相同。
- dtype 只允许 `float32`，shape 推导字节数必须等于 `byte_size`。
- 张量名称不能重复，二进制区间不能重叠，必需张量必须存在且 shape 匹配。

错误约定：`ParseManifest()` 失败时抛出 `std::runtime_error`；`LoadModel()` 失败时返回 `loaded == false` 且填写 `load_error` 的模型。

## 与已有模块的关系

```text
Loader → MiniLlamaModel
Tokenizer → token IDs
PromptBuilder → prompt 文本
Context/KvCache → 运行时状态
Forward → logits
Sampler → 下一个 token
ChatSession/Terminal → 多轮状态和输出
```

Loader 只负责生成静态模型对象，不负责自动创建 `ITokenizer`、执行前向、采样、驱动聊天或提供进程级 `main()`。当前应用仍需：

```text
LoadModel() → 根据 TokenizerInfo 创建 tokenizer → 调用 Generate() / Run()
```

## 测试覆盖

新增 loader 测试覆盖：正常解析和加载 tiny 模型、重复 JSON 键、非法 Attention 维度、默认 tokenizer ID、权重文件尾随字节。

完整验证命令：

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## 当前边界

- 只支持自定义 `model.json + model.bin` 和 F32 权重。
- 暂不支持量化、GGUF 模型权重或 safetensors。
- `model.json` 使用固定格式的轻量解析器，不是通用 JSON 解析器。
- 没有统一 CLI `main()`，Loader 与 `Run()` 尚未自动串接。
- tokenizer 描述已解析，但需要上层手动实例化 tokenizer。
- `MiniBatch` 仍按 token 顺序执行；`RequestContext` 尚未接入 Generate/Run。

## 一句话总结

本分支把已有推理链路接上了磁盘模型输入：Loader 严格解析模型清单并读取 F32 权重，生成可直接供 Forward 使用的 `MiniLlamaModel`；下一步是补齐 tokenizer 自动创建和最终 CLI 装配。
