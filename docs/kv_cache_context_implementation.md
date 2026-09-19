# KV Cache、推理上下文与前向推理阶段实现汇总

本阶段（`feature/kv_cache-and-context`）在 Tensor、基础算子和 Tokenizer 的基础上，补充了模型结构、批处理、KV Cache、压缩前缀树、请求追踪以及单 token/批量前向推理能力。实现面向教学用途，当前使用 CPU 上的 F32 张量，同时为后续量化和 GPU 后端保留了包装层接口。

## 实现范围

### 模型结构与推理上下文

- `ModelConfig` 描述词表、隐藏维度、层数、注意力头、KV 头、头维度、最大序列长度、RoPE 和 RMSNorm 参数。
- `LayerWeights` 保存每个 Transformer 层的 RMSNorm、Q/K/V/O 投影、可选偏置以及 SwiGLU FFN 权重。
- `MiniLlamaModel` 统一保存配置、嵌入表、层权重、最终归一化和 LM Head，并通过 `loaded` 与 `load_error` 表示加载状态。
- `MiniLlamaContext` 持有模型指针、KV Cache、当前位置 `pos`、token 历史和 prefill/decode 统计；复制被禁止，移动可用。
- 使用模型配置构造上下文时，会自动创建匹配尺寸的 KV Cache；空模型上下文保持默认空状态。

### KV Cache

`KvCache` 使用两个连续的四维 F32 张量保存键和值：

```text
keys   : [n_layers, max_seq_len, n_kv_heads, head_dim]
values : [n_layers, max_seq_len, n_kv_heads, head_dim]
```

`Write(layer, pos, k, v)` 将形状为 `[n_kv_heads, head_dim]` 的当前 token 键和值复制到指定层和位置。`KeyPtr` 与 `valuePtr` 返回指定头的连续内存指针，供注意力计算遍历历史位置。写入前会检查缓存秩、层/位置范围以及输入张量形状，访问接口也会拒绝越界参数。

### MiniBatch 与前向入口

- `MiniBatch::Single(token, pos)` 构造单 token 解码批次。
- `MiniBatch::FromTokens(tokens, start_pos)` 构造连续位置的多 token prefill 批次。
- `ForwardToken` 处理一个 token，当前位置由 `ctx.pos` 指定；函数更新各层 KV Cache，但不自动推进 `ctx.pos`。
- `ForwardBatch` 按批次顺序调用 `ForwardToken`，在每一步设置位置并追加 token 历史，返回最后一个 token 的 logits，因此统一了 prefill 和 decode 路径。
- 空批次以及 tokens/positions 长度不一致会立即报错。

### Transformer 前向计算

单 token 前向流程如下：

1. 校验模型加载状态、配置约束、token/位置范围、缓存形状和所有层权重形状。
2. 从 `[vocab_size, dim]` 嵌入表复制 token 对应的隐藏向量。
3. 每层执行 Pre-Norm 注意力：RMSNorm → Q/K/V Linear（支持可选 bias）→ RoPE → 因果自注意力 → O 投影 → 残差。
4. 注意力使用 GQA 映射：多个 query head 按比例共享 KV head；分数乘以 `1/sqrt(head_dim)`，只读取 `0..pos` 的缓存位置以保证因果性。
5. 每层继续执行 Pre-Norm SwiGLU FFN：RMSNorm → `w_gate/w_up` → `Silu(gate) * up` → `w_down` → 残差。
6. 最终执行 RMSNorm 和 LM Head，输出形状为 `[vocab_size]` 的 logits。

前向实现通过 `ForwardLinear`、`ForwardRmsNorm`、`ForwardSwiGlu`、`ForwardSoftmax` 和 `ForwardRope` 进行算子封装。当前封装调用 CPU F32 实现，后续可在不改动 Transformer 流程的情况下替换为量化或设备实现。

### RoPE 与算子扩展

`Rope` 支持两种布局：

- `RopeType::kNormal`：相邻维度成对旋转。
- `RopeType::kNeoX`：前后半区对应维度成对旋转。

Q 和 K 应用位置旋转，V 不参与旋转。实现检查输入为二维张量、位置非负、theta 有限且为正，并要求 head dimension 为偶数。

### RadixTree

`RadixTree` 用压缩边保存 token 序列前缀，适合缓存请求之间的公共 prompt 前缀：

- `Insert` 支持普通序列、共享前缀和空序列；重复插入不会重复计数。
- 插入过程中会在部分匹配时拆分压缩边。
- `LongestPrefix` 返回从输入起始位置开始的最长可匹配 token 数量。
- `size` 表示已插入的不同终止序列数，`Clear` 清理整棵树；类型支持移动、不支持复制。

### 请求级追踪

`RequestContext` 为一次请求保存 trace ID、模式、后端、模型路径、总耗时、各阶段耗时、token 计数、事件列表和错误信息。

- `Start`/`Finish` 使用 `steady_clock` 计算耗时，避免系统时间校准影响统计。
- `NewTraceId` 使用本地时间加原子递增序号生成进程内唯一 ID。
- `RecordEvent` 保存阶段名、耗时、token 数和可选诊断信息。
- `SetError` 与 `Ok` 统一管理请求成功/失败状态。
- `FormatRequestTraceSummary` 输出单行请求摘要，耗时统一保留三位小数；失败摘要附带错误文本。
- `FormatRequestTraceEvents` 按原始顺序输出事件行，只有 detail 非空时才输出 detail 字段。

## 校验与错误处理

前向入口集中校验以下条件：模型必须已加载；配置参数必须为正且满足 `dim == n_heads * head_dim`、`n_heads` 可被 `n_kv_heads` 整除；RoPE/RMSNorm 参数必须有效；token、位置和缓存索引必须在范围内；嵌入、LM Head、层权重和偏置形状必须匹配。错误通过 `std::runtime_error` 或 `std::out_of_range` 报告，并在信息中包含相关阶段或张量名称。

## 单元测试与验证

本阶段新增以下测试文件，每个测试前均注释了测试目的：

- `tests/unit/kv_cache_test.cc`：写入/读取寻址，以及形状和越界校验。
- `tests/unit/radix_tree_test.cc`：共享前缀、最长匹配、重复插入、清空和移动语义。
- `tests/unit/chat_test.cc`：当前仓库尚无独立 Chat API，因此覆盖聊天推理所依赖的 `MiniLlamaContext` 生命周期、KV Cache 初始化和 token 历史。
- `tests/unit/request_context_test.cc`：请求启动、事件记录、耗时、摘要格式化、错误状态和事件顺序。
- `tests/unit/forward_test.cc`：单 token 前向结果、KV Cache 写入、批量与逐 token 等价性、上下文状态维护和非法输入校验。

构建和测试命令：

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

当前测试矩阵共 33 项，包含原有 Tensor/Tokenizer 测试和本阶段新增测试，全部通过。

## 当前边界

本阶段未实现模型文件加载、完整对话编排、采样器或独立 Chat 服务接口；`chat_test.cc` 因而只验证现有推理上下文基础。前向计算目前为 CPU F32 实现，批处理仍按 token 顺序执行，后续可在既有包装层和上下文结构上继续扩展。
