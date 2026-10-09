# GGUF 模块实现总结

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

本分支在现有 CPU/F32 推理链路基础上补充了轻量 GGUF v3 读取能力，目标是为模型检查、元数据访问和后续权重加载提供稳定的底层接口。

## 实现内容

- `GgufReader::Load()` 逐步解析 magic、版本、头部计数、元数据和张量描述。
- 严格校验版本、条目数量、重复键名/张量名、维度、对齐值和文件范围；失败只写入 `load_error`，不进行终端输出。
- `InspectGguf()` 输出开发者快照，包括头信息、元数据摘要、张量形状、类型、偏移和字节数。
- 支持 F32、F16、Q4_0、Q4_1、Q8_0 的张量大小计算，以及原始张量字节读取。
- 提供整数、浮点、字符串、字符串数组和整数数组元数据读取接口，并检查类型和转换范围。
- 新增 `scripts/make_test_gguf.py`，生成数百字节级的最小 GGUF fixture，避免将 GB 级正式模型纳入 CI。

## 测试与构建

- `tests/gguf_test.cc` 位于 `tests` 根目录，覆盖正常加载、类型化元数据、原始张量、仓库 fixture 和损坏输入。
- `tests/CMakeLists.txt` 为 `tests/unit/gguf_test.cc` 注册 `mini-llama-gguf-tests`，
  根目录文件对应 `mini-llama-gguf-integration-tests`。
- 生成 fixture：

  ```bash
  python scripts/make_test_gguf.py --output models/tiny/test.gguf
  ```

- 该 Reader 阶段历史完整测试矩阵为 56 项，全部通过；不代表当前工作区重新验证。

## 边界

当前读取器只实现 GGUF v3 和上述 GGML 类型子集；未知张量布局不会被当作可读取的权重。GGUF tokenizer 仍由独立的
`GgufTokenizer` 负责词表和 BPE 元数据处理。
