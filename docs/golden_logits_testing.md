# Python ↔ C++ golden logits 回归

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

依赖：CMake、当前 C++ 工具链、Python 3 和 NumPy。参考实现直接读取
`models/tiny/model.json` 中的 float32 张量形状、字节偏移和
`models/tiny/model.bin` 权重，不调用 C++ 算子。

```bash
cmake --build build -j4
ctest --test-dir build -R golden-logits --output-on-failure
```

CTest 使用 `MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>` 指定本次构建产物。
也可以直接运行：

```bash
python3 scripts/test_golden.py
MINI_LLAMA_BIN=/path/to/mini-llama python3 scripts/test_golden.py
```

自动测试按 C++ `--dump-logits` → `make_golden.py` → `compare_logs.py`
执行，在临时目录导出并比较以下文件，结束时清理临时目录：

- C++：`logits_stepN.bin`、`generation_tokens.txt`。
- Python：`python_logits_stepN.bin`、`python_generation_tokens.txt`。

二进制内容是连续的本机 float32 logits，token 文件是空格分隔的整数序列。
step 从 0 开始；prompt 包含 BOS=1，随后每个 ASCII 字符直接映射到 0..127。
原有 tiny 词表是稀疏的，因此测试通过现有 `--tokenizer` 参数提供完整 ASCII
词表，不修改项目分词器和模型资源。非 ASCII prompt 在 Python 中明确拒绝。

参考模型的 KV cache 按 `[layer, pos, kv_head, head_dim]` 持续复用。
`forward_token(model, pos, token_id)` 要求位置从 0 连续递增，每层先写入当前位置，
再对 `[0, pos]` 做 causal attention。Q 头映射为
`q_head // (n_heads // n_kv_heads)`，scale 为 `1 / sqrt(head_dim)`。
RMSNorm 使用清单中的 eps；RoPE 使用绝对 pos 和相邻维度配对；线性权重为
`[out, in]`，FFN 顺序为 `SiLU(gate) * up` 再 down projection。

P 个 prompt token、实际生成 N>0 个 token 时导出 `P + N - 1` 份 logits。
第一个生成 token 直接取最后一份 prefill logits 的 argmax，最后一个生成 token
无需再 forward。EOS=2 保留在生成序列中并立即停止。N=0 时仅导出 P 份
prefill logits。greedy 并列时选择最小 token ID。

比较要求两侧 step 集合相同且从 0 连续编号，logits 形状一致、全部有限，
每步 `max_abs_error <= 1e-4`、`max_rel_error <= 1e-4`，relative error
仅统计 `abs(python_logits) >= 1e-3` 的元素；每步 top-1 和最终生成序列
也必须一致。失败返回非零退出码，全部通过时最后输出 `RESULT: ALL PASSED`。

覆盖原始 tiny 模型、从现有权重截取 K/V 行得到的 4→2 GQA 模型（同时采用
非默认 eps/theta）、空 prompt 单 token 生成、prefill-only、greedy tie 和
提前 EOS。还验证 dump 与正常 batch prefill 的生成结果一致，以及比较器
确实拒绝 step 缺失、绝对/相对超差、top-1 不同、非有限数值、截断文件和
生成序列不同。

单独生成或比较（请使用新的输出目录，避免旧 step 文件混入）：

```bash
build/mini-llama generate --model models/tiny --prompt hello --n-predict 4 \
  --temperature 0 --top-k 0 --dump-logits /tmp/mini-llama-cpp-golden
python3 scripts/make_golden.py --model models/tiny --prompt hello \
  --n-predict 4 \
  --output-dir /tmp/mini-llama-python-golden
python3 scripts/compare_logs.py --cpp-dir /tmp/mini-llama-cpp-golden \
  --python-dir /tmp/mini-llama-python-golden
```

上述 `hello` 字符均已包含在现有稀疏 tiny 词表中；任意 ASCII prompt 的
完整映射由 `test_golden.py` 的临时词表提供。
