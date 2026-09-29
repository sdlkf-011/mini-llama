# Tensor、基础算子与 Matmul Dispatch 阶段总结

## 实现范围

- `Tensor` 提供 F32 稠密张量，保存 `shape` 与连续 `data`，支持 1D 至 4D 便捷访问、通用多维索引、二维行指针、shape 校验和安全重塑。
- `MatmulDispatch` 实现 `[m, k] × [k, n] -> [m, n]`。当前仅暴露并默认使用已完成的朴素模式；分发入口保持独立，后续后端实现完成后可显式接入。
- `LinearDispatch` 支持 `[in_features]` 与 `[1, in_features]` 输入，按 `x @ weight^T` 计算并保持输入秩。
- `RmsNorm`、`Softmax`、`Silu`、`ElementwiseMul`、`SwiGlu` 和 `ArgMax` 提供推理所需的基础逐元素及归约算子。

## 正确性审查

本阶段在单元测试中加入小规模、可手算的矩阵和向量，覆盖正常结果、输出 shape、边界检查、维度错误和数值稳定性。审查发现并修正了以下问题：

- `Linear` 原先读取 `x.data[j]`，应按输入特征维读取 `x.data[k]`。
- `Linear` 的二维结果原先错误地使用输入特征数作为第二维，应返回 `[1, out_features]`。
- `SwiGlu` 原先固定使用 `gate.data[0]`，且重复乘了一次 gate；现按 `Silu(gate[i]) * up[i]` 逐元素计算。
- `Tensor::AssertShape` 原先忽略调用者名称，现使用传入的 `caller` 生成错误信息。
- 张量元素计数使用 `size_t` 保存，避免从 `size_t` 缩窄到 `int`。
- 空 shape 张量按标量处理，`FlatIndex({})` 不再因无符号下溢访问越界。

## 测试内容

`tests/unit/tensor_test.cc` 包含 13 个独立测试组，每个测试函数前均有注释说明测试目标，覆盖：

- Tensor 构造、填充、索引展开、At 访问器、RowPtr、非法 shape/索引、AssertShape 和 ReshapeChecked。
- 2×3 与 3×2 矩阵乘法，验证当前朴素 dispatch 模式和错误维度处理。
- 一维与二维 Linear、RMSNorm、稳定 Softmax、Silu、逐元素乘法、SwiGLU 和 ArgMax。

## 验证方式

```bash
cmake -S . -B build
cmake --build build -j2
./build/mini-llama-tests
```

测试矩阵保持在 3×3 以内，便于定位数值和索引错误；测试程序输出每个测试组的通过/失败状态及汇总结果。
