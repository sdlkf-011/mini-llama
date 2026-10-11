# Tensor、基础算子与 Matmul Dispatch 阶段总结

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

## 实现范围

- `Tensor` 提供 F32 稠密张量，保存 `shape` 与连续 `data`，支持 1D 至 4D 便捷访问、通用多维索引、二维行指针、shape
  校验和安全重塑。
- `MatmulDispatch` 实现 `[m, k] × [k, n] -> [m, n]`。支持 `kNaive`、`kThreaded`、
  `kSimd`、`kThreadedSimd`；默认模式为 `kThreadedSimd`，通用 Matmul 的 SIMD
  模式回退到按输出行并行的标量实现。
- `LinearDispatch` 支持 `[in_features]` 与 `[1, in_features]` 输入，按 `x @ weight^T` 计算并保持输入秩。
- F32 Linear 默认按输出通道调用 `ParallelFor()`，每行使用 AVX2/FMA、NEON 或 scalar 点积；权重仍按
  `[out_features, in_features]` 连续存储。
- `Linear(x, QuantizedTensor)` 独立按 F32/Q8_0/Q4_0/Q4_1 格式分发。本阶段完善的
  `LinearQ80/Q40/Q41` 均支持 batch，并按输入行与输出通道并行；Q8_0/Q4_0 使用 NEON 或 scalar 即时解码点积，
  Q4_1 使用 scalar 点积。
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

`tests/tensor_test.cc` 有 5 项 Tensor 测试；`tests/ops_test.cc` 有 11 项
算子测试，分别注册独立 target，覆盖：

- Tensor 构造、填充、索引展开、At 访问器、RowPtr、非法 shape/索引、AssertShape 和 ReshapeChecked。
- 2×3 与 3×2 矩阵乘法，验证当前朴素 dispatch 模式和错误维度处理。
- 一维与二维 Linear、RMSNorm、稳定 Softmax、Silu、逐元素乘法、SwiGLU 和 ArgMax。

`tests/threaded_matmul_test.cc` 现有 14 项测试，覆盖多种执行模式和线程数、SIMD
尾部、小任务、非整除切分、回调异常，以及带 bias/GQA 的并行 Attention；量化
测试覆盖 batch、块尾部、非法输入、兼容接口、误差比较和 Q4_0 舍入边界。
实现与测试矩阵见[CPU 多线程与 F32 SIMD 实现说明](cpu_threaded_inference_implementation.md)。
新增量化功能与验证边界见
[SIMD 点积、量化 Linear 与 batch 支持阶段总结](simd_quantized_linear_implementation.md)。

## 验证方式

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build -R '^TensorTest\.' --output-on-failure
ctest --test-dir build -R 'mini-llama-tests\.ThreadedMatmulTest\.' --output-on-failure
python3 scripts/test_threaded_cli.py
```

基础算子测试保留可手算的小矩阵；多线程测试另用 67 个输出行跨过并行阈值，
验证真实多线程路径。CTest 输出每个独立测试的通过/失败状态及汇总结果。
