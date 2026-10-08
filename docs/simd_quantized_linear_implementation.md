# SIMD 点积、量化 Linear 与 batch 支持阶段总结

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

本阶段在已有 CPU 算子上完善 F32 SIMD 点积，补充量化 NEON 点积和外层并行，
统一量化 Linear 的 batch 输入支持，并补齐上传版本中的公开兼容接口。
正确的已有实现继续沿用，未引入完整 SIMD GEMM 或改变 Q4_0 舍入规则。

## 实现文件与范围

| 文件 | 本阶段内容 |
| --- | --- |
| [src/matmul_optimized.cc](../src/matmul_optimized.cc) | 拆分平台点积助手，补充 NEON FMA 分支，保留 Linear 和通用 Matmul 分发 |
| [src/quant.cc](../src/quant.cc) | Q8_0/Q4_0 NEON 点积、量化 Linear 并行与 batch、兼容 Matmul 和误差比较接口 |
| [include/mini_llama/quant.h](../include/mini_llama/quant.h) | 补齐公开声明，说明三种量化 Linear 的输入输出形状 |
| [tests/threaded_matmul_test.cc](../tests/threaded_matmul_test.cc) | 扩展尾部与 batch 测试，补充非法输入、兼容接口及舍入边界测试，说明测试目的 |

原有 AVX2 非对齐加载、horizontal reduction、scalar tail，以及 F32 Linear 的
输出通道并行结构已经正确，本阶段保留其计算逻辑。Q8_0/Q4_0 量化和三种格式的
反量化也沿用原有实现。

## F32 SIMD 点积与执行模式

`DotSimd()` 保持编译期选择，平台实现作为本文件内部的 `static` 函数：

| 条件 | 点积实现 | 每轮元素数 |
| --- | --- | --- |
| 定义 `__AVX2__` 和 `__FMA__` | `DotSimdAvx2()` | 8 |
| 定义 `__ARM_NEON` 或 `__ARM_NEON__` | `DotSimdNeon()` | 4 |
| 未启用以上 SIMD | `DotSimd()` 中的 scalar fallback | 1 |

AVX2 使用 `_mm256_loadu_ps` 和 `_mm256_fmadd_ps`，不要求 Tensor 额外满足
32-byte 对齐。NEON 使用 `vld1q_f32`；AArch64 或支持 FMA 的 ARM 使用
`vfmaq_f32`，其他 NEON 平台保留 `vmlaq_f32` 兼容路径。两种实现均先水平归约，
再以标量处理不足一个向量宽度的尾部。

这里没有运行时 CPU 指令集检测，是否启用 SIMD 由编译目标决定。

| 模式 | F32 Linear | 通用 Matmul |
| --- | --- | --- |
| `kNaive` | 单线程 scalar 点积 | 单线程 scalar 矩阵乘法 |
| `kThreaded` | 输出通道并行，scalar 点积 | 输出行并行 |
| `kSimd` | 单线程调用 `DotSimd()` | 复用 `MatmulThreaded()` |
| `kThreadedSimd` | 输出通道并行，线程内调用 `DotSimd()` | 复用 `MatmulThreaded()` |

每个 F32 Linear 输出通道使用输入数据和一行连续权重执行一次点积。
未启用 SIMD 时，`DotSimd()` 回退为 scalar，外层并行仍然有效。
通用 Matmul 的 B 列不连续，本阶段保持原有回退，未增加转置、packing、tiling、
cache blocking 或 register blocking。

## 量化块布局与 NEON 点积

量化块沿权重最后一维排列，每行独立补齐到 32 元素块。缩放因子 `d` 保存为
FP16 位模式；量化时使用实际存储的 FP16 scale，以便与解码尺度一致。

| 格式 | 块内表示及解码 | 点积路径 |
| --- | --- | --- |
| Q8_0 | 有符号 int8，权重为 `d * qs[i]` | NEON 或 scalar |
| Q4_0 | 16 字节保存 32 个值，权重为 `d * (nibble - 8)` | NEON 或 scalar |
| Q4_1 | 与 Q4_0 相同的半块布局，权重为 `d * nibble + m` | scalar |

Q8_0 的 NEON 助手每轮加载 8 个 int8，依次扩展为 int16、int32，再转换为
两组 F32，乘以块级 scale，与对应 F32 输入做乘加。

Q4_0 中，`qs[j]` 的低 nibble 对应第 `j` 个值，高 nibble 对应第 `j+16` 个值。
NEON 助手每轮拆出 8 个值，减 8 恢复有符号量化值，扩展、转 F32、缩放后乘加。
8 元素步长不会跨越 16 元素半块边界。

两种助手均水平归约后处理 scalar tail。最后一块只传入有效元素数，填充值不
参与点积，也不会为尾块读取超出有效范围的输入。Q8_0 未写入的尾部保留零编码；
Q4_0 使用 nibble 8 编码零值。

`LinearQ80()` 和 `LinearQ40()` 各保留 NEON/scalar 两组条件编译定义，签名和
校验一致；编译时只保留一组，属于平台分支而非 C++ 重载。两组定义复用
`LinearBlocks()`，NEON 分支使用 `DotBlockNeon()` 的格式重载，scalar 分支使用
单元素解码函数。本阶段未增加 AVX2 量化点积或 Q4_1 NEON 点积，也没有 Q4_1
重新量化入口。

## 量化 Linear 的 batch 与线程调度

Q8_0、Q4_0、Q4_1 统一支持以下形状；`in`/`out` 分别表示 `in_features`/`out_features`：

| 输入 | 权重形状 | 输出 |
| --- | --- | --- |
| `[in]` | `[out, in]` | `[out]` |
| `[batch, in]` | `[out, in]` | `[batch, out]` |

输入必须为一维或二维，batch 必须为正数，权重必须为正维度的二维形状，输入
特征数必须匹配。权重块数为：

```text
blocks_per_row = ceil(in_features / 32)
weight_blocks = out_features * blocks_per_row
```

batch 中的各行复用同一份权重，权重块数不乘 batch，也不创建完整 F32 权重矩阵。

外层按 `batch * out_features` 展平任务，每个任务计算一个输出元素：

```text
row          = flat_index / out_features
channel      = flat_index % out_features
input_offset = row * in_features
block_offset = channel * blocks_per_row
output_index = flat_index
```

输入和量化块只读，每个线程独立累加，并写入不同输出位置。内部点积不创建
线程；Q4_1 或无 NEON 时使用 scalar 点积，外层并行仍有效。小任务和单线程配置
复用已有 `ParallelFor()` 的调用线程执行策略。

`ParallelFor()` 的任务数类型是 int，因此在计算乘积和分配输出前检查
`batch * out_features <= INT_MAX`，超限时抛出 `std::runtime_error`。

F32 Linear 仍只支持 `[in_features]` 和 `[1, in_features]`；本阶段扩展的是量化
Linear 算子，没有改变前向推理、KV Cache 或 token 的执行顺序。

## 公开接口与兼容语义

`quant.h` 与 `quant.cc` 的声明和定义已对应，保留上传版本的公开函数签名：

| 接口 | 作用 |
| --- | --- |
| `QuantizeToQ80()`、`QuantizeToQ40()` | 将 F32 张量按最后一维逐行量化 |
| `DequantizeFromQ80/Q40/Q41()` | 根据原始 shape 恢复 F32 张量 |
| `LinearQ80/Q40/Q41()` | 对量化权重执行即时解码 Linear，支持单向量和 batch |
| `MatmulQ80()`、`MatmulQ40()` | 兼容矩阵乘法：完整反量化权重后调用 F32 Matmul |
| `CompareMatmulError()` | 返回 F32 Matmul 与 Q8_0 兼容 Matmul 输出的最大绝对误差 |
| `CompareQ40Error()` | 返回 F32 Linear 与 Q4_0 Linear 输出的最大绝对误差 |

兼容 Matmul 使用 `[m, k] × [k, n] -> [m, n]` 方向，会分配完整 F32 权重张量；
它不是即时解码的量化 GEMM。量化 Linear 使用 `input @ weight^T`，权重方向和
输入顺序与兼容 Matmul 不同。

`CompareQ40Error()` 目前使用单 token F32 Linear 作为基准，输入仍限于 `[k]`
或 `[1, k]`。误差比较函数测量输出误差，不是权重 roundtrip 误差。

## Q4_0 舍入规则

保留原有 `round(v) + 8` 编码，其中 `v = weight / stored_scale`。先对有符号
量化值舍入，再加存储偏移；没有替换为上传版本的 `round(v + 8)`。

在未触发 clamp 时，两种规则的解码整数如下：

| v | 当前 `round(v)` | 上传版本 `round(v + 8) - 8` |
| --- | --- | --- |
| 0.5 | 1 | 1 |
| -0.5 | -1 | 0 |
| -1.5 | -2 | -1 |

当前规则对正负半整数采用对称的远离零舍入。两种规则在表中精确半整数处的
绝对误差相同；本阶段没有进行真实模型权重或推理质量对比，不据此判断哪种
整体误差更小。新增边界测试固定当前行为。

## 测试覆盖与验证结果

`tests/threaded_matmul_test.cc` 现有 14 项测试，保留原有线程调度、F32、Attention
测试，并补充量化路径。每个用例都有目的说明，重点覆盖：

- F32 的 naive、threaded、SIMD、threaded-SIMD 对照；长度 8 覆盖整块，9/10/7
  等长度覆盖 NEON 剩余 1～3 个元素以及 AVX2 尾部。
- Q8_0/Q4_0/Q4_1 的单向量、batch 1/3/5，以及 1/2/3/4 线程结果；批量输出与
  逐行量化 Linear 一致，逐行结果再与反量化权重的 naive Linear 对照。
- 输入特征数 7/8/16/17/24/31/32/33/65，覆盖 SIMD、nibble 半块、整块和跨块尾部。
- 输出通道数 10/48/67，覆盖小任务、并行和不能均分的任务区间。
- 错误输入秩、特征数、权重块数、非正 batch 和输出任务数溢出。
- Q8_0/Q4_0 兼容 Matmul 的计算方向、尾块、线程一致性和异常。
- 误差比较接口的可手算最大误差与全零权重结果；Q4_0 正负半整数两侧的编码
  和解码，以及低、高 nibble 布局。

浮点对照采用 `1e-5 * (1 + abs(reference))` 容差，允许归约顺序及 FMA 导致的
正常差异；量化结果不要求与原始 F32 权重的计算完全一致。

本阶段记录的最新 x86 验证结果为 24/24 通过：14 项 `ThreadedMatmulTest`、
4 项已有 `QuantTest`、6 项已有 `ForwardTest`。复现命令：

```bash
cmake --build build --target mini-llama-threaded-matmul-tests \
  mini-llama-quant-tests mini-llama-forward-tests -j2
ctest --test-dir build \
  -R 'ThreadedMatmulTest|^QuantTest\.|^ForwardTest\.' --output-on-failure
```

F32 点积此前还运行过显式 AVX2+FMA 和 scalar 构建测试。NEON 核心函数通过
AArch64、ARMv7 FMA 和 ARMv7 普通 NEON 语法检查，更新后的完整 `quant.cc` 通过
AArch64 C++17 语法检查。尚未在 ARM 上运行本阶段测试，未进行性能基准比较，
因此没有 NEON 运行结果或加速比结论。上述 24 项也不代表本阶段重新运行了全量
CTest 或 CLI 烟雾测试。

线程调度和 F32 算子的其他背景见
[CPU 多线程与 F32 SIMD 实现说明](cpu_threaded_inference_implementation.md)。
