# CPU 多线程与 F32 SIMD 实现说明

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。

本文描述当前 CPU Linear、Matmul、Attention 和 CLI 的并行执行路径。
统一接口使用 C++17 `std::thread + join`，每次调用创建线程并等待结束。
本次量化 NEON 点积、batch 与兼容接口的完整总结见
[SIMD 点积、量化 Linear 与 batch 支持阶段总结](simd_quantized_linear_implementation.md)。

## 源码与入口

| 文件 | 职责 |
| --- | --- |
| `include/mini_llama/thread_pool.h`、`src/thread_pool.cc` | 线程配置、连续区间切分与异常回收 |
| `include/mini_llama/matmul_dispatch.h`、`src/matmul_optimized.cc` | F32 执行模式和 SIMD 点积 |
| `include/mini_llama/quant.h`、`src/quant.cc` | 量化 Linear 的 NEON/scalar 点积、batch 和外层并行 |
| `include/mini_llama/ops.h`、`src/ops.cc` | 公开算子与量化格式分发 |
| `src/forward.cc` | CPU Attention 按 query head 并行，保留 GQA 和 bias 语义 |
| `src/main.cc` | `--threads` 参数解析、执行配置和结果展示 |
| `tests/threaded_matmul_test.cc` | 算子、线程切分、异常和 Attention 正确性测试 |
| `tests/CMakeLists.txt` | 注册 `mini-llama-threaded-matmul-tests` 目标和测试前缀 |
| `scripts/test_threaded_cli.py` | 线程配置、logits 与生成结果的进程级测试 |

测试源码使用当前文件名 `tests/threaded_matmul_test.cc`。

## ParallelFor 与线程配置

```cpp
int GetThreadCount();
void SetThreadCount(int n);
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);
```

显式线程数大于零时，`GetThreadCount()` 返回配置值；自动模式使用
`std::thread::hardware_concurrency()`，返回零时回退为 4。`SetThreadCount(0)`
恢复自动模式，配置通过原子变量读写。每次 `ParallelFor()` 读取一次配置，取
`min(配置线程数, n)` 作为本次线程数。`n <= 0` 时不调用回调。

单线程直接执行 `fn(0, n)`。多线程使用 `kMinChunk = 16` 阈值，当
`n < n_threads * 16` 时也直接执行。源码用等价的除法比较避免整数乘法溢出。

超过阈值时，基础区间长度为 `n / n_threads`；前 `n % n_threads` 个线程各多
分配一个任务。区间连续、互不重叠，大小最多相差 1。例如 `n=67、threads=3`：

```text
thread 0: [0, 23)
thread 1: [23, 45)
thread 2: [45, 67)
```

`n=10、threads=3` 属于小任务，实际只有一次 `[0, 10)` 回调，不会创建线程。

各线程写入独立的异常槽，并行回调内部没有异常记录锁。作用域回收对象在正常
退出或后续线程创建失败时 join 已启动线程；回调异常在全部 join 后重新抛出。
因此调用返回或抛异常前，已启动线程对捕获对象的访问已经结束。

## F32 Linear 与 Matmul 分发

公开 `Linear()` 和 `Matmul()` 调用各自的 dispatch，默认模式是 `kThreadedSimd`。

| MatmulMode | F32 Linear | 通用 Matmul |
| --- | --- | --- |
| `kNaive` | 单线程标量点积，正确性基线 | 单线程标量矩阵乘法 |
| `kThreaded` | 输出通道并行，标量点积 | 输出行并行，标量矩阵乘法 |
| `kSimd` | 单线程 SIMD 点积 | 回退到输出行并行 |
| `kThreadedSimd` | 输出通道并行，SIMD 点积 | 回退到输出行并行 |

Linear 权重为 row-major `[out_features, in_features]`。每个输出通道 `j` 使用
`weight.data.data() + j * in_features`，同一个点积完整地交给一个线程。多线程
只划分输出通道；输入和权重只读，结果写入区间互不重叠。

一维输入 `[in_features]` 返回 `[out_features]`；二维输入 `[1, in_features]`
返回 `[1, out_features]`。已有 Q/K/V bias 仍在投影后按原有流程累加。

Matmul 计算 `[m, k] × [k, n] -> [m, n]`，按输出的 `m` 行切分。B 的列在内存中
不连续，所以当前不实现 SIMD GEMM，SIMD 模式复用多线程标量 Matmul。

### SIMD 选择

`DotSimd()` 根据编译目标选择连续向量点积实现：

- `__AVX2__` 与 `__FMA__` 同时定义：启用 AVX2/FMA，向量宽度为 8。
- `__ARM_NEON` 或 `__ARM_NEON__` 定义：启用 NEON，向量宽度为 4。
- 其他情况：使用 scalar fallback。

向量循环之后使用标量处理尾部，支持输入特征数不是向量宽度整数倍。
平台助手分别为 `DotSimdAvx2()` 和 `DotSimdNeon()`，由 `DotSimd()` 统一分发。
NEON 在支持 FMA 的目标上使用 `vfmaq_f32`，其他 NEON 目标保留普通乘加路径。
这些宏反映编译目标，不做运行时 CPU 指令集检测。默认 x86 构建未启用
AVX2/FMA 时使用 scalar fallback；在支持这些指令的目标上，可单独构建：

```bash
cmake -S . -B build-avx2 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-mavx2 -mfma"
cmake --build build-avx2 -j4
python3 scripts/test_threaded_cli.py --binary build-avx2/mini-llama
```

## CPU Attention

`AttentionForward()` 先串行写入当前 K/V，再调用 `ParallelFor(n_heads, ...)`。
每个 query head 独立计算 scores、Softmax 和加权 V，临时张量位于各自的循环
作用域，只写入 `output[head, :]`。

GQA 的多个 query head 可以同时只读同一个 KV head，缓存写入和读取阶段分开，
不会发生跨 head 写冲突。head 数较少时仍使用统一的小任务退化策略。
`ForwardBatch()` 继续按 token 顺序执行，此处并行的是单个 token 内的 head。

## CLI 与 Benchmark

`generate`、`bench`、`run` 接受 `--threads N`，解析后在推理前调用
`SetThreadCount(N)`。新进程默认自动模式；库入口 `Run()` 未指定该选项时保留
调用方已设置的线程数。`0` 恢复自动模式，CLI 拒绝负数、非法文本和超限数值。

```bash
./build/mini-llama generate models/tiny \
  --tokenizer models/tiny/vocab.json \
  --prompt hello --n-predict 4 --threads 4

./build/mini-llama bench models/tiny \
  --tokenizer models/tiny/vocab.json \
  --prompt hello --n-predict 4 --threads 4 --verbose
```

`generate`、`bench` 输出 `threads: 4`，表示已解析的配置线程数，不表示每个算子
都会创建 4 个线程。`bench --verbose` 保留 `[verbose] prefill` 等调试输出，
结果包含 `tokens/s (total):` 和 `tokens/s (Decode):`，也保留原有 `throughput:`
与 `decode throughput:` 标签。正式比较性能时关闭 verbose 并固定线程数。

## 测试与验证

```bash
cmake --build build -j4
ctest --test-dir build -R mini-llama-tests --output-on-failure
ctest --test-dir build --output-on-failure -j4
python3 scripts/test_threaded_cli.py
```

`tests/threaded_matmul_test.cc` 包含 14 项测试，覆盖 1/2/4 线程 F32 Linear/Matmul
与 naive 基准的浮点对照、SIMD 尾部、两种输入秩、小任务与阈值边界、非整除
区间、空任务、超大线程配置和回调异常传播。
量化测试另外覆盖 Q8_0/Q4_0/Q4_1 的 batch 1/3/5 与 1/2/3/4 线程对照、块尾部、
非法 batch、任务数溢出、兼容 Matmul、误差比较接口和 Q4_0 舍入边界。

Attention 使用 4 和 68 个 query head 的模型测试退化与真实并行路径，覆盖 bias、
GQA、逐 token logits 和最终 KV Cache。浮点比较采用 `1e-5 * (1 + abs(reference))`
容差。CLI 测试比较单线程与 4 线程的所有导出 logits 和生成 token，检查自动模式、
非法参数以及 benchmark 输出，成功时打印 `PASS`。

此前多线程阶段的验证快照：全量 CTest 113/113，通过 `mini-llama-tests` 前缀
筛选的 22 项全部通过；CLI 测试比较了 9 份 logits，结果为 `PASS`。
最新量化与 batch 阶段针对 `ThreadedMatmulTest`、`QuantTest`、`ForwardTest` 的
24 项全部通过，未重新运行全量 CTest 或 CLI 测试。NEON 通过语法检查，ARM
运行验证和性能基准仍未完成。

## 当前边界与后续模块

每次调用创建、调度和回收线程的成本仍然存在，频繁的小算子可能无法获得加速。
阈值目前只依据任务数，未考虑每行点积长度或各 head 的序列长度。常驻线程池与
更精细的调度策略可在 `ParallelFor()` 接口之后作为独立模块演进。

Q8_0/Q4_0/Q4_1 继续按 `QuantType` 分发，不进入 `MatmulMode`。三种量化 Linear
均支持 `[in_features]` 和 `[batch, in_features]`，按输入行与输出通道展平任务。
Q8_0/Q4_0 使用 NEON 或 scalar 即时解码点积，Q4_1 使用 scalar 点积，外层并行
在两类路径中均有效。F32 Linear 的输入范围仍为单 token。
完整 SIMD GEMM 与 AVX2 量化点积仍未实现。Backend/CUDA Runtime、权重上传与
设备 Tensor 已有库实现；上传不切换 CPU 前向，CUDA 算子、CLI CUDA 推理和 GPU
Attention/KV Cache 尚未完成，见 [CUDA 说明](backend_selection_cuda_runtime_implementation.md)。
