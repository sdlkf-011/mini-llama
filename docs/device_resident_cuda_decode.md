# Device-resident CUDA Decode 实现与验证

当前回归日期：2026-10-12；环境：C++17、CUDA 12.3 / Clang 18、RTX 4060 Laptop GPU。
当前 CPU 为 236 注册项、180 通过、56 跳过，CUDA 为 236 注册项、228 通过、
8 跳过，均无失败；细节见[CLI 同步说明](cli_integration_status.md#回归验证)。
后文 2026-10-10 的执行日志与数字保留为历史记录。

## 按指定结构调整

`forward.cc`、`model.cc`、`main.cc` 的私有辅助函数统一采用 `namespace mini_llama` 内的 `static` 函数，保留必要中文注释。

- Forward：使用 `FindCudaWeight`、`RecordCuda*`、主机/设备 Linear 包装、`ForwardQkvProjectionDevice`、`ForwardRmsNormDevice`、`ForwardRopeDevice`、`AttentionForwardDevice`、`FfnForwardDevice`、`ForwardLayerDevice` 和 `ComputeLogitsDevice`。参数、权重完整性和 Cache 配置在计算前校验，CUDA 主路径不走 CPU Attention。
- Model：按 `UploadF32LinearWeight`、`UploadF32TensorWeight`、`UploadQ80LinearWeight`、`UploadQ40LinearWeight`、`UploadQ41LinearWeight` 和 `UploadLinearWeight` 分发上传。统计 getter/reset 使用既定的 `CudaModelWeights` 字段。量化采用 `QuantizeQtToQ80/Q40`；保留先准备全部结果再提交、权重改变后解除旧 CUDA 快照的既有契约。
- CLI：使用 `ParseIntArg/ParseFloatArg/ParseUintArg/ParseBackendArg/ParseDeviceArg`、`ValidateBackendOrPrint`、`PrepareCudaWeightsOrPrint`、`PrintBackendInfo`、`FormatMb`、权重分类/执行/运行摘要、`ApplyQuantOverride`、`RunLogitsForTokens` 和 `LogitsError`。新增 `--quant q8_0|q4_0`，Benchmark 可输出模型原始格式与量化格式的 Logits 误差及权重大小。

2026-10-10 的结构清理移除了多余历史字段和统计继承结构。后续五项改进保持
`context.h`、`cuda_runtime.h`、`debug.h`、`cuda_tensor.h` 的原有定义，
算子工作缓冲区、句柄缓存和额外测量放在现有实现层。
缓存内部使用设备 memset 清零，`WriteDevice` 接受分头布局及同元素数的扁平 Linear 输出。

为兼容当前仓库，保留已有流注入的 `RunMain/Run`、模型/分词器装配、元数据检查、别名和严格参数校验。提供代码里的 `ValuePtr` 按固定接口使用 `valuePtr`；Linear 的扁平 K/V 直接由 `WriteDevice` 写入相同的连续缓存布局，无需 reshape 或额外 Tensor 拷贝。必需的空 Embedding/Norm 仍报错，只有可选 Bias 可省略，避免将 `Tensor::size()` 与 `data.size()` 的同值比较误当作存储校验。

## GPU 数据流与统计

`GPU Embedding → Attention RMSNorm → Q/K/V（GPU Bias）→ RoPE → KV WriteDevice → CUDA Attention → WO → Residual → FFN RMSNorm → Gate/Up → SiLU × Up → Down → Residual → Final RMSNorm → LM Head → D2H Logits → CPU Sampler`

权重、激活和 KV 历史常驻 GPU。`ForwardBatch` 逐 token 调用原有 `ForwardToken`，
每步执行 Final Norm、LM Head 和一次 logits 下载，整批返回最后一份，不新增
内部前向接口。Cache 初始化通过设备 memset 清零，避免隐含 H2D 零值上传。
没有新增 Kernel 或 `cudaDeviceSynchronize`，默认 stream 的依赖顺序与末尾同步下载保持一致。

计数按 `RecordCuda*` 在成功操作后累计。上传与推理统计分开；测试/调试显式回读不纳入模型前向计数。计数器不作为 CUDA 驱动或 cuBLAS 内部操作的全局 profiler。

设 L 为层数、T 为处理 token 数、B 为每 token 的 Bias Add 数：

- Linear：`(7L+1)T`。
- Activation：`(7L+B+1)T`，包括 Norm、RoPE、SiLU、Mul、Add，不包含 Embedding。RoPE 按一次同时旋转 Q/K 的接口调用计数。
- Attention：`LT`；正常 CUDA 主路径 CPU fallback 为 0。
- H2D：0；D2H 为 T 次，`T × vocab_size × 4` bytes。prefill、decode 与逐步
  导出都逐 token 返回 logits。
- KV Write：每 Token 每层 `2 × n_kv_heads × head_dim × 4` bytes。
- KV Read：每位置每层 `2 × (pos+1) × n_heads × head_dim × 4` bytes。统计各查询头的逻辑 K/V 访问，GQA 的共享数据按查询头重复计入，不代表物理 DRAM 流量。

两层、三 token 普通 batch、无 bias、128 词表模型当前验证：Linear **45**、
Activation **45**、Attention **6**、H2D **0**、D2H **3 / 1536 bytes**。
单 token 和逐步导出各自仍下载一份 logits。后文的旧测试数字保留为历史记录。

generate 的 CLI 运行摘要为当前请求的 **Prefill + Decode 总和**，bench 为
最后一次正式测量的总和；
run 的设备计数自权重准备后跨轮累计，`/clear` 只重建会话与 Cache，不清零设备
计数。不能将多个 Forward 的 D2H 总数理解为每次 Token 的拷贝数。

## 2026-10-10 阶段执行与结果

```bash
cmake --build build -j 4
cmake --build build-cuda -j 4
ctest --test-dir build-cuda -N
ctest --test-dir build-cuda --output-on-failure \
  -R 'cuda_forward|cuda_forward_device_resident|cuda_forward_linear|CudaForwardAttentionTest'
ctest --test-dir build --output-on-failure -j 4 \
  --output-junit /tmp/mini-llama-specified-decode-cpu-full.xml
ctest --test-dir build-cuda --output-on-failure -j 4 \
  --output-junit /tmp/mini-llama-specified-decode-gpu-full.xml
./build-cuda/mini-llama generate --model models/chat --backend cuda \
  --prompt hello --n-predict 8 --seed 1
./build-cuda/mini-llama bench models/chat --backend cuda -p hello -n 32 --seed 1
./build/mini-llama bench models/tiny --quant q4_0 -n 2 --seed 1
```

| 验证 | 结果 |
| --- | --- |
| CPU Release 构建 | 成功；211 注册项，163 通过、48 跳过、0 失败 |
| CUDA Release 构建 | 成功；211 注册项，203 通过、8 跳过、0 失败 |
| CUDA Forward 专项 | 15/15 真实 GPU 执行通过 |
| GPU 跳过项 | 7 个 CPU stub 专用测试；1 个多 GPU 测试 |
| 功能覆盖 | CPU/GPU Logits 对照、单步和连续 Decode、GQA、Normal/NeoX、Bias、四种 Linear、Cache 历史/重放及边界检查 |
| CLI 自动化 | generate/bench/run、实际调用数和拷贝字节数、CPU/CUDA Q8_0/Q4_0 覆盖、Benchmark 量化误差输出 |
| 测试组织 | 四个目标 `cuda_forward*_test.cc` 已注册；KV 文件使用 `cuda_kv_cache_test.cc`；没有 `test_*.cc` |

真实模型：`models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf`，24 层、词表 151936；上传 291 个权重，1,469,554,176 bytes。

| Prefill + Decode 指标 | generate 8：8 次 Forward | bench 32：32 次 Forward |
| --- | ---: | ---: |
| Linear Calls | 1352 | 5408 |
| Activation Calls | 1928 | 7712 |
| Attention Calls | 192 | 768 |
| CPU Attention Fallback | 0 | 0 |
| KV Write Bytes | 196608 | 786432 |
| KV Logical Read Bytes | 6193152 | 90832896 |
| H2D Copies / Bytes | 0 / 0 | 0 / 0 |
| D2H Copies / Bytes | 8 / 4861952 | 32 / 19447808 |

两次运行各有一次 Prefill Forward；generate 8 后续 7 次 Decode，bench 32 后续 31 次 Decode。每次下载 607744 bytes，数据量与词表 F32 Logits 一致。生成 ID 为 `[11, 600, 2776, 4460, 311, 1855, 264, 4285]`，与先前 CPU 参考一致。没有 CUDA 执行错误。

本轮日志、JUnit 和测试列表：`.cache/specified-device-decode-20261010/`。上一版逐步真实模型对照与最大绝对误差 2.67e-4 记录仍保留在旧验证目录，但不是本轮重新导出的 Logits 结果。

## 使用约定与限制

现有头文件不增加缓存快照身份或有效前缀字段；前向通过已有 `ctx.model`
检查模型对象身份，整批输入范围在计算前校验。ForwardBatch 不保证显式位置
连续；GPU 前向在历史非空时依据 `token_history` 拒绝越过已有历史的位置，
直接调用 ForwardToken 时历史位置由调用方管理。同配置的同权重重新上传可继续
使用旧 Cache；实际更换权重、切换后端或外部清空 Cache 后，应新建上下文并重放历史。

只测试单 GPU；未运行 compute-sanitizer。共享快照计数仍沿用既有单调用线程约定。
当前由 attention/ops/matmul 实现内部按线程、设备缓存分数工作区、归约标量
及 cuBLAS 句柄，保留 Tensor 独占所有权，不增加独立 workspace 模块；具体行为与验证见
[五项推理改进](inference_improvements.md)。

## 移除多余接口后的复验

`cuda_runtime.h`、`cuda_tensor.h` 及对应 `.cc` 与原有版本一致，不保留新增声明或实现。KV Cache 内部直接调用设备 memset，已有 WriteDevice 直接接受扁平/分头的相同连续布局，无需额外 Tensor 拷贝。相关测试改为验证两种写入形状、错误尺寸和缓存内容保持。

CPU/CUDA Release 重新构建成功；各自全量 211 项注册测试均无失败（CPU：163 通过、48 跳过；CUDA：203 通过、8 跳过）。包括单 Token 的 H2D=0、D2H=1/512 bytes 和三 Token 的 45/45/6 计数验证。日志与 JUnit 保存于 `.cache/remove-extra-cuda-api-20261010/`。

## 项目链路与文档同步复验

2026-10-10 按当前工作区重新配置 CPU/CUDA Release，并检查源码、构建入口、
CLI 帮助、模块说明、README 文档导航与历史阶段描述。历史测试数字保留在各自
阶段，当前结果统一见[CLI 同步说明](cli_integration_status.md#回归验证)。

本次修正两处 CLI 入口差异：`run --quant q8_0|q4_0` 现在复用既有量化准备，
与 generate/bench 一致；bench 默认生成预算恢复为帮助和原接口约定的 16。
多线程脚本注册为 `threaded-cli-smoke`；后端 smoke 增加默认预算校验和
CPU/CUDA 量化聊天、CUDA 量化 benchmark，并检查每轮实际成功，避免聊天
捕获错误后仍返回零退出码导致误报。连续 CUDA 聊天与 `/clear` 后重放也被覆盖。

| 本次验证 | 结果 |
| --- | --- |
| CPU 全量 CTest | 212 注册，164 实际通过、48 跳过、0 失败 |
| CUDA 单 GPU 全量 CTest | 212 注册，204 实际通过、8 跳过、0 失败 |
| 线程 CLI | 1/4/自动配置、9 份 logits、生成 ID 和 benchmark 输出一致 |
| 文档与构建注册 | README + 23 份模块文档的 205 个本地链接/锚点通过；所有模块文档可从 README 到达；全部生产/测试源码已注册；git diff --check 通过 |
| 真实 GGUF 检查 | inspect 与 inspect-gguf 成功 |
| Qwen2 CPU/CUDA 生成 | hello、8 token、seed=1、threads=4；8 份逐位置 logits 有限，生成 ID 完全一致 |
| Qwen2 logits 误差 | 最大绝对误差 `8.01086426e-5`，逐元素满足 `5e-4 + 2e-4 * abs(cpu)` 容差 |
| Qwen2 CUDA bench | 1 个 prompt token、8 个生成 token、7 次 decode；1352 Linear、1928 Activation、192 Attention |
| Qwen2 CUDA 传输 | generate/bench 各 0 H2D、8 D2H / 4861952 bytes、0 CPU Attention fallback |
| Qwen2 CUDA 三轮聊天 | 每轮生成 4 token；第二轮复用 24 token 前缀，仅 prefill 11 个新 token；清空后重新 prefill 20 个 token |

生成 ID 仍为 `[11, 600, 2776, 4460, 311, 1855, 264, 4285]`。真实模型为
`models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf`，无需重新下载。
真实模型 smoke 显式指定预算，未对它运行重量化质量或长上下文性能评估。
仅验证单 GPU，未运行 compute-sanitizer；没有据此报告多 GPU 或并发计数支持。

构建日志、CTest 原始输出、JUnit、真实模型 stdout/stderr 与导出的 logits
保存在 `.cache/project-sync-20261010/`；缓存为本地验证产物，不属于提交内容。
可复现全量与线程专项：

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure -j4
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure -j4
ctest --test-dir build -R 'backend-cli-smoke|threaded-cli-smoke|golden-logits' \
  --output-on-failure --no-tests=error
```

CUDA 配置需要本机完整 Toolkit 与兼容编译器，选择方法见
[README 构建说明](../README.md#构建与测试)。全量测试在 GPU 设备查询失败时
可能跳过计算用例，应核对跳过名单，不能只凭 CTest 的成功状态认定 GPU 链路通过。
