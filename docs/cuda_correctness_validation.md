# CUDA 正确性审计与验证指南

下文的阶段验证日期为 2026-10-11。当时仅执行 `MINI_LLAMA_CUDA=OFF` 的本地构建、
CPU/Stub 测试和脚本跳过路径；没有连接远程服务器、运行 CUDA 二进制或执行
GPU 计算；该阶段表格中的 GPU 状态保留为 **NOT RUN / PENDING GPU VALIDATION**。

2026-10-12 当前工作区已在真实单 GPU 上完成 Release 全量回归：236 项注册，
228 实际通过、8 跳过、0 失败；CPU 为 180 通过、56 跳过、0 失败。
当前结果见[CLI 同步说明](cli_integration_status.md#回归验证)，阶段表格不代表
当前整体状态。后端计数口径在本文中按当前实现更新，远程服务器部署仍未执行。

## 审计结果与最小补充

| 层次 | 已有实现与实际测试名 | 本次补充 |
| --- | --- | --- |
| Backend/Runtime | `backend.cc` 严格校验 cpu/cuda/device；`cuda_runtime.*` 覆盖查询、分配、三方向拷贝、移动所有权、错误和多设备 | 复用；不增加后端注册系统 |
| Tensor/算子 | `cuda_tensor.*`、`cuda_matmul.*`、`cuda_linear.*`、`cuda_ops.*` 覆盖 MatMul、F32 Linear、RMSNorm、SiLU、Add/Mul、Softmax、两种 RoPE | 统一比较诊断；补独立 Embedding 行偏移/边界与大偏移 Softmax |
| 量化 | `cuda_quant_q8_0.*`、`cuda_quant_q4.*` 覆盖主机/常驻权重/设备输入、bias、尾块、大 batch；同量化块 CPU 对照 | Q4_1 增加向量/batch/多种尾块及非法元数据测试 |
| KV/Attention | `cuda_kv_cache.*`、`cuda_attention.*` 覆盖读写、位置/头/设备边界、生命周期、GQA、因果端点与独立 CPU 参考 | KV 数值比较复用诊断；保留 Attention 严格绝对阈值 |
| Forward | `cuda_forward.*`、`cuda_forward_linear.*`、`cuda_forward.CudaForwardAttentionTest.*`、`cuda_forward_device_resident.*` | 明确先计算 CPU logits；计数专用 batch 补数值对照；重上传补数值与路径对照；三种量化 Forward 分别注册 |
| CLI/真实模型 | `scripts/test_backend_cli.py` 已覆盖参数、错误、CPU/CUDA logits、generate/bench/run 和量化 CLI | CTest 直接调用 `scripts/test_backend_cli.py`；GPU 量化 CLI 单列跳过；新增 `cuda-chat-smoke` |
| CMake/CTest | CPU 构建使用 CUDA stub，无 SDK 类型泄漏；CUDA ON 编译三个 `.cu` 并链接 cudart/cuBLAS；GTest discover 注册实际用例 | 保持结构；真实模型 smoke 配置超时与跳过码 77；NumPy 缺失时 golden-logits 明确跳过 |

没有独立名为 `cuda_forward_buffer` 或 `cuda_kv_attention` 的 CTest 项。
Buffer 用例位于 `cuda_forward_device_resident.CudaForwardBufferTest.*`，
KV Attention 位于 `cuda_attention.CudaAttentionGpuTest.*`。
新增量化 Forward 的完整名称为
`cuda_forward_linear.cuda_forward_quant.Q80ProjectionsRemainResident` 和
`cuda_forward_linear.cuda_forward_q4.{Q40,Q41}ProjectionsRemainResident`。
用户要求的专项正则可以直接命中它们，无需重复创建测试目标。

## 数值与执行路径契约

CPU 模型与 CUDA 模型使用同一配置、权重、输入 token/位置及各自上下文。
前向正确性比较 CPU logits 与 CUDA logits 的形状、有限值及逐元素误差；
不能以最终生成文本替代数值对照。关键用例先完成 CPU 基线，再上传 GPU 权重，
并在 CUDA 测量区间之前调用 `ResetModelCudaRuntimeStats`。
F32 常驻权重测试在 Prefill 后再次清零，独立验证 Decode 热路径，并故意修改
CPU 权重副本以确认 Decode 使用设备快照。重上传用例验证兼容 KV 历史仍数值正确。

共享 `RequireCloseTensor` 使用：

```text
abs_error = abs(CUDA - CPU)
relative_error = abs_error / max(1, abs(CPU))
accept = abs_error <= abs_tol OR relative_error <= rel_tol
```

失败包含 Tensor 名称、扁平索引、多维坐标、CPU/CUDA 值、两种误差及两个阈值。
CPU 上的 `CudaComparisonTest.*` 验证绝对/归一化相对接受分支、小值语义、
NaN/Inf 拒绝、形状拒绝和诊断内容。

| 比较 | 阈值与依据 |
| --- | --- |
| 真实模型 Prefill smoke | abs/normalized-relative 各 `2e-3`；参考任务默认值 |
| MatMul/Linear/KV、共享比较器旧调用 | 各 `2e-5`；保留原有更严格的阈值 |
| 基础 Ops | abs `2e-5`、relative `0`；保留原小幅输入绝对误差标准 |
| 独立 Attention | abs `1e-4`、relative `0`；保留原双精度参考标准 |
| F32/GQA Forward | 依用例各 `1e-4` 或 `2e-4`；原有阈值保持 |
| Q8_0/Q4_0/Q4_1 算子 | 各 `2e-5`；CPU/GPU 解码完全相同的量化块，不包含重量化损失 |
| 三格式量化 Forward | 各 `3e-4`；保留原多层累计浮点误差标准，与同格式 CPU 模型对照 |
| CLI tiny logits | 保留 `2e-4 + 2e-4 * abs(CPU)` 的逐元素阈值，补充诊断 |

Q4_1 用手工 FP16 `d/m` 与高低半字节覆盖 `d*q+m`，没有引入新的 Q4_1 重量化器。
模型原始 F32 与重量化结果之间的质量损失属于另一问题；CLI `bench --quant`
已有 model-native logits 误差输出，不能通过放宽 CUDA 同块对照阈值掩盖计算问题。

## 统计范围与 Fallback

F32/Q8_0/Q4_0/Q4_1 均有真实 CUDA Linear 实现。上传分类统计从成功上传的
`CudaUploadedWeight` 快照读取，Embedding/Norm/Bias 不计为 Linear。
本次修正 F32 摘要分母为 F32 权重数量；缺失快照按类型与数量报告。

现有设备常驻 Forward 要求完整设备权重，缺失/不支持的类型直接失败，
没有可安全复用的混合 CPU 量化 fallback。本次保留该契约，诊断明确说明
`CPU fallback unavailable in device-resident forward`，不将其宣传为 CPU 回退。
CPU 路径仍独立可用；显式选择不可用 CUDA 时返回失败，不静默改变用户后端。

- 初始化/上传：`uploaded weights` 与 `gpu memory used` 表示成功上传的静态权重数/字节；后者不是进程总显存，也不包括 KV 与临时激活。
- Forward 计数：成功算子调用后记录 Linear、activation、Attention；`cpu attention fallback calls` 独立记录 CPU Attention 路径。
- 传输：模型 Forward 的 H2D/D2H 次数和字节；权重上传、设备 memset、显式测试 KV 回读不计入该范围。它们不是全局 CUDA profiler。
- CLI generate：`CUDA runtime (prefill + decode)` 累计当前请求；bench 为最后一次正式测量的计数，不叠加预热和其他轮次。聊天 run 沿用跨轮累计口径，`/clear` 重建上下文但不清零模型统计。
- Prefill 与 Decode：trace 分别记录输入位置数及耗时；F32 常驻测试在 Prefill 后清零并独立检查 Decode。没有将累计 CLI 指标冒充 Decode 独立指标。

设 `L` 为层数、`T = prefill_tokens + decode_tokens` 为处理 token 数：
Linear 为 `(7L+1)T`、Attention 为 `LT`、完整 GPU 路径的 CPU Attention fallback 为 0。
热路径 H2D 次数/字节为 0；D2H 为 `T` 次、`T * vocab_size * sizeof(float)` 字节。
ForwardBatch 逐 token 调用原有 ForwardToken，每个位置都计算并下载 logits，
整批返回最后一份；逐步 dump 也使用相同前向接口。
KV 写入为 `2L * n_kv_heads * head_dim * sizeof(float) * T`；读取按查询头的
逻辑访问计数，GQA 会重复计算共享 KV 访问，不等于物理显存带宽。
Kernel/调用计数证明实际分流和成功 API 调用，未引入 CUPTI。
最新全量验证与工作区复用见[五项推理改进](inference_improvements.md)。

## 本次本地验证

实际环境：Linux、CMake 3.28.3、`/usr/bin/c++`、C++17，现有 build 缓存为 Release。

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF
cmake --build build -j4
ctest --test-dir build --output-on-failure \
  --output-junit /tmp/mini-llama-cuda-correctness-cpu-tests.xml
MINI_LLAMA_BIN=./build/mini-llama python3 scripts/test_backend_cli.py
MINI_LLAMA_BIN=./build/mini-llama python3 scripts/test_cuda_chat_smoke.py
```

| 检查 | 实际结果 |
| --- | --- |
| CPU 配置/构建 | 成功，无 CUDA Toolkit 依赖 |
| 全量 CTest | 220 注册，167 实际通过，53 跳过，0 失败 |
| backend_cli_test.py | 16 项：11 通过，5 CUDA 项跳过，0 失败 |
| cuda_chat_smoke_test.py CPU 跳过路径 | 1 项跳过，退出码 77；CTest 正确显示 Skipped |
| Python 语法、git diff --check | 通过 |

53 个 CTest 跳过项为 51 个 GPU 计算用例、1 个需要 CUDA 构建的参数校验用例、
1 个真实模型 smoke；CPU/Stub 通过不表示 GPU 算子执行通过。
本次没有发现 CPU 编译或测试失败。新增 `.cc` CUDA 测试在 CPU 构建下编译成功，
`.cu` 编译与真实 GPU 数值/计数仍未验证。
日志保存在 `/tmp/mini-llama-cuda-correctness-{cpu-build,cpu-ctest,backend-cli}.log`，
JUnit 与测试列表分别为 `/tmp/mini-llama-cuda-correctness-cpu-tests.xml` 和
`/tmp/mini-llama-cuda-correctness-test-list.json`。

## 服务器就绪后的验证（尚未执行）

需要 NVIDIA GPU、兼容驱动、完整 CUDA Toolkit（编译器、头文件、libdevice、
cudart、cuBLAS）、兼容 C++17 主机编译器，以及 CMake/GTest/Python3。
项目 CMake 最低版本为 3.17；指定 `CMAKE_CUDA_ARCHITECTURES` 时使用 3.18+。
源码没有锁定 Toolkit 版本；现有构建说明以 CUDA 12.3 / Clang 18 为示例，
实际服务器应记录所用 Toolkit/编译器版本与 GPU Compute Capability。
GTest 优先使用已安装包，否则 FetchContent 获取 v1.15.2，需要网络或预置缓存。
golden-logits 需要 NumPy；真实模型 smoke 使用 Python 标准库。

先在服务器本地检查依赖与模型，不需要由本次任务连接服务器：

```bash
nvidia-smi
nvcc --version
cmake --version
c++ --version
python3 -c 'import numpy; print(numpy.__version__)'
ls models/tiny models/chat
sha256sum models/chat/*.gguf
```

`models/tiny` 的仓库测试数据必须存在。`models/chat` 需有 CLI 可加载的 GGUF
或 JSON/model.bin，以及匹配的内嵌/外部 Tokenizer；可用
`MINI_LLAMA_CHAT_MODEL=/absolute/path/model.gguf` 覆盖真实模型位置。
缺少真实模型跳过 smoke；模型已存在但配置、Tokenizer、权重损坏则失败。
记录模型哈希、Tokenizer、采样参数与实现版本后才可以定义额外固定 token 回归。
本次没有写入某次示例文本或某个模型的固定生成 ID。

```bash
cmake -S . -B build-cuda \
  -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
./build-cuda/mini-llama --cuda-info
./build-cuda/mini-llama inspect models/chat

ctest --test-dir build-cuda --show-only
ctest --test-dir build-cuda \
  -R 'cuda_quant_q4|cuda_forward_q4|cuda_forward_quant|cuda_forward_device_resident|CudaForwardAttentionTest|cuda-chat-smoke' \
  --output-on-failure --no-tests=error

ctest --test-dir build-cuda --output-on-failure \
  --output-junit /tmp/mini-llama-gpu-tests.xml
MINI_LLAMA_BIN=./build-cuda/mini-llama python3 scripts/test_backend_cli.py
MINI_LLAMA_BIN=./build-cuda/mini-llama python3 scripts/test_cuda_chat_smoke.py
./build-cuda/mini-llama generate --model models/chat --backend cuda \
  -p hello -n 8 --seed 1
```

Toolkit 在非标准位置时使用 `CUDAToolkit_ROOT`；需要时明确选择 CUDA 编译器、
主机编译器与目标 GPU 架构，详见 [README 构建说明](../README.md#构建与测试)。
使用新的 build-cuda 缓存，避免继承另一台机器的 SDK/编译器路径。

GPU smoke 用 `--dump-logits` 增加逐位置数值/有限值检查；CPU 只算相同 prompt 的
Prefill 基线。后续 CUDA Decode 可能因采样形成不同历史，因此只校验有限值、
token 范围和计数；完整多步同输入数值对照由合成模型 Forward 测试完成。

验收必须核对 JUnit/CTest 的跳过名单，而不仅是 `100% tests passed`。
CUDA 构建可跳过 7 个 CPU stub 专用测试；单 GPU 可以跳过多设备所有权测试。
任何 Runtime、MatMul、Linear、Quant、Ops、KV、Attention、Forward 的 GPU
计算用例跳过，或缺模型导致 cuda-chat-smoke 跳过，都表示相应验证尚未完成。
确认上传分类数量、CPU/CUDA 数值误差、真实调用次数、fallback=0、H2D=0、
逐 Forward logits D2H 字节和 KV 逻辑字节全部符合上述契约。
性能耗时与 tokens/s 可以记录为基线；本次未设硬件相关速度阈值。
