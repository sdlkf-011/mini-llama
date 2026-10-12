# 五项推理改进

本次改动沿用单上下文、逐 token Transformer 和 CPU 采样的架构，依次处理状态
约束、benchmark、CUDA 执行资源、prefill 和 CPU 工作线程。按照接口约束，
已撤回末尾 logits 优化，prefill 继续使用原有 ForwardToken。

## 1. 模型与上下文匹配

`ForwardToken` 和 `ForwardBatch` 要求 `ctx.model == &model`。形状相同的另一份
模型也不能沿用当前 KV；`ForwardBatch` 在设置 position 前拒绝不匹配的模型。
默认构造、未绑定模型的上下文也会报错。切换模型时新建上下文并重放输入。

此检查验证模型对象身份，不跟踪同一对象中的权重版本；修改权重后仍须由调用方
按原有约定新建上下文。聊天的前缀回退保持原来的历史截断和 position 更新，
不增加 `RetainPrefix` 公开接口。

## 2. Benchmark 阶段与重复测量

CLI 默认预热一次、测量三次，使用 `--warmup` 和 `--repeat` 调整：

```bash
./build/mini-llama bench models/tiny -p hello -n 16 --threads 4 \
  --warmup 1 --repeat 5
```

- `debug.cc` 恢复提供的 `Dump*` 与 `RunBenchmark()` 实现。prefill 计时包括批次
  构造与前向；decode 包括循环内前向、采样和 verbose 输出，首次采样不计时。
- `total` 为 prefill + decode 的平均耗时，另报告 min/median/max；正式比较关闭 verbose。
- token 计数是每次运行的工作量，不乘重复次数，每次调用创建独立上下文并从零重放。
- CUDA 算子/传输摘要为最后一次测量，每轮调用前由 CLI 清零运行计数。
- `debug.h` 保持原有结构与两个公开重载。输出流重载用局部适配复用五参数实现，
  结束或抛异常时恢复 std::cout；不保留 RunBenchmarkMeasured 或跨文件手工声明。
- 新增计时参数只在 main.cc 的 CommandOptions 中，汇总数据保存在实现层私有结构。

## 3. CUDA 执行资源

`context.h`、`cuda_runtime.h`、`debug.h`、`cuda_tensor.h` 均与原版本一致，
不增加字段、类型或公开重载，也不增加独立 workspace 模块。

`CudaTensor` 与 `CudaDeviceBuffer` 继续独占显存，构造、移动、`bytes()` 和释放
行为保持原样。撤掉通用 Tensor 资源池，在现有算子内部复用实际需要的工作区：

- `cuda_ops.cu` 按线程、设备保留一个 RMSNorm 平方和标量，每次使用前清零。
- `cuda_attention.cu` 按线程、设备保留可增长的分数工作区，较小工作区按 2 的幂
  增长；超过 64 MiB 时使用临时分配，不长期占用显存。
- `cuda_matmul.cc` 按线程、设备复用 cuBLAS 句柄，在所属线程退出时释放。

工作区不向调用方返回借用对象。内置算子使用默认 CUDA stream，按 stream 内的
先后顺序复用内存，每步 logits 下载完成同步。共享模型 CUDA 计数仍遵守单调用
线程约定，不提供并发请求的独立统计。

## 4. Prefill 保持原有接口

移除新增的 ForwardTokenInternal。ForwardBatch 直接调用既有 ForwardToken，
逐 token 完成 Transformer、KV 写入、final norm、LM head 与 logits 下载，整批
返回最后一份 logits。模型身份检查与整批边界预检保留，未新增模块或接口。
--dump-logits 继续逐位置导出，与普通 prefill 使用相同的前向实现。

设处理 token 数为 T，层数为 L，每 token 的 bias add 数为 B：

| CUDA 指标 | 当前计算口径 |
| --- | --- |
| Linear 调用 | `(7 × L + 1) × T` |
| Activation 调用 | `(7 × L + B + 1) × T` |
| Attention 调用 | `L × T` |
| D2H 次数 / 字节 | `T` / `T × vocab_size × sizeof(float)` |

此前只计算末尾 logits 的性能与传输数字保留为历史记录，不代表当前实现。

## 5. 常驻 CPU 工作线程

`ParallelFor` 的公开接口、16 项最小分块阈值和均分规则保留。进程级线程池按需
增长，线程在进程结束时统一回收。每次调用只读取一次线程配置，配置降低时减少
提交分块数，已创建的工作线程保持可复用。

每批任务有独立的完成通知和异常槽，调用方等待本批全部任务结束后返回或抛错。
不同调用方可以并发提交独立任务；工作线程中的嵌套调用串行执行，避免等待同一
池中的工作线程。提交失败会撤回半批任务，回调异常不会终止工作线程。

## 验证与实测

2026-10-12 全项目检查另修复 CPU KV 空缓存/存储校验和 F32 转换静默清零，
新增 3 个回归用例。当前 CPU/CUDA Release 均构建成功，各注册 236 项。验证结果区分
实际执行与跳过；sanitizer 行为此前实现阶段的专项记录：

| 验证 | 实际通过 | 跳过 | 失败 |
| --- | ---: | ---: | ---: |
| CPU 全量 CTest | 180 | 56 | 0 |
| CUDA 全量 CTest，真实 GPU，串行执行 | 228 | 8 | 0 |
| 此前 CPU ASan/UBSan 专项 | 55 | 2 | 0 |
| 此前 CUDA ASan/UBSan 专项 | 0 | 12 | 0 |

CPU 跳过项为 CUDA 用例；CUDA Release 跳过项为 CPU stub 和多 GPU 用例。
CUDA sanitizer 构建在 WSL 的 `cudaGetDeviceCount` 驱动初始化阶段返回 out of
memory，12 项未执行，不能视为通过 GPU sanitizer 校验；诊断还记录了驱动中的
LeakSanitizer 报告。普通 CUDA Release 的数值、生命周期、量化和真实模型聊天
用例均实际执行通过。并行全量测试曾遇到设备 busy/unavailable，最终采用串行 GPU
测试完成验证。

本机 RTX 4060 Laptop GPU、CUDA 12.3、Release，模型为
`Qwen2-0.5B-Instruct-Q8_0.gguf`，prompt 为
`Explain how attention and KV cache work in a language model.`，13 个 prompt
token、16 个生成 token、15 次 decode、4 个 CPU 线程：

| 指标 | 改动前 | 撤回末尾 logits 优化前的历史测量 |
| --- | ---: | ---: |
| Prefill 平均耗时 | 410.763 ms | 262.726 ms |
| Decode 平均耗时 | 403.214 ms | 351.847 ms |
| 推理总耗时 | 813.977 ms | 617.176 ms |
| Logits D2H 次数 / 字节 | 28 / 17,016,832 | 16 / 9,723,904 |

该表为恢复 debug.cc、撤回末尾 logits 优化之前的历史测量，不代表当前实现。
改动前为三个独立单次运行的均值；该阶段预热一次、正式测量三次，采样
单独计时并包含在总耗时中。该场景 prefill 约 1.56x，总耗时约 1.32x；包含
算子工作区、句柄复用、prefill 和预热策略共同影响，不单独归因于某一项。
当前 CudaTensor 保持原类型与分配行为，仅保留算子工作区和句柄复用，
当前逐 token 数值回归保持一致，传输计数恢复为每个前向 token 一份 logits。

CPU 调度微基准：4 个线程，每次处理 256 项，预热 10 次后运行 1,000 次；每个
版本测量五次。原实现中位数 163.141 ms，常驻池 51.127 ms，约 3.19x。这只衡量
小任务调度开销，不代表完整模型推理的加速比例。

原始日志和 JSON 位于 `.cache/five-improvements/`。恢复原 Tensor 后的全量日志/JUnit 保存为
`/tmp/mini-llama-original-tensor-{cpu,cuda}.{log,xml}`，CPU sanitizer 保存为
`/tmp/mini-llama-existing-modules-asan.{log,xml}`；GPU sanitizer 的旧诊断保留在
`/tmp/mini-llama-five-asan-cuda.{log,xml}`。测试覆盖同形状模型误用、慢输出
计时语义、重复测量、GPU Buffer 生命周期、缓存重放、逐 token logits 计数、线程复用、
并发、嵌套及异常恢复，NumPy golden 与 CLI dump 回归也已通过。报告脚本同时验证
原有 prefill＋decode 总耗时、逐 token logits 计数及 CSV 字段，同时兼容此前
带 sample 字段的日志。恢复 debug.cc 并移除内部前向入口后的日志/JUnit 为
`/tmp/mini-llama-original-batch-{cpu,cuda}.{log,xml}`。

全项目复验的最新日志/JUnit 为 `/tmp/mini-llama-project-audit-{cpu,cuda}.{log,xml}`；
完整检查范围见[CLI 同步说明](cli_integration_status.md#2026-10-12-全项目检查)。
