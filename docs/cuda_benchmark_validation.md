# CUDA Benchmark 工具链与远程验收

当前远程 NVIDIA 服务器尚未部署。本次远程 Release 编译、GPU correctness、
generate/run/bench、CUDA 性能和 counters 验收均为 **NOT_RUN**。
本地 CPU 结果不能用于推断 CUDA 性能或 speedup。

提供的目标环境参考：RTX 4090 D、驱动 590.44.01、Toolkit 12.4.131。
这些是部署参考，实际报告必须重新采集。历史 `24/24 passed`、Q8_0 `168/168`、
fallback `0`、H2D `0` 均不是本次测量结果。

## 部署前置条件

- 在目标服务器检出待验收 commit，记录 `git status --short` 和提交号。
- 安装 C++17 编译器、CMake 3.21+（报告使用 CTest JUnit）、完整 CUDA Toolkit、
  兼容驱动和 Python 3。脚本本身仅使用标准库；既有 golden/模型工具使用 NumPy。
- `nvidia-smi` 能访问 GPU；`nvcc --version` 能访问编译器。CMake 与运行时的
  Toolkit/库路径一致。4090 D 可显式指定 `-DCMAKE_CUDA_ARCHITECTURES=89`。
- 保留 `models/tiny` 的 config、bin、词表；`models/chat` 放置项目支持的真实
  Qwen2 Q8_0 GGUF（含 tokenizer 元数据）。不要把其他量化格式命名成 Q8_0。
  脚本会检查 Chat 的原生投影权重确实为 Q8_0。
- 可选真实 Q4：准备支持的 Q4_0/Q4_1 GGUF 和匹配 tokenizer；混合量化会在权重
  类型覆盖表中呈现。`--q4-model` 保留原生格式，不重新量化，也不下载模型。
- 首次构建需要可用 GTest 包或访问 CMake 的 GTest 下载地址。
- Benchmark 前停止其他 GPU/CPU 密集任务，并保持 CPU 线程数、功率设置、prompt、
  token 预算和 seed 一致。默认固定 4 线程、prompt `hello`、32 个生成 token、seed 1。

## 完整远程执行命令

以下命令从仓库根目录执行；只在目标 GPU 部署完成后运行。

<!-- markdownlint-disable MD013 -->
```bash
set -euo pipefail
cmake -S . -B build-cuda \
  -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4

nvidia-smi
nvcc --version
./build-cuda/mini-llama --cuda-info

# 两个前置 suite 由后面的脚本再次自动执行；单独运行便于部署诊断。
ctest --test-dir build-cuda \
  -R 'cuda_quant_q4|cuda_forward_q4|cuda_forward_quant|cuda_forward_device_resident|CudaForwardAttentionTest|cuda-chat-smoke' \
  --output-on-failure
ctest --test-dir build-cuda --output-on-failure

# 主验收：自动记录 targeted/full CTest、6 个 Benchmark 用例、generate/run smoke。
python3 scripts/run_cuda_benchmarks.py \
  --build-dir build-cuda \
  --out "artifacts/benchmarks/$(date +%F)-cuda-final.md"
```
<!-- markdownlint-enable MD013 -->

脚本顺序为环境采集 → CUDA targeted CTest → full CTest → Benchmark Matrix →
Chat CUDA generate/run smoke → 报告。前置 suite 失败或超时会阻止全部性能用例；
CUDA targeted GPU 测试跳过会阻止 CUDA 用例。CTest 返回码为 0 仍会读取 JUnit
中的 GTest `SKIPPED` 输出，避免将没有运行的 GPU 测试计为通过。
CUDA 构建中的 CPU-only stub 测试不适用，仍记录 `NOT_RUN`，不会阻止门禁。
完整 suite 中可选 golden、CPU stub 或多 GPU 测试的跳过也逐项保留，不归入 passed。

无 GPU/未启用 CUDA 时 CUDA 用例记录 `NOT_RUN`；未显式 `--skip-cuda` 时退出码
为 2。任一实际命令、解析、counter 校验或正确性测试失败时退出码为 1。
显式跳过所允许的部分执行可以退出 0，但报告仍标记 `PARTIAL`。
报告不会把环境探测工具缺失作为 CPU benchmark 失败。

单独复现 CLI（脚本的 smoke 同样执行这些行为；run 用 `/exit` 结束首轮后的输入）：

```bash
./build-cuda/mini-llama generate models/chat \
  --backend cuda -p hello -n 8 --seed 1 --threads 4
printf '/exit\n' | ./build-cuda/mini-llama run models/chat \
  --backend cuda -p hello -n 8 --seed 1 --threads 4
./build-cuda/mini-llama bench models/chat \
  --backend cuda -p hello -n 32 --seed 1 --threads 4
```

`run` 可以在单轮失败后返回进程码 0；因此必须确认输出有
`mode=run status=ok` 且 `generated_tokens>0`，不能只看退出码。
脚本已检查成功 trace 和 `status=error`。

真实 Q4 扩展验证（替换为实际模型文件或目录）：

```bash
python3 scripts/run_cuda_benchmarks.py \
  --build-dir build-cuda \
  --q4-model /path/to/real-Q4_0.gguf \
  --out "artifacts/benchmarks/$(date +%F)-cuda-real-q4.md"
```

扩展后是 8 个用例；真实 Q4 CPU/CUDA 使用同一路径、prompt、预算和 seed。
Tiny Q4 用 `--quant q4_0`，Chat Q8_0 使用模型原生量化，Tiny 默认保持加载格式。
Q4_1 如果存在会单独报告覆盖率。格式不支持、模型缺失/损坏、词表不匹配应失败，
不得以跳过或 CPU 回退伪装成功。

## 验收清单与检查指标

<!-- markdownlint-disable MD013 -->
| 项目 | 验收标准 | 当前远程状态 | 结果回填位置 |
| --- | --- | --- | --- |
| CUDA Release | configure/build 成功，CUDA=ON、Release | NOT_RUN | 环境元数据、构建日志 |
| targeted CTest | 实际执行测试全部通过，适用 GPU 测试无跳过；数量从 JUnit 采集 | NOT_RUN | `tests[0]`、原始输出 |
| full CTest | 所有实际执行测试通过；逐项解释 NOT_RUN | NOT_RUN | `tests[1]`、JUnit、原始输出 |
| Chat CUDA generate | 返回 0、成功 trace、有效生成，已有 smoke 比较 CPU/GPU prefill logits | NOT_RUN | `smokes`、cuda-chat-smoke |
| Chat CUDA run | 首轮生成成功，无 status=error，正常 /exit | NOT_RUN | `smokes`、原始输出 |
| Chat CUDA bench | 有完整有限性能字段、counter 校验通过 | NOT_RUN | `chat-q8_0-cuda` |
| 默认 matrix | 6 行均 PASSED，无失败/超时/指标缺失 | NOT_RUN | `benchmarks` |
| 真实 Q4 matrix | 指定真实 Q4 时 8 行均 PASSED，原生 Q4_0/Q4_1 存在 | NOT_RUN | `real-q4-cpu/cuda` |
| Linear 权重覆盖 | 各类型 uploaded=expected；Q8_0 非零；Tiny Q4 全部 Linear 为 Q4_0 | NOT_RUN | `weights` |
| Attention | CUDA 调用数=层数×前向 token 数；CPU fallback=0 | NOT_RUN | 累计 counters |
| GPU KV Cache | 写字节随 token 数线性增加，读字节随历史长度累计；累计读写>0 | NOT_RUN | KV bytes、correctness tests |
| Decode H2D | 累计 copies=0、bytes=0，从而确认 Decode 无 H2D | NOT_RUN | `cuda_counters.total` |
| D2H | 每个 prompt/decode 前向下载一次 logits；累计次数/字节与前向工作量一致，无额外下载 | NOT_RUN | copies/bytes、correctness tests |
| Artifact | MD/JSON/CSV 可读取；默认 CSV 6 行，扩展 8 行；包含 raw logs | NOT_RUN | 同一输出 stem 的三个文件 |
| CPU/CUDA 对比 | 同组有效 token 数一致，Decode speedup=CUDA/CPU；不限定必须>1 | NOT_RUN | `speedup_decode` |
<!-- markdownlint-enable MD013 -->

脚本自动检查 Linear/Attention 调用数、完整权重覆盖、fallback/H2D、D2H 调用数、
KV/logits 累计字节与 token/历史位置工作量一致。
既有 CUDA correctness 测试进一步检查 logits 容差、KV 数据内容和精确字节数。

上传总权重数包括稠密 Embedding、Norm、Bias，不能用它充当 Linear 总数。
Qwen2 有 tied/转换后输出权重，不能预设所有类型之和等于 `168`。
`168/168` 只是在原参考 Chat 模型上 Q8_0 Linear 的历史覆盖数。

报告有效性检查：

<!-- markdownlint-disable MD013 -->
```bash
python3 - "artifacts/benchmarks/$(date +%F)-cuda-final" <<'PY'
import csv, json, pathlib, sys
stem = sys.argv[1]
report = json.loads(pathlib.Path(stem + '.json').read_text())
with open(stem + '.csv', newline='') as stream:
    rows = list(csv.DictReader(stream))
assert pathlib.Path(stem + '.md').read_text().strip()
assert len(rows) == len(report['benchmarks']) == 6
assert report['environment']['cuda_available']
assert report['environment']['build_type'] == 'Release'
assert report['status'] == 'PASSED'
assert all(r['status'] == 'PASSED' for r in report['tests'] + report['benchmarks'] + report['smokes'])
assert all(r['failed'] == 0 for r in report['tests'])
assert all(r['counter_validation'] == 'PASSED' for r in report['benchmarks'] if r['backend'] == 'cuda')
assert all(r['speedup_decode'] is not None for r in report['benchmarks'] if r['backend'] == 'cuda')
print('Artifacts and requested CUDA cases validated; review individual CTest NOT_RUN reasons.')
PY
```
<!-- markdownlint-enable MD013 -->

真实 Q4 报告将 `stem` 换成真实 Q4 报告路径，并将行数改为 8。

## 性能和 Counter 口径

- `Prefill Time`：prompt 的批次构造与 ForwardBatch wall time。当前 CUDA Prefill 内部仍逐 token
  前向。
- `Decode Time`：从第二个生成 token 开始的前向与 CPU 采样循环。
  首个 token 从 prefill logits 采样，计入 Generated，不计入 Decode Tokens。
  正常 `Decode Tokens = max(Generated Tokens - 1, 0)`；上下文上限可能截断预算。
- `Total Time = Prefill + Decode`，不含模型读取、量化、上传、分词、Context 分配及
  首次采样。它是两段计算区间之和，不是进程端到端时延。
- `Tokens/s (total) = Generated / Total seconds`；重点指标
  `Tokens/s (Decode) = Decode Tokens / Decode seconds`。无 decode 工作量时实测 CLI
  返回 0；未执行用例则是 JSON `null` / CSV 空单元格，不能用 0 代替。
- 保留旧 `prefill: ... ms`、`decode:`、throughput 标签，新增 `total: ... ms`。
  解析器也支持 `Prefill/Decode/Total Time (ms)` 和不同大小写；旧日志 total 缺失时
  由 Prefill+Decode 推导，并记录 `derived_metrics`。
- ForwardBatch 返回主机 logits 前使用同步 D2H 下载，因此计时已包含此前 GPU
  工作完成和必要 logits 回读，不是只测异步 kernel enqueue。
- 初始化上传后 CLI 调用 `ResetModelCudaRuntimeStats()`，保留上传/权重显存统计，
  清零模型前向执行统计。基准结束后直接读取原有累计计数，不额外保存阶段快照。
  累计 H2D 为零可证明 Decode 无 H2D；非零累计值无法定位传输发生在哪个阶段，
  需要额外诊断，不能据此声称该传输全部来自 Decode。
- Counters 统计范围是项目插桩的模型前向，不是 CUPTI/Nsight 或所有 Runtime API
  调用。内存分配、初始化上传、D2D、Context 的临时显存不属于热路径 copies。
  `gpu memory used` 仅为上传权重 bytes，不是峰值显存。
- benchmark 总 counters 在 `--quant` 的量化误差对照前向之前输出；误差对照和模型
  量化不计入 Prefill/Decode。实时多线程共享同一模型的其他请求可能污染模型计数，
  正式测试每个用例启动独立进程。
- 不启用 `--verbose`：它会将解码循环内调试 I/O 计入 Decode 时延。
- 当前单次进程运行没有 warm-up、重复统计或方差；首次 kernel/JIT/allocator 成本
  可能影响 Prefill。部署后可多次执行并使用不同输出路径存档，检查波动后再做回归阈值。
  后续可增加 warm-up、重复轮次/中位数/分位数，保持首次响应与稳态指标分开记录。

## 本地可执行验证

```bash
cmake -S . -B build -DMINI_LLAMA_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python3 scripts/run_cuda_benchmarks.py \
  --build-dir build --out artifacts/benchmarks/local-smoke.md \
  --skip-cuda --skip-tests --fast
# 额外跑 full CTest 门禁并保留日志/计数，再执行 CPU matrix：
python3 scripts/run_cuda_benchmarks.py \
  --build-dir build --out artifacts/benchmarks/local-validated.md \
  --skip-cuda --fast
```

`--out` 可用 stem 或 `.md/.json/.csv` 路径，统一生成 `<stem>.md/.json/.csv`，自动
创建父目录。`--fast` 保留所有用例，只将生成预算降至 4；CUDA 行仍保存 NOT_RUN。
缺失模型/二进制、执行失败、超时、缺失或异常 stdout 字段均明确失败；单项失败后
仍生成其余用例状态和三个报告文件。参数输入错误或输出目录不可写会直接报错。

`tests/fixtures/cuda_benchmark_stdout.txt` 是**人工合成的解析 fixture**，
仅用于没有 GPU 时的单元测试，不能作为实测日志或性能结果。

本次本地实际验证（2026-10-11，CPU Release，CUDA=OFF）：

- `cmake --build build -j4` 成功。
- 离线 Python 测试 15/15 通过，已接入 CTest；覆盖解析、缺失/异常字段、CTest
  跳过识别、门禁、失败后继续、子进程超时清理、真实量化格式检查及三类报告。
- `local-smoke` 命令退出 0；3 个 CPU 用例 PASSED，3 个 CUDA 用例 NOT_RUN。
- `local-validated` 命令退出 0；full CTest 实际发现 221 项，168 通过、0 失败、
  53 NOT_RUN，suite wall time 3.583 秒；随后 3 个 CPU 用例通过。
  targeted GPU suite 和 CUDA generate/run smoke 均 NOT_RUN。
- 两套 MD/JSON/CSV 均已生成并读回验证；每份 CSV 6 行，CUDA 性能/counters 为空，
  JSON 为 null，无 CUDA speedup。两套报告总状态均为 PARTIAL。

本地单次 fast CPU 样本（hello、预算 4、seed 1、4 线程）：

| 用例 | 实际生成 / Decode tokens | Decode tok/s |
| --- | --- | ---: |
| tiny-cpu | 4 / 3 | 1220.130 |
| chat-q8_0-cpu | 4 / 3 | 1.512 |
| tiny-q4_0-cpu | 4 / 3 | 1261.371 |

实测完整环境、时延和日志见 [本地验证报告](../artifacts/benchmarks/local-validated.md)、
[JSON](../artifacts/benchmarks/local-validated.json)、
[CSV](../artifacts/benchmarks/local-validated.csv)。fast 只有 3 次 Decode，不能作为
稳态性能结论。本次未提供真实 Q4 模型路径，真实 Q4 CPU/CUDA 扩展实测仍为 NOT_RUN。

## 远程实测回填模板

部署后在新文档或报告中填入实际值，未执行项目保留 NOT_RUN。当前模板无测量数据。

| 信息 | 当前值 |
| --- | --- |
| timestamp_utc / commit / branch / dirty state | NOT_RUN |
| 实际 GPU / Driver / Toolkit / Runtime | NOT_RUN |
| CMake / compiler / Release build | NOT_RUN |
| targeted total / passed / failed / NOT_RUN / seconds | NOT_RUN |
| full total / passed / failed / NOT_RUN / seconds | NOT_RUN |
| generate / run / bench | NOT_RUN |
| MD / JSON / CSV 路径 | NOT_RUN |

<!-- markdownlint-disable MD013 -->
| 对比组 | CPU Decode tok/s | CUDA Decode tok/s | CUDA/CPU | 权重覆盖 | fallback | Decode H2D/D2H | GPU KV bytes | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tiny default | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | NOT_RUN |
| Chat native Q8_0 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | NOT_RUN |
| Tiny Q4_0 override | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | NOT_RUN |
| Real Q4（可选） | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | NOT_RUN |
<!-- markdownlint-enable MD013 -->
