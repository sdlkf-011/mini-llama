# CUDA Ops / Quant 本阶段验证记录

日期：2026-10-09。本记录对应撤回跨阶段实现后的当前代码。
上一轮完整设备前向和 benchmark 报告已归档至
`.cache/cuda-validation-20261009/device-forward-report-superseded.md`，
其中的性能、显存和传输计数不再作为当前实现的验证结论。

## 当前范围与接口

- 移除 `include/mini_llama/cuda_forward.h`、`src/cuda_forward.cc`、`src/cuda_forward.cu`，并清除构建、源码与测试引用。
- 所有此前已提交的 `include/mini_llama` 头文件接口声明、成员和默认值均保持原样；允许修正与实现不符的注释。
- 用户新增的 `cuda_ops.h`、`cuda_quant.h` 接口未改动，仅补充主机/设备数据位置与拷贝约定的注释。
- 撤回 CLI 的 CUDA 推理装配、`--quant` 参数、整模型计数输出，以及前向后端开关、设备上下文成员和 Q4 模型上传扩展。
- 保留独立 CUDA Ops / Quant 的实现、必要输入校验、C++17 构建和测试接入。
- 原有 CPU 算子、F32 cuBLAS、模型及上下文接口保持原有阶段约定。
- 验证前已有的 `src/model.cc` 上传调整和共享所有权测试保留；它们不是本次新增的接入代码。

完整设备前向、Attention/KV 常驻、CLI 量化装配和整模型性能基准属于后续模块，
本阶段不新增这些模块或为它们修改既定接口。

## 环境和命令

Ubuntu 24.04.4 / WSL2，CMake 3.28.3，GCC 13.3，C++17。
实际 GPU 为 RTX 4060 Laptop，Compute Capability 8.9，CUDA Runtime 12.9。
本机临时 CUDA 编译器被 CMake 识别为 Clang 18.1.3。
GPU 测试在授权后可访问实际 GPU 的环境中执行。

```bash
cmake -S . -B build
cmake --build build -j4
ctest --test-dir build --output-on-failure --no-tests=error -R 'cuda_ops|cuda_quant_q8_0|cuda_quant_q4'
ctest --test-dir build --output-on-failure -j4

# 本机临时 SDK 的工具路径；未写入项目代码。
export PATH=/tmp/mini-llama-cuda-compiler/usr/bin:/tmp/mini-llama-cuda-sdk/bin:$PATH
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure --no-tests=error -R 'cuda_ops|cuda_quant_q8_0|cuda_quant_q4'
ctest --test-dir build-cuda --output-on-failure -j4
```

实际 CTest 还使用 `--output-junit /tmp/mini-llama-stage-scope-*.xml` 保存逐项结果。
两套配置和构建均成功，`git diff --check` 通过。

## 实测结果

| 范围 | 注册/执行入口 | 实际通过 | 失败 | 跳过 |
| --- | ---: | ---: | ---: | ---: |
| CPU 算子专项 | 8 | 3 | 0 | 5 |
| GPU 算子专项 | 8 | 5 | 0 | 3 |
| CPU 全量回归 | 189 | 161 | 0 | 28 |
| GPU 全量回归 | 189 | 182 | 0 | 7 |

CPU 专项实际验证 CUDA Ops 和量化入口的 Stub 异常；5 项 GPU 算子测试跳过。
GPU 专项实际执行 5 项算子测试；3 项 CPU 专用 Stub 测试跳过。
GPU 全量跳过 6 项 CPU Stub 和 1 项需要两张 GPU 的测试。
跳过均未计入通过。

Q8_0/Q4_0 独立 Linear 测试覆盖向量、三行 batch、输入宽度
16/32/35/64/86、尾块、bias、常驻设备权重和设备输入，结果与相同量化权重
的 CPU Linear 对照；还下载原始块，确认权重数据未被修改。
非法指针、块数、形状、设备以及截断输入/bias 均按预期抛异常。
Ops 测试对照 CPU 的 RMSNorm、SiLU、Softmax、逐元素计算和两种 RoPE 布局。

测试注册使用真实 `cuda_ops` / `cuda_quant` 名称，没有用它们冒充后续设备前向测试。
原加载器、设备 Buffer、Tensor、F32 前向及 CLI 测试继续按原有入口运行。
CLI 测试自身仍区分 CUDA Runtime 和尚未接入的 CUDA 推理。

## 验证边界

独立算子的 GPU 正确性和 Stub 已验证；当前结果不证明完整显存常驻模型推理
或三条 CUDA benchmark 已实现。前一轮的完整设备前向、计数器和 benchmark
结果仅保留为已撤回实现的历史记录，后续模块接入后需要重新验证。

Q4_1 仅执行了 CPU Stub，未进行其 GPU 专项验证。只有一张 GPU，
多 GPU 测试未执行。之前发现的模型必需稠密权重空存储问题涉及已有上传模块；
按照保持原模块的范围约束，跨模块修复已撤回，该问题没有列为本阶段已修复。

原始日志、JUnit XML 和跳过名称清单见
[本阶段 summary.json](../.cache/cuda-stage-validation-20261009/summary.json)。
