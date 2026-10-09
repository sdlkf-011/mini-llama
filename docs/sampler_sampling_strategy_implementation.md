# Sampler、采样策略与生成入口阶段实现汇总

> 当前完整模块清单见 [README](../README.md)，入口与构建状态见
> [CLI 同步说明](cli_integration_status.md)。本文中的阶段测试数字、真实模型输出
> 与计时是历史记录；当前实现与最新回归结果以 README 和 CLI 同步说明为准。
>
> 本文记录对应分支完成时的阶段状态，文中的未接入功能不代表当前仓库现状。
> 当前 CLI 装配、特殊 token 校验、终端帮助与逐轮追踪见[命令行与模块同步说明](cli_integration_status.md)。

本阶段（`feature/sampler-sampling-strategy`）在已有 Tensor、Tokenizer、推理上下文和前向传播能力上，
补充了采样参数、贪心采样、温度采样、Top-k 采样，以及把采样参数接入当前生成流程的基础入口。实现保持在现有项目接口范围内，没有提前引入尚未完成的模型加载、
Chat、PromptBuilder、终端交互或后端选择模块。

## 实现范围

### SamplingParams 与 MiniSampler

- `SamplingParams` 集中保存 `temperature`、`top_k` 和 `seed`，供命令行入口和采样器使用。
- `temperature` 默认为 `0.0`，表示使用确定性的贪心采样。
- `top_k` 默认为 `0`，表示不启用 Top-k 截断；设为 `1` 时直接退化为贪心采样。
- `seed` 默认为 `0`，此时使用系统随机源初始化伪随机数生成器；非零 seed 可以复现相同的采样序列。
- `MiniSampler::Sample` 根据参数统一分发到贪心、温度或 Top-k 采样路径。
- 保留自由函数 `SampleGreedy`，兼容已有调用方式。

### 贪心采样

`SampleGreedy` 直接调用现有 `ArgMax` 算子，返回 logits 最大值对应的 token ID。温度小于 `1e-6` 或
`top_k == 1` 时也会进入该路径，避免不必要的概率计算和随机数消耗。

### 温度采样

温度采样对每个候选计算：

```text
probability[i] = exp((logit[i] - max_logit) / temperature)
```

计算前减去最大 logit，降低指数运算溢出的风险。随后对概率归一化，并通过 `std::mt19937` 和均匀分布完成离散抽样。

### Top-k 采样

Top-k 采样先按 logit 从高到低排序，只保留前 `k` 个候选，再在候选范围内应用温度采样：

- `top_k == 0`：直接调用 `SampleTopK` 时属于非法参数；通过统一 `Sample` 入口时表示禁用 Top-k。
- `top_k == 1`：等价于贪心采样。
- `top_k` 大于词表候选数量：自动限制为实际候选数量。
- logit 相同时按 token ID 升序排列，使候选排序行为稳定。

### 输入校验

采样前集中检查以下条件：

- logits 不能为空。
- 所有 logits 必须是有限浮点数。
- `temperature` 必须有限且非负。
- `top_k` 必须非负。

非法输入通过 `std::runtime_error` 报告，避免空候选、NaN、无穷值或非法参数进入概率计算。

## 阶段生成入口与当前接入

`src/main.cc` 提供当前阶段所需的最小生成编排：

1. 解析 `--temperature`、`--top-k` 和 `--seed`，并写入 `SamplingParams`。
2. 同时解析当前生成流程使用的 `--prompt` 与 `--max-tokens`。
3. 使用调用方提供的已加载 `MiniLlamaModel` 和 `ITokenizer` 编码 prompt。
4. 通过 `ForwardBatch` 完成 prompt prefill，取得最后一个 prompt token 的 logits。
5. 使用同一个 `MiniSampler` 循环采样，并通过 `MiniBatch::Single` 执行后续 token 的 decode。
6. 遇到 EOS、达到最大生成数或到达模型上下文上限时停止。

上述步骤是阶段初期的最小入口。当前 `generate` 使用单轮 `GenerateTokens()`；
`run` 使用独立交互循环、ChatSession、PromptBuilder、RadixTree 前缀复用和终端
命令。两者均已接入同一 `SamplingParams`，EOS 和上下文上限由编排层处理。

## 参数解析

阶段初期支持以下采样与生成选项；当前完整选项见 CLI 同步说明：

```text
--prompt <text>
--max-tokens <positive integer>
--temperature <non-negative finite float>
--top-k <non-negative integer>
--seed <unsigned integer>
```

解析使用严格的完整字符串校验，拒绝尾随字符、缺失参数、负 seed、超出整数范围的数值以及未知选项。解析失败由统一入口捕获并写入错误输出流。

## 单元测试

新增 `tests/unit/sampler_test.cc`，每个测试前均有注释说明测试目的，覆盖：

- 贪心采样选择最大 logit，并验证并列最大值的 `ArgMax` 行为。
- 相同 seed 在温度采样下产生相同的 token 序列。
- Top-k 采样不会返回候选集合之外的 token。
- 负温度、NaN 温度、负 Top-k、直接调用时的零 Top-k、空 logits 和非有限 logits 均被拒绝。
- `top_k == 1` 和接近零的温度退化为贪心采样。

构建系统新增独立的 `mini-llama-sampler-tests` 测试目标，`src/sampler.cc` 和 `src/main.cc` 被纳入 `mini_llama_core`。

## 验证方式

```bash
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

当前环境已完成核心库构建、代码格式检查和独立采样烟雾验证。若 CMake 配置时启用测试并可获得 GTest，可通过上述 `ctest` 命令运行
sampler 单元测试。

## 当前边界

- 采样器接受已有 logits，不负责资源加载、前向或 ChatSession 生命周期。
- 当前全局 main 只在可执行 target 启用，模型/词表/模板由 RunMain 装配。
- generate/run 使用采样参数；bench 仍只使用贪心，seed 不影响 benchmark 结果。
- 现有策略是贪心、温度和 Top-k；未实现 Top-p 或重复惩罚。
- 模型加载、GGUF、量化、CPU 调度、请求追踪和基准模块均已存在；CUDA 当前
  接入边界与回归结果见 [后端说明](backend_selection_cuda_runtime_implementation.md)。
