# CUDA KV Cache 与 Decode 注意力实现说明

## 当前实现与阶段关系

本阶段增加 GPU KV Cache 和 CUDA Decode 注意力，并在 `src/forward.cc`
接入 F32 QKV 投影、RoPE、设备缓存写入与历史注意力读取。沿用用户提供的
`cuda_kv_cache.h` 和 `cuda_attention.h` 接口。

上述首次接入时使用混合路径。当前已继续接通 Norm、O 投影、FFN 和最终输出，
支持 F32/Q8_0/Q4_0/Q4_1 完整设备前向；CLI generate/run/bench 已接入。
不完整快照直接报错。阶段验证快照保留在末节，完整当前链路见
[设备驻留 CUDA Decode](device_resident_cuda_decode.md)。

## 缓存布局与资源管理

Key 和 Value 分别由一个 `CudaDeviceBuffer` 独占管理，元素类型为 F32，
采用与 CPU `KvCache` 一致的连续行主序布局：

```text
[n_layers, max_seq_len, n_kv_heads, head_dim]
```

主机单 token 输入形状为 `[n_kv_heads, head_dim]`；设备写入也接受同元素数的
扁平 `[n_kv_heads * head_dim]`。槽位字节偏移为：

```text
((layer * max_seq_len) + pos) * n_kv_heads * head_dim * sizeof(float)
```

单头再增加 `kv_head * head_dim * sizeof(float)`。Reset 检查正维度、非负设备编号、
各次乘法及双 Buffer 总容量溢出；先在临时对象完成分配和清零，再替换旧缓存。
`bytes()` 返回两块显存的实际字节容量之和。

`Clear()` 保留显存与配置，直接通过设备 memset 清零，不上传主机零值。
该操作用于初始化或显式清理，不在逐 token 注意力热路径中执行。

`Write()` 校验输入形状与存储元素数，执行主机到设备拷贝。
`WriteDevice()` 额外校验设备、物理容量与非空状态，仅执行设备到设备拷贝。
两种写入均在拷贝前校验两个输入及索引。

`ReadKey()`、`ReadValue()` 返回 `[n_kv_heads, head_dim]`，单头读取返回 `[head_dim]`。
这些同步回读仅供测试和调试使用。注意力使用 `keys_data()`、`values_data()`
借用整个缓存的设备指针；调用方不得释放指针或在 CPU 上解引用。

## CUDA 注意力

`CudaAttentionDecode()` 是主机调试入口，上传查询向量并下载二维输出。
`CudaAttentionDecodeDeviceInput()` 接受二维或扁平设备查询，返回扁平设备向量，
计算期间不上传查询或回读历史 KV，也不负责写入当前 K/V。

注意力仅访问指定层的 `[0, pos]`，包含当前 token。GQA 按查询头所属分组映射到
KV 头，查询头数必须能被 KV 头数整除。

分数内核将查询头与历史位置展开到网格 x 维，每个线程块归约缩放点积。
输出内核按查询头归约最大值与指数和，再累加历史 Value。共享内存复用前同步，
避免最大值读取与指数和写入发生竞争。启动前检查设备网格限制；启动错误立即
报告，执行期错误由后续同步拷贝或 Runtime 操作报告。

## Context、前向与统计

Context 构造时分配 CPU Cache，不分配 GPU Cache；存在完整上传快照时，
首次 CUDA Forward 从位置 0 分配设备 Cache，后续 token 复用显存与历史。
不自动迁移已有 CPU 前缀，切换后端应新建 Context 并重放历史。

逐 token 链路从设备 Embedding/Norm 开始，执行设备 QKV、可选偏置和 RoPE，
通过 `WriteDevice()` 写入当前槽位。CUDA 注意力直接读取历史 KV；注意力输出
保留在设备继续执行 O 投影和 FFN，仅最终 logits 下载供 CPU 采样。

GPU 推理不镜像更新 CPU Cache。已有 GPU 历史的 Context 若失去匹配设备或完整
权重快照，前向会报错，避免回退后使用陈旧 CPU 历史；应为切换后的路径
创建新的 Context。

KV 写入统计在每层前向调用边界记录两个 token 槽位。
KV 读取统计按查询头累计逻辑 Key/Value 流量，GQA 共享头按各查询头分别计入。
初始化清零和测试回读不计入前向运行统计。正常完整设备链路没有激活 H2D，
每 token 仅一份 logits D2H；KV 逻辑读取不等于物理 DRAM 流量。

## 测试与验证

`cuda_kv_cache_test.cc` 覆盖空状态、构造与 Reset、Clear、主机和设备写入、
逐槽逐头读取、完整行主序布局、非法索引与形状、设备边界、CPU 数据一致性，
以及前向延迟初始化、历史驻留、CPU 前缀切换保护和统计。
`cuda_forward_buffer_test.cc` 另外验证扁平/分头设备写入等价、缓存配置/历史边界。

`cuda_attention_test.cc` 覆盖无 CUDA Stub、独立 CPU 数值参考、多层 GQA、
因果端点、跨线程块宽度的头维度，以及形状、存储和设备校验。
设备探测不可用时明确跳过；探测成功后的计算异常仍使测试失败。

本次 CPU/CUDA 全量回归与真实模型验证见
[CLI 同步说明](cli_integration_status.md#回归验证)，已实际验证单 GPU Cache、
Attention、完整前向与聊天历史复用。多 GPU 与 compute-sanitizer 尚未验证。

历史首次接入阶段：CPU Release 全量为 163 项通过、40 项跳过、0 项失败；
其中 Cache/Attention 专项为 2 项通过、6 项 GPU 用例跳过。

当前可复现专项：

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure -R 'cuda_kv_cache|cuda_attention'
```
