# CUDA KV Cache 与 Decode 注意力实现说明

## 本阶段范围

本阶段增加 GPU KV Cache 和 CUDA Decode 注意力，并在 `src/forward.cc`
接入 F32 QKV 投影、RoPE、设备缓存写入与历史注意力读取。沿用用户提供的
`cuda_kv_cache.h` 和 `cuda_attention.h` 接口。

Norm、O 投影、FFN 和最终输出继续使用已有分发方式。当前实现仍在这些模块边界
上传或下载激活，不代表完整设备驻留前向。量化或不完整 QKV 快照继续走原有混合
回退路径；CLI 的 CUDA 后端装配属于后续模块。

## 缓存布局与资源管理

Key 和 Value 分别由一个 `CudaDeviceBuffer` 独占管理，元素类型为 F32，
采用与 CPU `KvCache` 一致的连续行主序布局：

```text
[n_layers, max_seq_len, n_kv_heads, head_dim]
```

单 token 输入形状为 `[n_kv_heads, head_dim]`。槽位字节偏移为：

```text
((layer * max_seq_len) + pos) * n_kv_heads * head_dim * sizeof(float)
```

单头再增加 `kv_head * head_dim * sizeof(float)`。Reset 检查正维度、非负设备编号、
各次乘法及双 Buffer 总容量溢出；先在临时对象完成分配和清零，再替换旧缓存。
`bytes()` 返回两块显存的实际字节容量之和。

`Clear()` 保留显存与配置，复用现有拷贝接口分块上传主机零值，以限制临时主机内存。
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

Context 同时保留 CPU 与 GPU Cache。构造 Context 不分配 GPU Cache；所有层
F32 QKV 权重可用时，在首次进入设备注意力链路时分配。已有 CPU 前缀仅在该次
初始化时迁移。

逐 token 链路先上传现有 CPU Norm 输出，再执行设备 QKV、可选偏置和 RoPE，
通过 `WriteDevice()` 写入当前槽位。CUDA 注意力直接读取历史 KV；注意力输出
下载后衔接原有 O 投影和后续计算。

GPU 推理不镜像更新 CPU Cache。已有 GPU 历史的 Context 若失去匹配设备或完整
F32 QKV 快照，前向会报错，避免回退后使用陈旧 CPU 历史；应为切换后的路径
创建新的 Context。

KV 写入统计在前向调用边界记录两个 token 槽位，首次前缀迁移也记录写入字节。
KV 读取统计按查询头累计逻辑 Key/Value 流量，GQA 共享头按各查询头分别计入。
初始化清零和测试回读不计入前向运行统计；前缀迁移记录主机到设备拷贝，
常规激活边界记录上传下载，避免重复统计。

## 测试与验证

`test_cuda_kv_cache.cc` 覆盖空状态、构造与 Reset、Clear、主机和设备写入、
逐槽逐头读取、完整行主序布局、非法索引与形状、设备边界、CPU 数据一致性，
以及前向延迟初始化、历史迁移和统计。

`cuda_attention_test.cc` 覆盖无 CUDA Stub、独立 CPU 数值参考、多层 GQA、
因果端点、跨线程块宽度的头维度，以及形状、存储和设备校验。
设备探测不可用时明确跳过；探测成功后的计算异常仍使测试失败。

上一轮 CPU Release 构建成功，完整 CTest 为 163 项通过、40 项跳过、0 项失败；
其中 Cache/Attention 专项为 2 项通过、6 项 GPU 用例跳过。本轮仅调整注释和文档，
未重复执行构建与测试。

CUDA 编译、GPU 数值、多设备行为和性能尚未验证，后续执行：

```bash
cmake -S . -B build-cuda -DMINI_LLAMA_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j4
ctest --test-dir build-cuda --output-on-failure -R 'cuda_kv_cache|cuda_attention'
```
