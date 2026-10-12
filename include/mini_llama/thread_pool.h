#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_THREAD_POOL_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_THREAD_POOL_H_

#include <functional>

namespace mini_llama {

// 获取为并行操作配置的线程数
// 默认为 std::thread::hardware_concurrency()，无法获取时回退为 4
int GetThreadCount();

// 设置用于并行操作的线程数
// 传入 0 可将其重置为 hardware_concurrency()
void SetThreadCount(int n);

// 常驻工作线程执行同步并行 for：将 [0, n) 大致均分为若干块
// 余数优先分配给前面的任务块；n <= 0 时不调用 fn
// 单线程或每块不足 16 项时，直接在调用线程执行
// 本次全部任务结束后返回，重新抛出回调异常；工作线程保留到进程结束
// 嵌套调用在工作线程内串行执行；不同调用方可以并发提交独立任务
// 线程配置在每次调用开始时读取，池按需增长，较小配置减少提交的分块数
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_THREAD_POOL_H_
