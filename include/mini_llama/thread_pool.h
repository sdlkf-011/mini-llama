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

// 简单的并行 for：将 [0, n) 大致均分为若干块
// 余数优先分配给前面的线程；n <= 0 时不调用 fn
// 单线程或每线程不足 16 个任务时，直接在调用线程执行
// 所有线程结束后才返回；回调异常在 join 后重新抛出
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_THREAD_POOL_H_
