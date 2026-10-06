#include "mini_llama/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <thread>
#include <vector>

namespace mini_llama {

// 进程级线程数配置：正数表示显式指定，0 表示自动使用硬件并发数
// 原子读写允许配置与查询并发进行；该变量不用于发布其他共享数据，
// 因此使用 relaxed 内存序即可，任务数据的同步由 thread 启动和 join 完成
static std::atomic<int> g_thread_count{0};

// 优先使用显式配置；hardware_concurrency() 无法提供结果时可能返回 0
int GetThreadCount() {
  const int configured = g_thread_count.load(std::memory_order_relaxed);
  if (configured > 0) {
    return configured;
  }
  unsigned int hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(hw) : 4;
}

// 非正数统一恢复自动模式，避免将零或负线程数传入任务切分计算
void SetThreadCount(int n) {
  g_thread_count.store(n > 0 ? n : 0, std::memory_order_relaxed);
}

// 同步执行 [0, n) 上的任务：回调收到连续的半开区间 [begin, end)
// 回调应只读共享输入，并只写自己的输出区间；本函数不替回调保护共享状态
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn) {
  // 空任务不启动线程，也不调用 fn，后续切分只需处理正数 n
  if (n <= 0) {
    return;
  }

  // 本次调用只读取一次配置，并限制线程数不超过任务数
  // 后续 SetThreadCount() 的修改不会改变已经确定的区间切分
  const int n_threads = std::min(GetThreadCount(), n);

  // 每线程至少分配 16 个任务，减少线程创建、调度和 join 的额外开销
  constexpr int kMinChunk = 16;
  // 除法比较等价于 n < n_threads * kMinChunk，并避免线程数乘法溢出
  if (n_threads == 1 || n_threads > n / kMinChunk) {
    fn(0, n);
    return;
  }

  // 每个线程使用独立异常槽，join 后统一检查，并行区域不需要锁
  // 异常槽放在工作线程作用域之外，确保引用它们的线程先结束，再销毁槽
  std::vector<std::exception_ptr> exceptions(n_threads);
  {
    std::vector<std::thread> threads;
    // 在启动线程之前预留全部容量，避免线程启动后再扩容容器
    threads.reserve(n_threads);
    // 正常退出、回调抛异常或后续线程创建失败，都先回收已启动线程
    // 回收对象在 threads 之后构造、之前析构；emplace_back 抛异常时也会 join，
    // 防止销毁仍处于 joinable 状态的 std::thread 导致进程终止
    struct JoinThreads {
      std::vector<std::thread>& threads;
      ~JoinThreads() {
        for (auto& thread : threads) {
          if (thread.joinable()) {
            thread.join();
          }
        }
      }
    } join_threads{threads};

    // 每个线程取得 chunk 个任务，前 remainder 个线程各多分配一个
    // 例如 n=67、线程数=3，区间为 [0,23)、[23,45)、[45,67)
    const int chunk = n / n_threads;
    const int remainder = n % n_threads;
    int start = 0;
    for (int t = 0; t < n_threads; ++t) {
      const int count = chunk + (t < remainder ? 1 : 0);
      const int end = start + count;
      // 区间和线程编号按值捕获，避免循环继续推进后工作线程读到新的 start
      // fn 和异常槽按引用捕获，其生命周期由本次同步调用和 join 保证
      threads.emplace_back([start, end, t, &fn, &exceptions]() {
        try {
          fn(start, end);
        } catch (...) {
          // 异常不能逃出线程入口；先记录在本线程的槽中，交给调用线程处理
          exceptions[t] = std::current_exception();
        }
      });
      start = end;
    }
  }

  // 按线程编号检查并重新抛出遇到的第一个异常，其他线程此时也已完成回收
  for (const auto& exception : exceptions) {
    if (exception) {
      std::rethrow_exception(exception);
    }
  }
}

}  // namespace mini_llama
