#include "mini_llama/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mini_llama {

// 进程级线程数配置：正数表示显式指定，0 表示自动使用硬件并发数
// 原子读写允许配置与查询并发进行；该变量不用于发布其他共享数据，
// 因此使用 relaxed 内存序即可，任务提交和完成通过互斥锁与条件变量同步
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

namespace {
// 工作线程里的嵌套 ParallelFor 串行执行，避免等待同一池中的空闲线程
thread_local bool is_pool_worker = false;

struct ParallelBatch {
  explicit ParallelBatch(int count, const std::function<void(int, int)>& callback)
      : remaining(count), exceptions(count), fn(callback) {}
  std::mutex mutex;
  std::condition_variable done;
  int remaining;
  std::vector<std::exception_ptr> exceptions;
  const std::function<void(int, int)>& fn;
};

struct ParallelTask {
  std::shared_ptr<ParallelBatch> batch;
  int index;
  int begin;
  int end;
};

class ThreadPool {
 public:
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& thread : threads_) thread.join();
  }

  void Run(int n, int count, const std::function<void(int, int)>& fn) {
    auto batch = std::make_shared<ParallelBatch>(count, fn);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      // 先 reserve，后启动线程；启动失败时已创建线程由 pool 持有并最终 join
      threads_.reserve(std::max(threads_.size(), static_cast<size_t>(count)));
      while (threads_.size() < static_cast<size_t>(count)) {
        threads_.emplace_back([this] { Worker(); });
      }
      const size_t original_size = tasks_.size();
      int begin = 0;
      try {
        for (int index = 0; index < count; ++index) {
          const int end = begin + n / count + (index < n % count ? 1 : 0);
          tasks_.push_back({batch, index, begin, end});
          begin = end;
        }
      } catch (...) {
        // 队列锁阻止工作线程取走半批任务；入队失败时整体撤回本次提交
        while (tasks_.size() > original_size) tasks_.pop_back();
        throw;
      }
    }
    ready_.notify_all();
    {
      std::unique_lock<std::mutex> lock(batch->mutex);
      batch->done.wait(lock, [&] { return batch->remaining == 0; });
    }
    for (const auto& error : batch->exceptions) {
      if (error) std::rethrow_exception(error);
    }
  }

 private:
  void Worker() {
    is_pool_worker = true;
    for (;;) {
      ParallelTask task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      try {
        task.batch->fn(task.begin, task.end);
      } catch (...) {
        task.batch->exceptions[task.index] = std::current_exception();
      }
      bool finished;
      {
        std::lock_guard<std::mutex> lock(task.batch->mutex);
        finished = --task.batch->remaining == 0;
      }
      if (finished) task.batch->done.notify_one();
    }
  }

  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<ParallelTask> tasks_;
  std::vector<std::thread> threads_;
  bool stopping_ = false;
};

ThreadPool& Workers() {
  static ThreadPool pool;
  return pool;
}
}  // namespace

void ParallelFor(int n, const std::function<void(int begin, int end)>& fn) {
  if (n <= 0) return;
  const int count = std::min(GetThreadCount(), n);
  constexpr int kMinChunk = 16;
  if (is_pool_worker || count == 1 || count > n / kMinChunk) {
    fn(0, n);
    return;
  }
  Workers().Run(n, count, fn);
}

}  // namespace mini_llama
