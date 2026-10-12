#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_DEBUG_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_DEBUG_H_

#include <iosfwd>
#include <string>
#include <vector>

#include "mini_llama/kv_cache.h"
#include "mini_llama/model.h"
#include "mini_llama/tensor.h"

namespace mini_llama {

// 基准测试结果
struct BenchmarkResult {
  int n_prompt_tokens = 0;
  int n_generated_tokens = 0;
  int n_decode_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;

  // 返回包含预填充和解码耗时的整体生成吞吐量
  double tokens_per_sec() const {
    const double total_ms = prefill_ms + decode_ms;
    if (total_ms <= 0.0) {
      return 0.0;
    }
    return n_generated_tokens * 1000.0 / total_ms;
  }

  // 返回仅计算逐 token 解码阶段的吞吐量
  double decode_tokens_per_sec() const {
    if (decode_ms <= 0.0 || n_decode_tokens <= 0) {
      return 0.0;
    }
    return n_decode_tokens * 1000.0 / decode_ms;
  }
};

// 调试信息输出辅助函数
void DumpTensorShape(const Tensor& tensor, const std::string& name);
void DumpLogitsTopK(const Tensor& logits, int k);
void DumpKvCacheInfo(const KvCache& cache, int current_pos = -1);

// 显式传递输出流，避免重定向共享的标准输出
void DumpTensorShape(const Tensor& tensor, const std::string& name,
                     std::ostream& output);
void DumpLogitsTopK(const Tensor& logits, int k, std::ostream& output);
void DumpKvCacheInfo(const KvCache& cache, int current_pos, std::ostream& output);

// 执行预填充和解码，并返回相应的耗时统计
// verbose 为 true 时，在各阶段输出详细调试信息
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens, int n_predict,
                             unsigned int seed, bool verbose);

// 使用调用方提供的输出流打印详细信息，便于命令行入口和测试注入输出
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens, int n_predict,
                             unsigned int seed, bool verbose, std::ostream& output);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_DEBUG_H_
