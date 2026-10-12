#include "mini_llama/debug.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "mini_llama/batch.h"
#include "mini_llama/context.h"
#include "mini_llama/forward.h"
#include "mini_llama/sampler.h"

namespace mini_llama {

// --------------------------------------
// 输出张量名称、形状、数据类型和元素数量
// --------------------------------------
void DumpTensorShape(const Tensor& t, const std::string& name) {
  std::cout << "  tensor: " << name;
  std::cout << " shape=[";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << t.shape[i];
  }
  std::cout << "] dtype=f32 size=" << t.size() << "\n";
}

// --------------------------------------
// 输出分数最高的若干个词元及其排名
// --------------------------------------
void DumpLogitsTopK(const Tensor& logits, int k) {
  if (logits.num_dims() != 1) {
    std::cout << "  logits: invalid shape for top-k dump\n";
    return;
  }
  int vocab_size = logits.shape[0];
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  // 收集分数及对应的词元编号
  std::vector<std::pair<float, int>> pairs;
  pairs.reserve(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    pairs.push_back({logits.data[i], i});
  }

  // 按分数降序排列，只排序需要展示的前 k 项
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  std::cout << "  logits top-" << k << ":\n";
  for (int i = 0; i < k; ++i) {
    std::cout << "    [" << std::setw(3) << i << "] id=" << std::setw(4)
              << pairs[i].second << " value=" << std::fixed << std::setprecision(6)
              << pairs[i].first << "\n";
  }
}

// --------------------------------------
// 输出键值缓存的配置、形状和当前位置
// --------------------------------------
void DumpKvCacheInfo(const KvCache& cache, int current_pos) {
  int n_layers = cache.keys.shape[0];
  int seq_len = cache.keys.shape[1];
  int n_kv_heads = cache.keys.shape[2];
  int head_dim = cache.keys.shape[3];
  std::cout << "  KV cache: layers=" << n_layers << " seq_len=" << seq_len
            << " n_kv_heads=" << n_kv_heads << " head_dim=" << head_dim << "\n";
  if (current_pos >= 0) {
    std::cout << "    current_pos: " << current_pos << "\n";
  }
  std::cout << "    keys shape: [" << cache.keys.shape[0] << ", "
            << cache.keys.shape[1] << ", " << cache.keys.shape[2] << ", "
            << cache.keys.shape[3] << "]\n";
  std::cout << "    values shape: [" << cache.values.shape[0] << ", "
            << cache.values.shape[1] << ", " << cache.values.shape[2] << ", "
            << cache.values.shape[3] << "]\n";
}
// --------------------------------------
// 执行预填充和逐词元解码，统计阶段耗时与吞吐量
// --------------------------------------
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens, int n_predict,
                             unsigned int seed, bool verbose) {
  BenchmarkResult result;
  result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

  MiniLlamaContext ctx(&model);
  MiniSampler sampler(seed);
  SamplingParams params;
  params.seed = seed;

  // 预填充：从位置零开始处理全部输入词元
  if (verbose) {
    std::cout << "[verbose] prefill " << prompt_tokens.size() << " tokens\n";
  }

  // 预填充计时包含批次构造与前向，不包含上下文创建及阶段前后的调试输出
  auto prefill_start = std::chrono::steady_clock::now();
  MiniBatch prefill = MiniBatch::FromTokens(prompt_tokens, 0);
  Tensor logits = ForwardBatch(ctx, model, prefill);
  auto prefill_end = std::chrono::steady_clock::now();
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

  if (verbose) {
    DumpTensorShape(logits, "logits");
    DumpLogitsTopK(logits, 5);
    DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
  }

  // 解码：首个生成词元使用预填充结果，后续词元逐步执行前向与采样
  if (n_predict > 0 && result.n_prompt_tokens < model.config.max_seq_len) {
    std::vector<int> tokens = prompt_tokens;
    int next_token = sampler.Sample(logits, params);
    tokens.push_back(next_token);
    ++result.n_generated_tokens;

    if (verbose) {
      std::cout << "[verbose] Decode step 0: token=" << next_token << "\n";
    }

    // 首次采样不计时；后续循环内的前向、采样和调试输出计入解码耗时
    auto decode_start = std::chrono::steady_clock::now();
    for (int i = 1;
         i < n_predict && static_cast<int>(tokens.size()) < model.config.max_seq_len;
         ++i) {
      MiniBatch decode =
          MiniBatch::Single(tokens.back(), static_cast<int>(tokens.size() - 1));
      logits = ForwardBatch(ctx, model, decode);
      ++result.n_decode_tokens;
      next_token = sampler.Sample(logits, params);
      tokens.push_back(next_token);
      ++result.n_generated_tokens;

      if (verbose) {
        std::cout << "[verbose] Decode step " << i << ": token=" << next_token
                  << "\n";
        DumpLogitsTopK(logits, 5);
      }
    }

    auto decode_end = std::chrono::steady_clock::now();
    if (result.n_decode_tokens > 0) {
      result.decode_ms =
          std::chrono::duration<double, std::milli>(decode_end - decode_start)
              .count();
    }
  }

  if (verbose) {
    DumpKvCacheInfo(ctx.kv_cache, ctx.pos);
  }

  return result;
}

// 保留 debug.h 已有的输出流重载，复用上面的原始实现。
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens, int n_predict,
                             unsigned int seed, bool verbose, std::ostream& output) {
  struct RestoreStdout {
    std::streambuf* buffer = std::cout.rdbuf();
    std::ios::fmtflags flags = std::cout.flags();
    std::streamsize precision = std::cout.precision();
    ~RestoreStdout() {
      std::cout.rdbuf(buffer);
      std::cout.flags(flags);
      std::cout.precision(precision);
    }
  } restore;
  std::cout.rdbuf(output.rdbuf());
  return RunBenchmark(model, prompt_tokens, n_predict, seed, verbose);
}

}  // namespace mini_llama
