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
namespace {

// --------------------------------------
// 内部调试输出实现
// --------------------------------------

// 把张量名称、各维度大小、数据类型和元素总数写入指定输出流
void DumpTensorShapeTo(const Tensor& tensor, const std::string& name,
                       std::ostream& output) {
  output << "  tensor: " << name << " shape=[";
  // shape 按张量的逻辑维度顺序输出，维度之间使用逗号分隔
  for (size_t index = 0; index < tensor.shape.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << tensor.shape[index];
  }
  output << "] dtype=f32 size=" << tensor.size() << "\n";
}

// 从一维 logits 中选出数值最大的 k 项，并写出排名、token 编号和数值
void DumpLogitsTopKTo(const Tensor& logits, int k, std::ostream& output) {
  // logits 必须直接对应词表维度，多维张量没有明确的 token 排名语义
  if (logits.num_dims() != 1) {
    output << "  logits: invalid shape for top-k dump\n";
    return;
  }

  const int vocab_size = logits.shape[0];
  // 非正数或超过词表大小时退化为输出完整词表
  if (k <= 0 || k > vocab_size) {
    k = vocab_size;
  }

  // 收集 logits 值和对应的 token 编号
  std::vector<std::pair<float, int>> pairs;
  pairs.reserve(vocab_size);
  for (int index = 0; index < vocab_size; ++index) {
    pairs.push_back({logits.data[index], index});
  }

  // 只排序需要展示的前 k 项，避免为调试少量候选完整排序整个词表
  std::partial_sort(
      pairs.begin(), pairs.begin() + k, pairs.end(),
      [](const auto& left, const auto& right) { return left.first > right.first; });

  // 临时使用固定六位小数，完成后恢复调用方输出流的格式状态
  const std::ios::fmtflags old_flags = output.flags();
  const std::streamsize old_precision = output.precision();
  output << "  logits top-" << k << ":\n";
  for (int index = 0; index < k; ++index) {
    output << "    [" << std::setw(3) << index << "] id=" << std::setw(4)
           << pairs[index].second << " value=" << std::fixed << std::setprecision(6)
           << pairs[index].first << "\n";
  }
  output.flags(old_flags);
  output.precision(old_precision);
}

// 输出四维 KV Cache 的配置、当前位置以及键和值的实际 shape
void DumpKvCacheInfoTo(const KvCache& cache, int current_pos, std::ostream& output) {
  // 正常构造的缓存布局为 [层数, 最大序列长度, KV 头数, 头维度]
  const int n_layers = cache.keys.shape[0];
  const int seq_len = cache.keys.shape[1];
  const int n_kv_heads = cache.keys.shape[2];
  const int head_dim = cache.keys.shape[3];
  output << "  KV cache: layers=" << n_layers << " seq_len=" << seq_len
         << " n_kv_heads=" << n_kv_heads << " head_dim=" << head_dim << "\n";

  // 负值表示调用方没有提供当前位置，此时省略该行
  if (current_pos >= 0) {
    output << "    current_pos: " << current_pos << "\n";
  }

  // 分别输出 keys 和 values，便于发现两个缓存张量的 shape 不一致
  output << "    keys shape: [" << cache.keys.shape[0] << ", " << cache.keys.shape[1]
         << ", " << cache.keys.shape[2] << ", " << cache.keys.shape[3] << "]\n";
  output << "    values shape: [" << cache.values.shape[0] << ", "
         << cache.values.shape[1] << ", " << cache.values.shape[2] << ", "
         << cache.values.shape[3] << "]\n";
}

// --------------------------------------
// 内部基准运行器
// --------------------------------------

// output 为空时只收集统计；非空时同时把详细调试信息写入指定输出流
BenchmarkResult RunBenchmarkTo(const MiniLlamaModel& model,
                               const std::vector<int>& prompt_tokens, int n_predict,
                               unsigned int seed, std::ostream* output) {
  BenchmarkResult result;
  result.n_prompt_tokens = static_cast<int>(prompt_tokens.size());

  // 每次基准创建独立上下文，避免复用上一次运行留下的 KV Cache 和位置状态
  MiniLlamaContext context(&model);

  // 当前 SamplingParams 使用默认零温度，因此基准采用贪心采样
  MiniSampler sampler(seed);
  SamplingParams params;
  params.seed = seed;

  // 预填充阶段一次处理全部提示词元
  if (output != nullptr) {
    *output << "[verbose] prefill " << prompt_tokens.size() << " tokens\n";
  }

  // steady_clock 不受系统时间校准影响；计时区间只覆盖批量前向计算
  const auto prefill_start = std::chrono::steady_clock::now();

  // prompt 从位置零开始一次性提交，返回最后一个提示词元对应的 logits
  const MiniBatch prefill = MiniBatch::FromTokens(prompt_tokens, 0);
  Tensor logits = ForwardBatch(context, model, prefill);
  const auto prefill_end = std::chrono::steady_clock::now();
  result.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

  // 预填充调试输出位于计时区间之外，不计入 prefill_ms
  if (output != nullptr) {
    DumpTensorShapeTo(logits, "logits", *output);
    DumpLogitsTopKTo(logits, 5, *output);
    DumpKvCacheInfoTo(context.kv_cache, context.pos, *output);
  }

  // 首个生成词元直接使用预填充产生的 logits，后续词元逐个解码
  if (n_predict > 0 && result.n_prompt_tokens < model.config.max_seq_len) {
    // 本地序列用于取得上一个 token，并根据总长度限制上下文位置
    std::vector<int> tokens = prompt_tokens;

    // 首次采样不需要额外前向，因此只增加生成数，不增加解码数
    int next_token = sampler.Sample(logits, params);
    tokens.push_back(next_token);
    ++result.n_generated_tokens;

    if (output != nullptr) {
      *output << "[verbose] Decode step 0: token=" << next_token << "\n";
    }

    // 解码计时从第二个生成 token 开始，覆盖循环内前向、采样和详细输出
    const auto decode_start = std::chrono::steady_clock::now();
    for (int index = 1; index < n_predict &&
                        static_cast<int>(tokens.size()) < model.config.max_seq_len;
         ++index) {
      // 上一次采样结果位于序列末尾，其位置等于当前序列长度减一
      const MiniBatch decode =
          MiniBatch::Single(tokens.back(), static_cast<int>(tokens.size() - 1));

      // 每次单 token 前向都会推进上下文状态并产生下一次采样所需的 logits
      logits = ForwardBatch(context, model, decode);
      ++result.n_decode_tokens;

      // 前向完成后立即采样并追加结果，使生成数始终反映实际产生的 token
      next_token = sampler.Sample(logits, params);
      tokens.push_back(next_token);
      ++result.n_generated_tokens;

      // 循环内详细输出属于解码计时的一部分，只应用于诊断而非性能对比
      if (output != nullptr) {
        *output << "[verbose] Decode step " << index << ": token=" << next_token
                << "\n";
        DumpLogitsTopKTo(logits, 5, *output);
      }
    }

    const auto decode_end = std::chrono::steady_clock::now();
    // 没有执行单 token 前向时保持默认零值，避免记录无意义的计时开销
    if (result.n_decode_tokens > 0) {
      result.decode_ms =
          std::chrono::duration<double, std::milli>(decode_end - decode_start)
              .count();
    }
  }

  // 最终缓存信息展示实际停下的位置和整个缓存的固定容量
  if (output != nullptr) {
    DumpKvCacheInfoTo(context.kv_cache, context.pos, *output);
  }

  return result;
}

}  // namespace

// --------------------------------------
// 公共调试接口
// --------------------------------------

// 默认把张量元数据写入标准输出
void DumpTensorShape(const Tensor& tensor, const std::string& name) {
  DumpTensorShapeTo(tensor, name, std::cout);
}

// 默认把 logits 排名写入标准输出
void DumpLogitsTopK(const Tensor& logits, int k) {
  DumpLogitsTopKTo(logits, k, std::cout);
}

// 默认把 KV Cache 信息写入标准输出
void DumpKvCacheInfo(const KvCache& cache, int current_pos) {
  DumpKvCacheInfoTo(cache, current_pos, std::cout);
}

// 便捷重载根据 verbose 开关选择标准输出或完全关闭详细信息
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens, int n_predict,
                             unsigned int seed, bool verbose) {
  return RunBenchmarkTo(model, prompt_tokens, n_predict, seed,
                        verbose ? &std::cout : nullptr);
}

// 输出流重载供命令行和测试注入目标流；verbose 为 false 时不写入该流
BenchmarkResult RunBenchmark(const MiniLlamaModel& model,
                             const std::vector<int>& prompt_tokens, int n_predict,
                             unsigned int seed, bool verbose, std::ostream& output) {
  return RunBenchmarkTo(model, prompt_tokens, n_predict, seed,
                        verbose ? &output : nullptr);
}

}  // namespace mini_llama
