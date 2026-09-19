#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_

#include <chrono>
#include <string>
#include <vector>

namespace mini_llama {

using RequestClock = std::chrono::steady_clock;

// 事件结构
struct TraceEvent {
  std::string stage;        // 事件处于什么阶段
  double elapsed_ms = 0.0;  // 花了多长时间
  int tokens = 0;           // 花了多少 tokens
  std::string detail;       // 扩展阶段
};

// 请求层级上下文，用于追踪单个用户请求从入口到结束
struct RequestContext {
  // 请求身份信息
  std::string trace_id;
  std::string mode;
  std::string backend;
  std::string model_path;

  RequestClock::time_point start_time;
  double total_ms = 0.0;  // 请求耗时

  // tokens 统计
  int prompt_tokens = 0;
  int prefill_tokens = 0;
  int decode_tokens = 0;
  int generated_tokens = 0;

  // 各阶段耗时
  double model_load_ms = 0;
  double tokenize_ms = 0;
  double prefill_ms = 0;
  double decode_ms = 0;
  double sample_ms = 0;

  std::string error;
  std::vector<TraceEvent> events;

  void Start();
  void Finish();
  void RecordEvent(const std::string& stage, double elapsed_ms, int tokens = 0,
                   const std::string& detail = "");
  void SetError(const std::string& message);
  bool Ok() const;
};

std::string NewTraceId();

double ElapsedMs(RequestClock::time_point start);
RequestContext StartRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path);

std::string FormatRequestTraceSummary(const RequestContext& request);
std::vector<std::string> FormatRequestTraceEvents(const RequestContext& request);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_REQUEST_CONTEXT_H_
