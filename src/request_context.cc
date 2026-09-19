#include "mini_llama/request_context.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

namespace mini_llama {
namespace {

// 进程内请求序号使用原子变量，保证并发创建请求时不会生成重复序号。
std::atomic<unsigned long long> g_request_counter{0};

// 所有耗时统一保留三位小数，确保摘要和事件日志采用一致格式。
std::string FormatMs(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << value;
  return out.str();
}

}  // namespace

// 使用 steady_clock 记录起点，避免系统时间被校准时影响耗时统计。
void RequestContext::Start() { start_time = RequestClock::now(); }

// 请求完成时计算从 Start 到当前时刻的总耗时。
void RequestContext::Finish() { total_ms = ElapsedMs(start_time); }

// 记录一个阶段事件；tokens 和 detail 可用于补充阶段工作量及诊断信息。
void RequestContext::RecordEvent(const std::string& stage, double elapsed_ms,
                                 int tokens, const std::string& detail) {
  events.push_back({stage, elapsed_ms, tokens, detail});
}

// 保存请求级错误。是否成功由 error 是否为空统一判断。
void RequestContext::SetError(const std::string& message) { error = message; }

bool RequestContext::Ok() const { return error.empty(); }

// Trace ID 由本地时间和进程内递增序号组成，兼顾可读性和并发唯一性。
std::string NewTraceId() {
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  const unsigned long long seq = ++g_request_counter;

  std::tm tm_snapshot;
#if defined(_WIN32)
  // localtime_s/localtime_r 将转换结果写入调用方缓冲区，避免共享静态 tm。
  localtime_s(&tm_snapshot, &time);
#else
  localtime_r(&time, &tm_snapshot);
#endif

  std::ostringstream out;
  out << "req_" << std::put_time(&tm_snapshot, "%Y%m%d%H%M%S") << "_" << std::setw(6)
      << std::setfill('0') << seq;
  return out.str();
}

// 返回从给定 steady_clock 时间点至今经过的毫秒数。
double ElapsedMs(RequestClock::time_point start) {
  return std::chrono::duration<double, std::milli>(RequestClock::now() - start)
      .count();
}

// 构造并启动一次请求，同时记录模式、后端和模型路径等身份信息。
RequestContext StartRequest(const std::string& mode, const std::string& backend,
                            const std::string& model_path) {
  RequestContext request;
  request.trace_id = NewTraceId();
  request.mode = mode;
  request.backend = backend;
  request.model_path = model_path;
  request.Start();
  return request;
}

// 将请求级状态、token 数量和各阶段耗时输出为单行摘要，便于日志检索。
std::string FormatRequestTraceSummary(const RequestContext& request) {
  std::ostringstream out;
  out << "trace summary: trace_id=" << request.trace_id << " mode=" << request.mode
      << " status=" << (request.Ok() ? "ok" : "error")
      << " backend=" << request.backend << " model=" << request.model_path
      << " prompt_tokens=" << request.prompt_tokens
      << " prefill_tokens=" << request.prefill_tokens
      << " decode_tokens=" << request.decode_tokens
      << " generated_tokens=" << request.generated_tokens
      << " model_load_ms=" << FormatMs(request.model_load_ms)
      << " tokenize_ms=" << FormatMs(request.tokenize_ms)
      << " prefill_ms=" << FormatMs(request.prefill_ms)
      << " decode_ms=" << FormatMs(request.decode_ms)
      << " sample_ms=" << FormatMs(request.sample_ms)
      << " total_ms=" << FormatMs(request.total_ms);

  // 仅失败请求附带错误文本，成功摘要保持简洁。
  if (!request.error.empty()) {
    out << " error=\"" << request.error << "\"";
  }
  return out.str();
}

// 将每个阶段事件格式化为独立日志行，并保持事件原始记录顺序。
std::vector<std::string> FormatRequestTraceEvents(const RequestContext& request) {
  std::vector<std::string> lines;
  lines.reserve(request.events.size());

  for (const TraceEvent& event : request.events) {
    std::ostringstream out;
    out << "trace event: trace_id=" << request.trace_id << " stage=" << event.stage
        << " tokens=" << event.tokens
        << " elapsed_ms=" << FormatMs(event.elapsed_ms);
    // detail 是可选字段，避免无内容时产生冗余的空字符串。
    if (!event.detail.empty()) {
      out << " detail=\"" << event.detail << "\"";
    }
    lines.push_back(out.str());
  }

  return lines;
}

}  // namespace mini_llama
