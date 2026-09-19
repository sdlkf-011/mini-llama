#include "mini_llama/request_context.h"

#include <gtest/gtest.h>

// 测试请求启动会生成身份信息、记录阶段事件，并正确格式化摘要和事件日志。
TEST(RequestContextTest, StartRecordAndFormat) {
  auto request = mini_llama::StartRequest("chat", "cpu", "model.gguf");
  EXPECT_FALSE(request.trace_id.empty());
  EXPECT_TRUE(request.Ok());
  request.prompt_tokens = 3;
  request.RecordEvent("tokenize", 1.25, 3, "ok");
  request.Finish();
  auto summary = mini_llama::FormatRequestTraceSummary(request);
  auto events = mini_llama::FormatRequestTraceEvents(request);
  EXPECT_NE(summary.find("status=ok"), std::string::npos);
  EXPECT_NE(summary.find("prompt_tokens=3"), std::string::npos);
  ASSERT_EQ(events.size(), 1u);
  EXPECT_NE(events[0].find("detail=\"ok\""), std::string::npos);
  EXPECT_GE(request.total_ms, 0.0);
}

// 测试错误状态会使请求失败，并将错误文本附加到摘要中；事件保持插入顺序。
TEST(RequestContextTest, ErrorAndEventOrder) {
  mini_llama::RequestContext request;
  request.Start();
  request.SetError("bad input");
  request.RecordEvent("a", 2.0);
  request.RecordEvent("b", 3.0, 1);
  EXPECT_FALSE(request.Ok());
  auto summary = mini_llama::FormatRequestTraceSummary(request);
  EXPECT_NE(summary.find("status=error"), std::string::npos);
  EXPECT_NE(summary.find("error=\"bad input\""), std::string::npos);
  auto events = mini_llama::FormatRequestTraceEvents(request);
  ASSERT_EQ(events.size(), 2u);
  EXPECT_NE(events[0].find("stage=a"), std::string::npos);
  EXPECT_NE(events[1].find("stage=b"), std::string::npos);
}
