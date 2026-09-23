#include "mini_llama/prompt_builder.h"

#include <cctype>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/chat.h"

// 本模块实现无第三方依赖的轻量 Jinja2 Chat Template 执行器
// 处理流程固定为：解析上下文 -> 计算表达式/条件 -> 编译指令 -> 执行指令
namespace mini_llama {
namespace {

// 模板执行上下文保存消息列表、当前循环消息和生成提示开关
struct EvalContext {
  const std::vector<ChatMessage>* messages = nullptr;
  const ChatMessage* current_message = nullptr;
  size_t message_index = 0;
  bool add_generation_prompt = false;
};

// 去除字符串首尾空白，供表达式和语句解析共用
std::string Trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

// 去除标签内部的 Jinja2 trim marker，保留轻量执行器的既定行为
std::string StripJinjaTrimMarkers(const std::string& raw) {
  std::string value = Trim(raw);
  if (!value.empty() && value.front() == '-') {
    value = Trim(value.substr(1));
  }
  if (!value.empty() && value.back() == '-') {
    value.pop_back();
    value = Trim(value);
  }
  return value;
}

// 计算字符串字面量、变量、消息下标访问以及加号拼接表达式
std::string EvalExpr(const std::string& expression_raw, const EvalContext& context) {
  const std::string expression = Trim(expression_raw);
  if (expression.empty()) {
    return "";
  }

  std::vector<std::string> parts;
  size_t position = 0;
  while (position < expression.size()) {
    while (position < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[position]))) {
      ++position;
    }
    if (position >= expression.size()) {
      break;
    }

    // 单引号内容直接作为字符串字面量，不再进行变量解析
    if (expression[position] == '\'') {
      ++position;
      std::string literal;
      while (position < expression.size() && expression[position] != '\'') {
        literal += expression[position++];
      }
      if (position < expression.size()) {
        ++position;
      }
      parts.push_back(std::move(literal));
    } else {
      const size_t start = position;
      while (position < expression.size() && expression[position] != '+' &&
             expression[position] != '\'') {
        if (expression[position] == '[') {
          ++position;
          while (position < expression.size() && expression[position] != ']') {
            ++position;
          }
          if (position < expression.size()) {
            ++position;
          }
        } else if (expression[position] == '.') {
          ++position;
          while (position < expression.size() &&
                 !std::isspace(static_cast<unsigned char>(expression[position])) &&
                 expression[position] != '+' && expression[position] != '\'' &&
                 expression[position] != '[' && expression[position] != '.') {
            ++position;
          }
        } else {
          ++position;
        }
      }

      // 非字面量部分按受限变量语法求值；未知变量统一为空字符串
      const std::string token = Trim(expression.substr(start, position - start));
      if (token == "add_generation_prompt") {
        parts.push_back(context.add_generation_prompt ? "true" : "false");
      } else if (token == "message['role']" || token == "message[\"role\"]") {
        parts.push_back(context.current_message ? context.current_message->role
                                                : "");
      } else if (token == "message['content']" || token == "message[\"content\"]") {
        parts.push_back(context.current_message ? context.current_message->content
                                                : "");
      } else if (token.rfind("messages[", 0) == 0) {
        // messages[i]['key'] 先验证分隔符，再转换下标，避免异常模板越界
        const size_t index_end = token.find(']');
        if (index_end == std::string::npos || index_end + 1 >= token.size() ||
            token[index_end + 1] != '[') {
          parts.push_back("");
        } else {
          const std::string index_text = Trim(token.substr(9, index_end - 9));
          const size_t key_begin = index_end + 2;
          const size_t key_end = token.find(']', key_begin);

          std::string key;
          if (key_end != std::string::npos) {
            key = token.substr(key_begin, key_end - key_begin);
            if (key.size() >= 2 && ((key.front() == '\'' && key.back() == '\'') ||
                                    (key.front() == '"' && key.back() == '"'))) {
              key = key.substr(1, key.size() - 2);
            }
          }

          try {
            const int index = std::stoi(index_text);
            if (context.messages != nullptr && index >= 0 &&
                index < static_cast<int>(context.messages->size())) {
              const ChatMessage& message = (*context.messages)[index];
              if (key == "role") {
                parts.push_back(message.role);
              } else if (key == "content") {
                parts.push_back(message.content);
              } else {
                parts.push_back("");
              }
            } else {
              parts.push_back("");
            }
          } catch (...) {
            // 非法或越界下标按空字符串处理，避免异常模板中止聊天流程
            parts.push_back("");
          }
        }
      } else if (token == "loop.first") {
        parts.push_back(context.message_index == 0 ? "true" : "false");
      } else if (!token.empty()) {
        parts.push_back("");
      }
    }

    while (position < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[position]))) {
      ++position;
    }

    if (position < expression.size() && expression[position] == '+') {
      ++position;
    }
  }

  std::string result;
  for (const std::string& part : parts) {
    result += part;
  }
  return result;
}

// 计算相等、逻辑与、逻辑或以及单变量条件
bool EvalCondition(const std::string& condition_raw, const EvalContext& context) {
  const std::string condition = Trim(condition_raw);
  if (condition.empty()) {
    return false;
  }

  // 逻辑运算通过递归拆分，子条件仍复用本函数的比较和变量规则
  const size_t and_position = condition.find(" and ");
  if (and_position != std::string::npos) {
    return EvalCondition(condition.substr(0, and_position), context) &&
           EvalCondition(condition.substr(and_position + 5), context);
  }

  const size_t or_position = condition.find(" or ");
  if (or_position != std::string::npos) {
    return EvalCondition(condition.substr(0, or_position), context) ||
           EvalCondition(condition.substr(or_position + 4), context);
  }

  const size_t not_equal_position = condition.find(" != ");
  if (not_equal_position != std::string::npos) {
    return EvalExpr(condition.substr(0, not_equal_position), context) !=
           EvalExpr(condition.substr(not_equal_position + 4), context);
  }

  const size_t equal_position = condition.find(" == ");
  if (equal_position != std::string::npos) {
    return EvalExpr(condition.substr(0, equal_position), context) ==
           EvalExpr(condition.substr(equal_position + 4), context);
  }

  const std::string value = EvalExpr(condition, context);
  if (value == "true" || value == "True") {
    return true;
  }
  if (value == "false" || value == "False") {
    return false;
  }

  return !value.empty();
}

// 编译后的模板指令只覆盖文本、输出、条件和循环
struct Instr {
  enum Type { kText, kOutput, kIf, kEndIf, kFor, kEndFor } type;
  std::string text;
  size_t jump = 0;
};

// 循环帧记录 for 指令位置和当前消息下标
struct ForFrame {
  size_t program_counter = 0;
  size_t index = 0;
};

// 将 Jinja2 子集编译为顺序指令，并为成对标签设置跳转位置
std::vector<Instr> CompileTemplate(const std::string& template_text) {
  std::vector<Instr> instructions;
  std::stack<size_t> if_stack;
  std::stack<size_t> for_stack;
  size_t position = 0;

  while (position < template_text.size()) {
    if (position + 1 < template_text.size() && template_text[position] == '{' &&
        template_text[position + 1] == '%') {
      position += 2;
      const size_t end = template_text.find("%}", position);
      if (end == std::string::npos) {
        break;
      }
      const std::string statement =
          StripJinjaTrimMarkers(template_text.substr(position, end - position));
      position = end + 2;

      // 成对控制标签在编译时记录跳转地址，执行阶段不再查找字符串标签
      if (statement.rfind("for ", 0) == 0) {
        instructions.push_back({Instr::kFor, statement});
        for_stack.push(instructions.size() - 1);
      } else if (statement == "endfor") {
        if (for_stack.empty()) {
          continue;
        }

        const size_t for_program_counter = for_stack.top();
        for_stack.pop();
        instructions.push_back({Instr::kEndFor, "", for_program_counter});

        // 循环结束时，for 指令可直接跳到 endfor 之后的位置
        instructions[for_program_counter].jump = instructions.size();
      } else if (statement.rfind("if ", 0) == 0) {
        instructions.push_back({Instr::kIf, statement.substr(3)});
        if_stack.push(instructions.size() - 1);
      } else if (statement == "endif") {
        if (if_stack.empty()) {
          continue;
        }

        const size_t if_program_counter = if_stack.top();
        if_stack.pop();
        instructions.push_back({Instr::kEndIf});

        // 条件为假时跳过整个 if 块，落在 endif 之后
        instructions[if_program_counter].jump = instructions.size();
      }
    } else if (position + 1 < template_text.size() &&
               template_text[position] == '{' &&
               template_text[position + 1] == '{') {
      position += 2;
      const size_t end = template_text.find("}}", position);
      if (end == std::string::npos) {
        break;
      }

      const std::string expression =
          StripJinjaTrimMarkers(template_text.substr(position, end - position));
      position = end + 2;
      instructions.push_back({Instr::kOutput, expression});
    } else {
      size_t end = position;
      while (end < template_text.size() &&
             !(template_text[end] == '{' && end + 1 < template_text.size() &&
               (template_text[end + 1] == '%' || template_text[end + 1] == '{'))) {
        ++end;
      }

      const std::string text = template_text.substr(position, end - position);
      position = end;
      if (!text.empty()) {
        instructions.push_back({Instr::kText, text});
      }
    }
  }

  // 未闭合标签统一跳到模板末尾，异常模板不会产生非法跳转
  while (!if_stack.empty()) {
    instructions[if_stack.top()].jump = instructions.size();
    if_stack.pop();
  }
  while (!for_stack.empty()) {
    instructions[for_stack.top()].jump = instructions.size();
    for_stack.pop();
  }

  return instructions;
}

// 执行编译后的模板指令并生成最终 prompt
std::string ExecuteTemplate(const std::vector<Instr>& instructions,
                            const std::vector<ChatMessage>& messages,
                            bool add_generation_prompt) {
  std::string output;
  EvalContext context;
  context.messages = &messages;
  context.add_generation_prompt = add_generation_prompt;

  std::stack<ForFrame> for_stack;
  size_t program_counter = 0;
  while (program_counter < instructions.size()) {
    const Instr& instruction = instructions[program_counter];
    switch (instruction.type) {
      case Instr::kText:
        output += instruction.text;
        ++program_counter;
        break;
      case Instr::kOutput:
        output += EvalExpr(instruction.text, context);
        ++program_counter;
        break;
      case Instr::kIf:
        if (EvalCondition(instruction.text, context)) {
          ++program_counter;
        } else {
          program_counter = instruction.jump;
        }
        break;
      case Instr::kEndIf:
        ++program_counter;
        break;
      case Instr::kFor:
        // 首次进入创建循环帧；回跳到同一 for 指令时推进到下一条消息
        if (for_stack.empty() ||
            for_stack.top().program_counter != program_counter) {
          for_stack.push({program_counter, 0});
          if (messages.empty()) {
            for_stack.pop();
            program_counter = instruction.jump;
          } else {
            context.current_message = &messages[0];
            context.message_index = 0;
            ++program_counter;
          }
        } else {
          ForFrame& frame = for_stack.top();
          ++frame.index;
          if (frame.index >= messages.size()) {
            for_stack.pop();
            program_counter = instruction.jump;
          } else {
            context.current_message = &messages[frame.index];
            context.message_index = frame.index;
            ++program_counter;
          }
        }
        break;
      case Instr::kEndFor:
        // EndFor 记录对应的 for 指令位置，用于开始下一轮迭代
        program_counter = instruction.jump;
        break;
    }
  }

  return output;
}

}  // namespace

void PromptBuilder::SetChatTemplate(const std::string& template_str) {
  // 仅保存模板文本；解析延后到 Build，以便同一模板复用不同消息历史
  chat_template_ = template_str;
}

std::string PromptBuilder::Build(const std::vector<ChatMessage>& messages) const {
  // 路由顺序与模板来源无关：空模板、内置 Qwen2、Jinja2 子集、普通回退
  if (chat_template_.empty()) {
    return BuildPlain(messages);
  }
  if (chat_template_ == "qwen2") {
    return BuildQwen2(messages);
  }
  if (chat_template_.size() >= 2 && chat_template_[0] == '{' &&
      (chat_template_[1] == '%' || chat_template_[1] == '{')) {
    return ExecuteTemplate(CompileTemplate(chat_template_), messages, true);
  }
  return BuildPlain(messages);
}

std::string PromptBuilder::BuildPlain(
    const std::vector<ChatMessage>& messages) const {
  // 纯文本格式仅接受三种标准角色，防止未知角色污染提示词
  std::string prompt;
  for (const ChatMessage& message : messages) {
    if (message.role == "system") {
      prompt += "System: " + message.content + "\n";
    } else if (message.role == "user") {
      prompt += "User: " + message.content + "\n";
    } else if (message.role == "assistant") {
      prompt += "Assistant: " + message.content + "\n";
    }
  }
  prompt += "Assistant: ";
  return prompt;
}

std::string PromptBuilder::BuildQwen2(
    const std::vector<ChatMessage>& messages) const {
  // Qwen2 在缺少系统消息时补充默认设定，保持其内置模板语义
  std::string prompt;
  bool has_system_message = false;
  for (const ChatMessage& message : messages) {
    if (message.role == "system") {
      has_system_message = true;
      break;
    }
  }

  if (!has_system_message) {
    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
  }

  for (const ChatMessage& message : messages) {
    // 非标准角色不输出，避免生成不被模型模板识别的段落
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      continue;
    }
    prompt +=
        "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n";
  }
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

}  // namespace mini_llama
