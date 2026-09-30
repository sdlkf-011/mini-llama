#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

// 该解析器只服务于数组形式的简单词表：
// [{"id": 0, "content": "<unk>", "special": true}, ...]
namespace {

// 跳过 JSON 语法允许的空白字符
void SkipWhitespace(const std::string& text, size_t& position) {
  while (position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
}

// 将 Unicode 码点追加为 UTF-8，保证中文等非 ASCII 转义能够正确加载
void AppendUtf8(unsigned int codepoint, std::string& output) {
  if (codepoint <= 0x7f) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ff) {
    output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
}

// 读取四位十六进制数；非法数字会立即终止词表加载
unsigned int ParseHex4(const std::string& text, size_t& position) {
  if (position + 4 > text.size()) {
    throw std::runtime_error("incomplete unicode escape in vocab JSON");
  }
  unsigned int value = 0;
  for (int index = 0; index < 4; ++index) {
    const char digit = text[position++];
    value <<= 4;
    if (digit >= '0' && digit <= '9') {
      value += digit - '0';
    } else if (digit >= 'a' && digit <= 'f') {
      value += digit - 'a' + 10;
    } else if (digit >= 'A' && digit <= 'F') {
      value += digit - 'A' + 10;
    } else {
      throw std::runtime_error("invalid unicode escape in vocab JSON");
    }
  }
  return value;
}

// 解析 JSON 字符串以及标准转义，\uXXXX 会被转换成 UTF-8
std::string ParseJsonString(const std::string& text, size_t& position) {
  SkipWhitespace(text, position);
  if (position >= text.size() || text[position] != '"') {
    throw std::runtime_error("expected JSON string in vocab entry");
  }
  ++position;

  std::string result;
  while (position < text.size()) {
    const char character = text[position++];
    if (character == '"') {
      return result;
    }
    if (character != '\\') {
      result.push_back(character);
      continue;
    }
    if (position >= text.size()) {
      throw std::runtime_error("unterminated escape in vocab JSON");
    }

    const char escaped = text[position++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        result.push_back(escaped);
        break;
      case 'b':
        result.push_back('\b');
        break;
      case 'f':
        result.push_back('\f');
        break;
      case 'n':
        result.push_back('\n');
        break;
      case 'r':
        result.push_back('\r');
        break;
      case 't':
        result.push_back('\t');
        break;
      case 'u': {
        unsigned int codepoint = ParseHex4(text, position);
        if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
          if (position + 2 > text.size() || text[position] != '\\' ||
              text[position + 1] != 'u') {
            throw std::runtime_error("missing low surrogate in vocab JSON");
          }
          position += 2;
          const unsigned int low = ParseHex4(text, position);
          if (low < 0xdc00 || low > 0xdfff) {
            throw std::runtime_error("invalid low surrogate in vocab JSON");
          }
          codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
        } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
          throw std::runtime_error("unexpected low surrogate in vocab JSON");
        }
        AppendUtf8(codepoint, result);
        break;
      }
      default:
        throw std::runtime_error("unsupported escape in vocab JSON");
    }
  }
  throw std::runtime_error("unterminated string in vocab JSON");
}

// 解析十进制整数，并检查空值和 int 范围溢出
int ParseJsonInt(const std::string& text, size_t& position) {
  SkipWhitespace(text, position);
  const size_t start = position;
  if (position < text.size() && text[position] == '-') {
    ++position;
  }
  const size_t digits = position;
  while (position < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  if (digits == position) {
    throw std::runtime_error("expected integer in vocab entry");
  }
  try {
    return std::stoi(text.substr(start, position - start));
  } catch (const std::exception&) {
    throw std::runtime_error("integer is out of range in vocab entry");
  }
}

// 解析 JSON 布尔字面量
bool ParseJsonBool(const std::string& text, size_t& position) {
  SkipWhitespace(text, position);
  if (text.compare(position, 4, "true") == 0) {
    position += 4;
    return true;
  }
  if (text.compare(position, 5, "false") == 0) {
    position += 5;
    return false;
  }
  throw std::runtime_error("expected boolean in vocab entry");
}

// 跳过未知字段的值；支持字符串、对象、数组和普通字面量
void SkipJsonValue(const std::string& text, size_t& position) {
  SkipWhitespace(text, position);
  if (position >= text.size()) {
    throw std::runtime_error("missing value in vocab entry");
  }
  if (text[position] == '"') {
    ParseJsonString(text, position);
    return;
  }

  if (text[position] == '{' || text[position] == '[') {
    const char opening = text[position++];
    const char closing = opening == '{' ? '}' : ']';
    int depth = 1;
    while (position < text.size() && depth > 0) {
      if (text[position] == '"') {
        ParseJsonString(text, position);
      } else {
        if (text[position] == opening) {
          ++depth;
        } else if (text[position] == closing) {
          --depth;
        }
        ++position;
      }
    }

    if (depth != 0) {
      throw std::runtime_error("unterminated value in vocab entry");
    }
    return;
  }

  const size_t start = position;
  while (position < text.size() && text[position] != ',' && text[position] != '}') {
    ++position;
  }
  if (start == position) {
    throw std::runtime_error("invalid value in vocab entry");
  }
}

}  // namespace

JsonVocabTokenizer::JsonVocabTokenizer(const std::string& vocab_path) {
  // 一次性读取词表文件，读取和格式错误统一通过异常报告给工厂函数
  std::ifstream input(vocab_path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open vocab: " + vocab_path);
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  if (input.bad()) {
    throw std::runtime_error("failed to read vocab: " + vocab_path);
  }
  const std::string text = buffer.str();

  size_t position = 0;
  SkipWhitespace(text, position);
  if (position >= text.size() || text[position++] != '[') {
    throw std::runtime_error("vocab JSON must be an array");
  }

  int max_id = -1;
  while (true) {
    SkipWhitespace(text, position);
    if (position < text.size() && text[position] == ']') {
      ++position;
      break;
    }
    if (position >= text.size() || text[position++] != '{') {
      throw std::runtime_error("vocab array item must be an object");
    }

    VocabEntry entry;
    bool has_id = false;
    bool has_content = false;
    while (true) {
      SkipWhitespace(text, position);
      if (position < text.size() && text[position] == '}') {
        ++position;
        break;
      }

      const std::string key = ParseJsonString(text, position);
      SkipWhitespace(text, position);
      if (position >= text.size() || text[position++] != ':') {
        throw std::runtime_error("expected ':' after vocab field name");
      }

      if (key == "id") {
        entry.id = ParseJsonInt(text, position);
        has_id = true;
      } else if (key == "content") {
        entry.content = ParseJsonString(text, position);
        has_content = true;
      } else if (key == "special") {
        entry.special = ParseJsonBool(text, position);
      } else {
        SkipJsonValue(text, position);
      }

      SkipWhitespace(text, position);
      if (position < text.size() && text[position] == ',') {
        ++position;
        continue;
      }
      if (position >= text.size() || text[position] != '}') {
        throw std::runtime_error("expected ',' or '}' in vocab entry");
      }
    }

    if (!has_id || !has_content) {
      throw std::runtime_error("vocab entry missing id or content");
    }
    if (entry.id < 0) {
      throw std::runtime_error("vocab entry id must be non-negative");
    }

    max_id = std::max(max_id, entry.id);
    if (static_cast<size_t>(entry.id) >= id_to_entry_.size()) {
      id_to_entry_.resize(static_cast<size_t>(entry.id) + 1);
    }
    id_to_entry_[entry.id] = entry;
    content_to_id_.emplace_back(entry.content, entry.id);

    // 兼容项目现有模型使用的常见控制 token 名称
    if (entry.content == "<bos>" || entry.content == "<s>" ||
        entry.content == "<|begin_of_text|>") {
      bos_id_ = entry.id;
    } else if (entry.content == "<eos>" || entry.content == "</s>" ||
               entry.content == "<|end_of_text|>") {
      eos_id_ = entry.id;
    } else if (entry.content == "<unk>" || entry.content == "<UNK>") {
      unk_id_ = entry.id;
    }

    SkipWhitespace(text, position);
    if (position < text.size() && text[position] == ',') {
      ++position;
      continue;
    }
    if (position >= text.size() || text[position] != ']') {
      throw std::runtime_error("expected ',' or ']' after vocab entry");
    }
  }

  SkipWhitespace(text, position);
  if (position != text.size()) {
    throw std::runtime_error("unexpected content after vocab array");
  }
  if (max_id < 0) {
    throw std::runtime_error("vocab file contains no entries");
  }

  vocab_size_ = max_id + 1;
  // 编码阶段执行贪心最长匹配；长度相同时按较小 ID 保持确定性
  std::sort(content_to_id_.begin(), content_to_id_.end(),
            [](const auto& left, const auto& right) {
              if (left.first.size() != right.first.size()) {
                return left.first.size() > right.first.size();
              }
              return left.second < right.second;
            });
}

std::vector<int> JsonVocabTokenizer::Encode(const std::string& text) const {
  // JSON 词表编码遵循模型输入约定：序列开头自动加入 BOS
  std::vector<int> tokens;
  tokens.push_back(bos_id_);

  size_t position = 0;
  while (position < text.size()) {
    bool matched = false;
    for (const auto& item : content_to_id_) {
      const std::string& content = item.first;
      if (!content.empty() && text.compare(position, content.size(), content) == 0) {
        tokens.push_back(item.second);
        position += content.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      // 参考格式按字节回退：每个无法匹配的输入字节产生一个 UNK
      tokens.push_back(unk_id_);
      ++position;
    }
  }
  return tokens;
}

std::string JsonVocabTokenizer::DecodeToken(int token) const {
  // 非法 ID 和稀疏词表中的空槽统一解码为未知 token
  if (token < 0 || static_cast<size_t>(token) >= id_to_entry_.size() ||
      id_to_entry_[token].content.empty()) {
    return "<unk>";
  }
  return id_to_entry_[token].content;
}

std::string JsonVocabTokenizer::Decode(const std::vector<int>& tokens) const {
  // 解码时保留传入的所有特殊 token，由上层决定是否过滤 BOS/EOS
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path) {
  // 路径有效时优先创建 JSON 分词器；格式错误同样安全回退到 ASCII
  if (!vocab_path.empty() && std::filesystem::is_regular_file(vocab_path)) {
    try {
      return std::make_unique<JsonVocabTokenizer>(vocab_path);
    } catch (const std::exception& error) {
      std::cerr << "警告：加载 JSON 词表失败 '" << vocab_path
                << "'，回退到 ASCII 分词器：" << error.what() << '\n';
    }
  }
  return std::make_unique<AsciiTokenizer>();
}

}  // namespace mini_llama
