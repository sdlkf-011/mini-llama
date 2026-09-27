#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

namespace {

// 本文件实现 ASCII 分词器和数组对象格式的 JSON 词表分词器。
// 这里使用受限 JSON 读取器，覆盖词表所需的字符串、整数和布尔字段，
// 从而避免给核心库引入第三方 JSON 依赖。
class JsonReader {
 public:
  explicit JsonReader(std::string text) : text_(std::move(text)) {}

  struct ParsedEntry {
    int id = 0;
    std::string content;
    bool special = false;
  };

  bool ParseEntry(ParsedEntry* entry) {
    // 逐字段读取一个词表对象，未知字段通过 SkipValue 跳过以保持兼容性。
    SkipSpace();
    if (position_ >= text_.size() || text_[position_++] != '{') {
      return false;
    }

    while (true) {
      SkipSpace();
      if (position_ < text_.size() && text_[position_] == '}') {
        ++position_;
        return true;
      }
      std::string key;
      if (!ParseString(&key)) {
        return false;
      }
      SkipSpace();
      if (position_ >= text_.size() || text_[position_++] != ':') {
        return false;
      }
      SkipSpace();
      if (key == "id") {
        int value = 0;
        if (!ParseInt(&value)) {
          return false;
        }
        entry->id = value;
      } else if (key == "content") {
        if (!ParseString(&entry->content)) {
          return false;
        }
      } else if (key == "special") {
        if (!ParseBool(&entry->special)) {
          return false;
        }
      } else if (!SkipValue()) {
        return false;
      }
      SkipSpace();
      if (position_ >= text_.size()) {
        return false;
      }
      if (text_[position_] == ',') {
        ++position_;
        continue;
      }
      if (text_[position_] == '}') {
        ++position_;
        return true;
      }
      return false;
    }
  }

  bool BeginArray() {
    // 词表顶层必须是 JSON 数组。
    SkipSpace();
    return position_ < text_.size() && text_[position_++] == '[';
  }

  bool EndArray() {
    // 确认所有词条后存在数组结束符。
    SkipSpace();
    if (position_ < text_.size() && text_[position_] == ']') {
      ++position_;
      return true;
    }
    return false;
  }

  bool HasValue() {
    SkipSpace();
    return position_ < text_.size() && text_[position_] != ']';
  }

  bool ConsumeComma() {
    // 词条之间允许空白，但必须由逗号分隔。
    SkipSpace();
    if (position_ < text_.size() && text_[position_] == ',') {
      ++position_;
      return true;
    }
    return false;
  }

 private:
  void SkipSpace() {
    // 所有解析入口统一忽略 JSON 空白字符。
    while (position_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[position_]))) {
      ++position_;
    }
  }

  bool ParseString(std::string* value) {
    // 解析 JSON 转义，并将 \uXXXX 编码转换为 UTF-8 字节序列。
    SkipSpace();
    if (position_ >= text_.size() || text_[position_++] != '"') {
      return false;
    }
    value->clear();

    while (position_ < text_.size()) {
      const char c = text_[position_++];
      if (c == '"') {
        return true;
      }
      if (c != '\\') {
        value->push_back(c);
        continue;
      }
      if (position_ >= text_.size()) {
        return false;
      }

      const char escape = text_[position_++];
      switch (escape) {
        case '"':
          value->push_back('"');
          break;
        case '\\':
          value->push_back('\\');
          break;
        case '/':
          value->push_back('/');
          break;
        case 'b':
          value->push_back('\b');
          break;
        case 'f':
          value->push_back('\f');
          break;
        case 'n':
          value->push_back('\n');
          break;
        case 'r':
          value->push_back('\r');
          break;
        case 't':
          value->push_back('\t');
          break;
        case 'u': {
          // 词表常含 ASCII 转义；这里将 Unicode 转义（含代理对）转换为 UTF-8。
          if (position_ + 4 > text_.size()) {
            return false;
          }

          unsigned codepoint = 0;
          for (int i = 0; i < 4; ++i) {
            const char digit = text_[position_++];
            codepoint <<= 4;
            if (digit >= '0' && digit <= '9') {
              codepoint += digit - '0';
            } else if (digit >= 'a' && digit <= 'f') {
              codepoint += digit - 'a' + 10;
            } else if (digit >= 'A' && digit <= 'F') {
              codepoint += digit - 'A' + 10;
            } else {
              return false;
            }
          }

          if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
              position_ + 6 <= text_.size() && text_[position_] == '\\' &&
              text_[position_ + 1] == 'u') {
            position_ += 2;
            unsigned low = 0;
            for (int i = 0; i < 4; ++i) {
              const char digit = text_[position_++];
              low <<= 4;
              if (digit >= '0' && digit <= '9') {
                low += digit - '0';
              } else if (digit >= 'a' && digit <= 'f') {
                low += digit - 'a' + 10;
              } else if (digit >= 'A' && digit <= 'F') {
                low += digit - 'A' + 10;
              } else {
                return false;
              }
            }
            if (low < 0xdc00 || low > 0xdfff) {
              return false;
            }
            codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
          }
          if (codepoint <= 0x7f) {
            value->push_back(static_cast<char>(codepoint));
          } else if (codepoint <= 0x7ff) {
            value->push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            value->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
          } else if (codepoint <= 0xffff) {
            value->push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            value->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            value->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
          } else {
            value->push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            value->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            value->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            value->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
          }
          break;
        }
        default:
          return false;
      }
    }
    return false;
  }

  bool ParseInt(int* value) {
    // ID 只接受十进制整数，并捕获越界转换异常。
    SkipSpace();
    const size_t start = position_;
    if (position_ < text_.size() && text_[position_] == '-') {
      ++position_;
    }

    while (position_ < text_.size() &&
           std::isdigit(static_cast<unsigned char>(text_[position_]))) {
      ++position_;
    }

    if (start == position_ || (text_[start] == '-' && start + 1 == position_)) {
      return false;
    }

    try {
      *value = std::stoi(text_.substr(start, position_ - start));
    } catch (...) {
      return false;
    }
    return true;
  }

  bool ParseBool(bool* value) {
    // special 字段只接受标准 JSON 布尔字面量。
    SkipSpace();
    if (text_.compare(position_, 4, "true") == 0) {
      position_ += 4;
      *value = true;
      return true;
    }
    if (text_.compare(position_, 5, "false") == 0) {
      position_ += 5;
      *value = false;
      return true;
    }
    return false;
  }

  bool SkipValue() {
    // 未知字段无需保存，但必须完整越过其值以继续解析当前对象。
    SkipSpace();
    if (position_ >= text_.size()) {
      return false;
    }
    if (text_[position_] == '"') {
      std::string ignored;
      return ParseString(&ignored);
    }
    if (text_[position_] == '{') {
      ++position_;
      int depth = 1;
      while (position_ < text_.size() && depth) {
        if (text_[position_] == '{') {
          ++depth;
        } else if (text_[position_] == '}') {
          --depth;
        }
        ++position_;
      }
      return depth == 0;
    }

    while (position_ < text_.size() && text_[position_] != ',' &&
           text_[position_] != '}') {
      ++position_;
    }
    return true;
  }

  std::string text_;
  size_t position_ = 0;
};

size_t NextUtf8Character(const std::string& text, size_t offset) {
  // 未匹配词表时按一个 UTF-8 字符前进，避免把多字节字符拆成多个 UNK。
  if (offset >= text.size()) {
    return offset;
  }
  const unsigned char c = static_cast<unsigned char>(text[offset]);
  size_t length = c < 0x80 ? 1 : (c < 0xe0 ? 2 : (c < 0xf0 ? 3 : 4));
  if (offset + length > text.size()) {
    return offset + 1;
  }
  return offset + length;
}

}  // namespace

JsonVocabTokenizer::JsonVocabTokenizer(const std::string& vocab_path) {
  // 加载数组词表，同时建立 id->内容和内容->id 两个方向的索引。
  std::ifstream input(vocab_path);
  if (!input) {
    return;
  }

  const std::string contents((std::istreambuf_iterator<char>(input)), {});
  JsonReader reader(contents);
  if (!reader.BeginArray()) {
    return;
  }

  while (reader.HasValue()) {
    // 每个词条按 ID 放入数组；稀疏 ID 会保留空槽，确保索引不偏移。
    JsonReader::ParsedEntry parsed;
    if (!reader.ParseEntry(&parsed)) {
      id_to_entry_.clear();
      content_to_id_.clear();
      return;
    }
    if (parsed.id < 0) {
      continue;
    }
    if (parsed.id >= static_cast<int>(id_to_entry_.size())) {
      id_to_entry_.resize(parsed.id + 1);
    }
    id_to_entry_[parsed.id] = {parsed.id, parsed.content, parsed.special};
    content_to_id_.emplace_back(parsed.content, parsed.id);
    if (parsed.content == "<bos>" || parsed.content == "<s>" ||
        parsed.content == "<|begin_of_text|>") {
      bos_id_ = parsed.id;
    }
    if (parsed.content == "<eos>" || parsed.content == "</s>" ||
        parsed.content == "<|end_of_text|>") {
      eos_id_ = parsed.id;
    }
    if (parsed.content == "<unk>" || parsed.content == "<UNK>") {
      unk_id_ = parsed.id;
    }
    reader.ConsumeComma();
  }

  if (!reader.EndArray()) {
    id_to_entry_.clear();
    content_to_id_.clear();
    return;
  }
  vocab_size_ = static_cast<int>(id_to_entry_.size());
  std::sort(content_to_id_.begin(), content_to_id_.end(),
            // 编码阶段采用贪心最长匹配，因此预先按内容长度降序排列。
            [](const auto& left, const auto& right) {
              if (left.first.size() != right.first.size()) {
                return left.first.size() > right.first.size();
              }
              return left.second < right.second;
            });
}

std::vector<int> JsonVocabTokenizer::Encode(const std::string& text) const {
  std::vector<int> result;
  size_t offset = 0;
  while (offset < text.size()) {
    // 内容索引已按长度降序排列，多字符词条优先于其单字符前缀。
    bool matched = false;
    for (const auto& item : content_to_id_) {
      if (!item.first.empty() &&
          text.compare(offset, item.first.size(), item.first) == 0) {
        result.push_back(item.second);
        offset += item.first.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      // 没有任何词条可匹配时，为当前 Unicode 字符输出一个 UNK。
      result.push_back(unk_id_);
      offset = NextUtf8Character(text, offset);
    }
  }
  return result;
}

std::string JsonVocabTokenizer::DecodeToken(int token) const {
  // 越界 token 不抛异常，直接返回空串以便调用方安全处理损坏输入。
  if (token < 0 || token >= static_cast<int>(id_to_entry_.size())) {
    return {};
  }
  return id_to_entry_[token].content;
}

std::string JsonVocabTokenizer::Decode(const std::vector<int>& tokens) const {
  // 逐个调用 DecodeToken，拼接完整文本。
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

}  // namespace mini_llama
