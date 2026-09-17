#include <algorithm>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {
namespace {

// 读取 JSON 字符串键和值，并处理常见的反斜杠转义。
bool ParseJsonString(const std::string& text, size_t* position, std::string* value) {
  // 跳过键值周围的空白，再从双引号处开始解析。
  while (*position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*position]))) {
    ++*position;
  }
  if (*position >= text.size() || text[(*position)++] != '"') {
    return false;
  }
  value->clear();
  
  while (*position < text.size()) {
    char c = text[(*position)++];
    if (c == '"') {
      return true;
    }
    if (c != '\\') {
      value->push_back(c);
      continue;
    }
    if (*position >= text.size()) {
      return false;
    }
    c = text[(*position)++];
    if (c == 'n') {
      value->push_back('\n');
    } else if (c == 'r') {
      value->push_back('\r');
    } else if (c == 't') {
      value->push_back('\t');
    } else if (c == 'b') {
      value->push_back('\b');
    } else if (c == 'f') {
      value->push_back('\f');
    } else if (c == '"' || c == '\\' || c == '/') {
      value->push_back(c);
    } else if (c == 'u' && *position + 4 <= text.size()) {
      // 将四位十六进制 Unicode 转义编码为 UTF-8。
      unsigned codepoint = 0;
      for (int i = 0; i < 4; ++i) {
        const char digit = text[(*position)++];
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
      if (codepoint <= 0x7f) {
        value->push_back(static_cast<char>(codepoint));
      } else if (codepoint <= 0x7ff) {
        value->push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        value->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      } else {
        value->push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        value->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        value->push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
      }
    } else {
      return false;
    }
  }
  return false;
}

bool IsSpecial(const std::string& token) {
  // GPT 类词表通常用尖括号包围特殊 token；<0x..> 保留给字节 token。
  return token.size() >= 2 && token.front() == '<' && token.back() == '>' &&
         token.find("<0x") != 0;
}

// 先按空白、字母数字和标点切分，模拟 GPT-2 的预分词，避免依赖正则库。
std::vector<std::string> SplitText(const std::string& text) {
  std::vector<std::string> chunks;
  size_t position = 0;
  while (position < text.size()) {
    size_t start = position;
    if (std::isspace(static_cast<unsigned char>(text[position]))) {
      // 连续空白保留，仅把最后一个空格附着到后面的普通片段。
      while (position < text.size() &&
             std::isspace(static_cast<unsigned char>(text[position]))) {
        ++position;
      }
      if (position == text.size()) {
        chunks.emplace_back(text.substr(start));
        break;
      }
      if (position - start > 1) {
        chunks.emplace_back(text.substr(start, position - start - 1));
      }
      start = position - 1;  // GPT-2 会把一个空格附着到后续片段。
      position = start;
    }
    if (position >= text.size()) {
      break;
    }
    if (text[position] == ' ') {
      ++position;
    }

    // 将英文缩写（例如 "'re"）保持为一个 GPT-2 片段。
    if (text[position] == '\'' && position + 1 < text.size()) {
      const std::string rest = text.substr(position);
      bool contraction = rest.rfind("'s", 0) == 0 || rest.rfind("'t", 0) == 0 ||
                         rest.rfind("'re", 0) == 0 || rest.rfind("'ve", 0) == 0 ||
                         rest.rfind("'m", 0) == 0 || rest.rfind("'ll", 0) == 0 ||
                         rest.rfind("'d", 0) == 0;
      if (contraction) {
        size_t contraction_end = position + 1;
        while (contraction_end < text.size() &&
               !std::isspace(static_cast<unsigned char>(text[contraction_end]))) {
          ++contraction_end;
        }
        chunks.emplace_back(text.substr(start, contraction_end - start));
        position = contraction_end;
        continue;
      }
    }
    if (position >= text.size()) {
      chunks.emplace_back(text.substr(start));
      break;
    }
    const bool alpha = std::isalpha(static_cast<unsigned char>(text[position])) ||
                       static_cast<unsigned char>(text[position]) >= 0x80;
    const bool digit = std::isdigit(static_cast<unsigned char>(text[position]));
    if (alpha || digit) {
      // 字母、数字及非 ASCII 字节连续归入同一片段。
      ++position;
      while (position < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[position]);
        if (!(std::isalpha(c) || std::isdigit(c) || c >= 0x80)) {
          break;
        }
        ++position;
      }
    } else {
      // 非空白标点连续归入同一片段。
      ++position;
      while (position < text.size() &&
             !std::isspace(static_cast<unsigned char>(text[position])) &&
             !std::isalnum(static_cast<unsigned char>(text[position])) &&
             static_cast<unsigned char>(text[position]) < 0x80) {
        ++position;
      }
    }
    chunks.emplace_back(text.substr(start, position - start));
  }
  return chunks;
}

}  // namespace

void BpeTokenizer::BuildByteMappings() {
  // 建立 GPT-2 byte encoder：可打印字节保持原字符，其余字节映射到扩展 Unicode。
  b2u_.resize(256);
  u2b_.clear();
  std::vector<int> bytes;
  for (int c = '!'; c <= '~'; ++c) {
    bytes.push_back(c);
  }
  for (int c = 0xa1; c <= 0xac; ++c) {
    bytes.push_back(c);
  }
  for (int c = 0xae; c <= 0xff; ++c) {
    bytes.push_back(c);
  }
  std::vector<bool> direct(256, false);
  for (int c : bytes) {
    direct[c] = true;
  }

  int extra = 256;
  for (int c = 0; c < 256; ++c) {
    // 直接字符沿用原码点，不可直接表示的字节从码点 256 起顺序分配。
    const int codepoint = direct[c] ? c : extra++;
    std::string symbol;
    if (codepoint <= 0x7f) {
      symbol.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
      symbol.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
      symbol.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
      symbol.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
      symbol.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
      symbol.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    b2u_[c] = symbol;
    u2b_[symbol] = static_cast<uint8_t>(c);
  }
}

bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
  // 清空旧状态，使同一个对象可以安全地重新加载另一套词表。
  voacb_.clear();
  id_to_token_.clear();
  mrege_ranks_.clear();
  special_tokens_.clear();
  vocab_size_ = bos_id_ = eos_id_ = unk_id_ = -1;
  std::ifstream vocab_file(vocab_path), merges_file(merges_path),
      special_file(special_path);
  if (!vocab_file || !merges_file) {
    return false;
  }

  // vocab.json 是“字符串到整数 ID”的对象；扫描时同时构造反向数组。
  std::string json((std::istreambuf_iterator<char>(vocab_file)), {});
  size_t position = 0;
  while (position < json.size()) {
    // 依次读取 token 字符串和冒号后的数字 ID。
    while (position < json.size() && json[position] != '"' &&
           json[position] != '}') {
      ++position;
    }
    if (position >= json.size() || json[position] == '}') {
      break;
    }
    std::string token;
    if (!ParseJsonString(json, &position, &token)) {
      return false;
    }
    while (position < json.size() &&
           (std::isspace(static_cast<unsigned char>(json[position])) ||
            json[position] == ':')) {
      ++position;
    }
    size_t begin = position;
    while (position < json.size() &&
           std::isdigit(static_cast<unsigned char>(json[position]))) {
      ++position;
    }
    if (begin == position) {
      return false;
    }
    const int id = std::stoi(json.substr(begin, position - begin));
    voacb_[token] = id;
    if (id >= static_cast<int>(id_to_token_.size())) {
      id_to_token_.resize(id + 1);
    }
    id_to_token_[id] = token;
  }

  vocab_size_ = static_cast<int>(id_to_token_.size());
  BuildByteMappings();
  std::string line;
  int rank = 0;

  while (std::getline(merges_file, line)) {
    // merges.txt 每行是一个待合并的符号对，行号就是合并优先级。
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream stream(line);
    std::string left, right;
    if (stream >> left >> right) {
      mrege_ranks_[{left, right}] = rank++;
    }
  }
  if (special_file) {
    // special_tokens.json 保存特殊字符串及其 ID，编码时需要优先识别。
    std::string special_json((std::istreambuf_iterator<char>(special_file)), {});
    position = 0;

    while (position < special_json.size()) {
      // 特殊 token 文件同样按“字符串: ID”的对象格式扫描。
      if (special_json[position] != '"') {
        ++position;
        continue;
      }
      std::string token;
      if (!ParseJsonString(special_json, &position, &token)) {
        break;
      }
      while (position < special_json.size() &&
             (std::isspace(static_cast<unsigned char>(special_json[position])) ||
              special_json[position] == ':')) {
        ++position;
      }
      size_t begin = position;
      while (position < special_json.size() &&
             std::isdigit(static_cast<unsigned char>(special_json[position]))) {
        ++position;
      }
      if (begin == position) {
        continue;
      }
      special_tokens_.emplace_back(
          token, std::stoi(special_json.substr(begin, position - begin)));
    }
  }

  for (int id = 0; id < static_cast<int>(id_to_token_.size()); ++id) {
    // 从词表内容推断常见 BOS/EOS/UNK 别名，兼容不同模型命名习惯。
    if (id_to_token_[id] == "<|endoftext|>" || id_to_token_[id] == "<eos>" ||
        id_to_token_[id] == "</s>") {
      eos_id_ = id;
    }
    if (id_to_token_[id] == "<|beginoftext|>" || id_to_token_[id] == "<bos>" ||
        id_to_token_[id] == "<s>") {
      bos_id_ = id;
    }
    if (id_to_token_[id] == "<|unk|>" || id_to_token_[id] == "<unk>") {
      unk_id_ = id;
    }
    if (IsSpecial(id_to_token_[id])) {
      special_tokens_.emplace_back(id_to_token_[id], id);
    }
  }
  std::sort(
      special_tokens_.begin(), special_tokens_.end(),
      [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  return true;
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
  std::vector<int> result;
  size_t offset = 0;
  while (offset < text.size()) {
    // 特殊 token 必须在普通 BPE 前匹配，否则尖括号会被拆成普通字符。
    bool special = false;
    for (const auto& item : special_tokens_) {
      if (text.compare(offset, item.first.size(), item.first) == 0) {
        result.push_back(item.second);
        offset += item.first.size();
        special = true;
        break;
      }
    }
    if (special) {
      continue;
    }

    size_t end = offset + 1;
    // 找到下一个特殊 token 起点，普通文本只在该边界前参与 BPE。
    while (end < text.size()) {
      bool found_special = false;
      for (const auto& item : special_tokens_) {
        if (text.compare(end, item.first.size(), item.first) == 0) {
          found_special = true;
          break;
        }
      }
      if (found_special) {
        break;
      }
      ++end;
    }

    for (const std::string& chunk : SplitText(text.substr(offset, end - offset))) {
      // 每个输入字节先变成一个 GPT-2 符号，再反复应用优先级最高的相邻合并。
      std::vector<std::string> symbols;
      for (unsigned char byte : chunk) {
        symbols.push_back(b2u_[byte]);
      }

      while (symbols.size() > 1) {
        // 每轮查找 rank 最小的相邻对；找不到规则时合并过程结束。
        int best_rank = -1;
        size_t best = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
          auto it = mrege_ranks_.find({symbols[i], symbols[i + 1]});
          if (it != mrege_ranks_.end() &&
              (best_rank < 0 || it->second < best_rank)) {
            best_rank = it->second;
            best = i;
          }
        }
        if (best_rank < 0) {
          break;
        }
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + best + 1);
      }

      for (const std::string& symbol : symbols) {
        // 合并后的符号必须在词表中，否则使用模型定义的 UNK。
        auto it = voacb_.find(symbol);
        result.push_back(it == voacb_.end() ? unk_id_ : it->second);
      }
    }
    offset = end;
  }
  return result;
}

std::string BpeTokenizer::DecodeToken(int token) const {
  // 将词表中的 Unicode 映射符号逆变换为原始字节；普通 Unicode 原样保留。
  if (token < 0 || token >= static_cast<int>(id_to_token_.size())) {
    return {};
  }
  const std::string& value = id_to_token_[token];
  std::string result;

  for (size_t position = 0; position < value.size();) {
    // Unicode 映射符号长度最多为四字节，按最长片段尝试逆查。
    bool matched = false;
    for (size_t length = std::min<size_t>(4, value.size() - position); length > 0;
         --length) {
      auto it = u2b_.find(value.substr(position, length));
      if (it != u2b_.end()) {
        result.push_back(static_cast<char>(it->second));
        position += length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      result += value.substr(position);
      break;
    }
  }
  return result;
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
  // 按顺序拼接每个 token 的字节结果。
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

}  // namespace mini_llama
