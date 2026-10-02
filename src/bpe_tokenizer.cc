#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {
//------------------------------------------
// UTF-8 辅助函数
//------------------------------------------

// 将单个 Unicode 码点编码为 UTF-8 字节序列
static std::string CodepointToUtf8(char32_t codepoint) {
  std::string result;
  if (codepoint <= 0x7F) {
    result += static_cast<char>(codepoint);
  } else if (codepoint <= 0x7FF) {
    result += static_cast<char>(0xC0 | (codepoint >> 6));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else if (codepoint <= 0xFFFF) {
    result += static_cast<char>(0xE0 | (codepoint >> 12));
    result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  } else {
    result += static_cast<char>(0xF0 | (codepoint >> 18));
    result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
    result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
    result += static_cast<char>(0x80 | (codepoint & 0x3F));
  }
  return result;
}

// 从指定位置读取一个 UTF-8 字符，并将位置推进到下一个字符
// 遇到非法首字节或不完整字符时跳过一个字节并返回空串
static std::string NextUtf8Codepoint(const std::string& text, size_t& position) {
  if (position >= text.size()) {
    return "";
  }

  const unsigned char first = text[position];
  size_t length = 1;
  if ((first & 0x80) == 0) {
    length = 1;
  } else if ((first & 0xE0) == 0xC0) {
    length = 2;
  } else if ((first & 0xF0) == 0xE0) {
    length = 3;
  } else if ((first & 0xF8) == 0xF0) {
    length = 4;
  } else {
    ++position;
    return "";
  }

  if (position + length > text.size()) {
    ++position;
    return "";
  }

  std::string result = text.substr(position, length);
  position += length;
  return result;
}

//------------------------------------------
// GPT-2 bytes_to_unicode 字节映射
//------------------------------------------

// 构造 GPT-2 的 byte-to-unicode 表，使任意原始字节都能表示为
// 可安全存入词表的 UTF-8 字符串
static std::vector<std::string> BuildBytesToUnicode() {
  std::vector<std::string> mapping(256);

  // 可打印 ASCII 和 Latin-1 字符直接使用原码点
  std::vector<int> bytes;
  for (int character = '!'; character <= '~'; ++character) {
    bytes.push_back(character);
  }
  for (int character = 0xA1; character <= 0xAC; ++character) {
    bytes.push_back(character);
  }
  for (int character = 0xAE; character <= 0xFF; ++character) {
    bytes.push_back(character);
  }

  // 其余字节依次映射到 256 以上的 Unicode 码点，避免与已有字符冲突
  std::vector<int> codepoints = bytes;
  int offset = 0;
  for (int byte = 0; byte < 256; ++byte) {
    if (std::find(bytes.begin(), bytes.end(), byte) == bytes.end()) {
      bytes.push_back(byte);
      codepoints.push_back(256 + offset);
      ++offset;
    }
  }

  for (size_t index = 0; index < bytes.size(); ++index) {
    mapping[bytes[index]] =
        CodepointToUtf8(static_cast<char32_t>(codepoints[index]));
  }
  return mapping;
}

//------------------------------------------
// 简单 JSON 解析辅助函数
//------------------------------------------

// 从当前位置解析一个 JSON 整数，并更新读取位置
static bool ParseJsonInt(const std::string& text, size_t& position, int& value) {
  while (position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  if (position >= text.size()) {
    return false;
  }

  const size_t start = position;
  if (text[position] == '-') {
    ++position;
  }
  while (position < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  if (start == position) {
    return false;
  }

  try {
    value = std::stoi(text.substr(start, position - start));
    return true;
  } catch (...) {
    return false;
  }
}

// 从当前位置解析一个 JSON 字符串，支持常见转义和 \uXXXX Unicode 转义
static bool ParseJsonString(const std::string& text, size_t& position,
                            std::string& value) {
  while (position < text.size() &&
         std::isspace(static_cast<unsigned char>(text[position]))) {
    ++position;
  }
  if (position >= text.size() || text[position] != '"') {
    return false;
  }

  ++position;
  value.clear();
  while (position < text.size() && text[position] != '"') {
    if (text[position] == '\\' && position + 1 < text.size()) {
      const char escaped = text[position + 1];
      switch (escaped) {
        case '"':
          value += '"';
          break;
        case '\\':
          value += '\\';
          break;
        case '/':
          value += '/';
          break;
        case 'b':
          value += '\b';
          break;
        case 'f':
          value += '\f';
          break;
        case 'n':
          value += '\n';
          break;
        case 'r':
          value += '\r';
          break;
        case 't':
          value += '\t';
          break;
        case 'u': {
          if (position + 5 >= text.size()) {
            return false;
          }
          const std::string hex = text.substr(position + 2, 4);
          try {
            const char32_t codepoint =
                static_cast<char32_t>(std::stoul(hex, nullptr, 16));
            value += CodepointToUtf8(codepoint);
          } catch (...) {
            return false;
          }
          position += 4;
          break;
        }
        default:
          return false;
      }
      position += 2;
    } else {
      value += text[position];
      ++position;
    }
  }

  if (position >= text.size() || text[position] != '"') {
    return false;
  }
  ++position;
  return true;
}

//------------------------------------------
// BPE 分词器实现
//------------------------------------------

// 同时建立“字节到 Unicode 符号”和“Unicode 符号到字节”的双向映射
void BpeTokenizer::BuildByteMappings() {
  b2u_ = BuildBytesToUnicode();
  u2b_.clear();
  for (int byte = 0; byte < 256; ++byte) {
    u2b_[b2u_[byte]] = static_cast<uint8_t>(byte);
  }
}

bool BpeTokenizer::Load(const std::string& vocab_path,
                        const std::string& merges_path,
                        const std::string& special_path) {
  // 清空旧模型状态，确保同一实例可以安全地重复加载不同词表
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  bos_id_ = -1;
  eos_id_ = -1;
  unk_id_ = -1;
  BuildByteMappings();

  // 加载 vocab.json，格式为 {"token": id, ...}
  {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open vocab: " << vocab_path << std::endl;
      return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();

    // 确认顶层是 JSON 对象，再逐项解析 token 和对应 ID
    size_t position = 0;
    while (position < content.size() &&
           std::isspace(static_cast<unsigned char>(content[position]))) {
      ++position;
    }
    if (position >= content.size() || content[position] != '{') {
      return false;
    }
    ++position;

    while (position < content.size()) {
      while (position < content.size() &&
             std::isspace(static_cast<unsigned char>(content[position]))) {
        ++position;
      }
      if (position < content.size() && content[position] == '}') {
        break;
      }

      std::string key;
      if (!ParseJsonString(content, position, key)) {
        return false;
      }
      while (position < content.size() &&
             std::isspace(static_cast<unsigned char>(content[position]))) {
        ++position;
      }
      if (position >= content.size() || content[position] != ':') {
        return false;
      }
      ++position;

      int token_id = 0;
      if (!ParseJsonInt(content, position, token_id) || token_id < 0) {
        return false;
      }
      vocab_[key] = token_id;

      while (position < content.size() &&
             std::isspace(static_cast<unsigned char>(content[position]))) {
        ++position;
      }
      if (position < content.size() && content[position] == ',') {
        ++position;
      } else if (position < content.size() && content[position] != '}') {
        return false;
      }
    }

    if (vocab_.empty()) {
      std::cerr << "Vocab is empty" << std::endl;
      return false;
    }

    // 根据最大 ID 构造反向映射，支持通过 ID 查找 token 字符串
    int max_id = 0;
    // 从词表识别 <|...|> 特殊 token，并兼容常见控制 token 名称
    for (const auto& item : vocab_) {
      max_id = std::max(max_id, item.second);
    }
    id_to_token_.resize(max_id + 1);
    for (const auto& item : vocab_) {
      id_to_token_[item.second] = item.first;
    }

    for (const auto& item : vocab_) {
      const std::string& token = item.first;
      if (token.size() >= 4 && token.front() == '<' && token[1] == '|' &&
          token[token.size() - 2] == '|' && token.back() == '>') {
        special_tokens_.push_back(item);
      }
      if (token == "<s>" || token == "<bos>" || token == "<|beginoftext|>") {
        bos_id_ = item.second;
      } else if (token == "</s>" || token == "<eos>" || token == "<|endoftext|>") {
        eos_id_ = item.second;
      } else if (token == "<unk>" || token == "<|unk|>") {
        unk_id_ = item.second;
      }
    }
  }

  // 加载 merges.txt；每条有效规则的出现顺序就是 BPE 合并优先级
  {
    std::ifstream file(merges_path);
    if (!file.is_open()) {
      std::cerr << "Failed to open merges: " << merges_path << std::endl;
      return false;
    }

    std::string line;
    int rank = 0;
    while (std::getline(file, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      if (line.empty() || line.front() == '#') {
        continue;
      }

      std::stringstream stream(line);
      std::string left;
      std::string right;
      if (stream >> left >> right) {
        merge_ranks_[{left, right}] = rank++;
      }
    }
  }

  // 加载特殊 token 配置。既支持 bos_id/eos_id/unk_id，
  // 也兼容 {"特殊token": id} 形式的映射
  {
    std::ifstream file(special_path);
    if (file.is_open()) {
      std::stringstream buffer;
      buffer << file.rdbuf();
      const std::string content = buffer.str();

      size_t position = 0;
      while (position < content.size()) {
        while (position < content.size() && content[position] != '"') {
          ++position;
        }
        if (position >= content.size()) {
          break;
        }

        std::string key;
        if (!ParseJsonString(content, position, key)) {
          return false;
        }
        while (position < content.size() &&
               std::isspace(static_cast<unsigned char>(content[position]))) {
          ++position;
        }
        if (position >= content.size() || content[position] != ':') {
          return false;
        }
        ++position;

        int token_id = 0;
        if (!ParseJsonInt(content, position, token_id)) {
          return false;
        }
        if (key == "bos_id") {
          bos_id_ = token_id;
        } else if (key == "eos_id") {
          eos_id_ = token_id;
        } else if (key == "unk_id") {
          unk_id_ = token_id;
        } else {
          const auto vocab_it = vocab_.find(key);
          if (vocab_it != vocab_.end() && vocab_it->second == token_id) {
            special_tokens_.emplace_back(key, token_id);
          }
        }
      }
    }
  }

  // 特殊 token 按长度降序排列，编码时优先执行最长匹配；随后去重
  std::sort(special_tokens_.begin(), special_tokens_.end(),
            [](const auto& left, const auto& right) {
              return left.first.size() > right.first.size();
            });
  special_tokens_.erase(std::unique(special_tokens_.begin(), special_tokens_.end()),
                        special_tokens_.end());

  std::cout << "BPE tokenizer loaded: vocab=" << vocab_.size()
            << " merges=" << merge_ranks_.size() << " bos=" << bos_id_
            << " eos=" << eos_id_ << " unk=" << unk_id_ << std::endl;
  return true;
}

std::vector<int> BpeTokenizer::Encode(const std::string& text) const {
  if (text.empty()) {
    return {};
  }

  std::vector<int> result;
  size_t position = 0;

  while (position < text.size()) {
    // 先匹配当前位置的特殊 token，防止其被普通 BPE 规则拆开
    bool matched = false;
    for (const auto& special_token : special_tokens_) {
      const std::string& token = special_token.first;
      if (position + token.size() <= text.size() &&
          std::memcmp(text.data() + position, token.data(), token.size()) == 0) {
        result.push_back(special_token.second);
        position += token.size();
        matched = true;
        break;
      }
    }
    if (matched) {
      continue;
    }

    // 普通文本只编码到下一个特殊 token 的起始位置
    size_t end = text.size();
    for (const auto& special_token : special_tokens_) {
      const size_t found = text.find(special_token.first, position);
      if (found != std::string::npos && found < end) {
        end = found;
      }
    }

    const std::string segment = text.substr(position, end - position);
    position = end;

    // 将原始 UTF-8 字节逐个转换成 GPT-2 使用的 Unicode 符号
    std::vector<std::string> word;
    for (const unsigned char byte : segment) {
      word.push_back(b2u_[byte]);
    }

    // 每轮选择 rank 最小的相邻符号对，直到没有可用合并规则
    while (word.size() > 1) {
      int best_rank = std::numeric_limits<int>::max();
      size_t best_index = word.size();

      for (size_t index = 0; index + 1 < word.size(); ++index) {
        const auto rank = merge_ranks_.find({word[index], word[index + 1]});
        if (rank != merge_ranks_.end() && rank->second < best_rank) {
          best_rank = rank->second;
          best_index = index;
        }
      }

      if (best_index >= word.size()) {
        break;
      }

      word[best_index] += word[best_index + 1];
      word.erase(word.begin() + best_index + 1);
    }

    // 将最终符号映射到词表 ID；缺失符号回退到 UNK
    for (const std::string& symbol : word) {
      const auto token = vocab_.find(symbol);
      result.push_back(token == vocab_.end() ? unk_id_ : token->second);
    }
  }

  return result;
}

std::string BpeTokenizer::DecodeToken(int token) const {
  // 单 token 解码仅返回词表中的内部字符串，不执行 byte-to-unicode 逆映射
  if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
    return id_to_token_[token];
  }
  return "";
}

std::string BpeTokenizer::Decode(const std::vector<int>& tokens) const {
  // 先拼接所有 token 的内部字符串，避免合并 token 边界影响字符扫描
  std::string text;
  for (const int token : tokens) {
    if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
      text += id_to_token_[token];
    }
  }

  // 按 UTF-8 字符扫描，并通过逆映射恢复原始输入字节
  std::string result;
  size_t position = 0;
  while (position < text.size()) {
    const size_t start = position;
    const std::string codepoint = NextUtf8Codepoint(text, position);
    if (codepoint.empty()) {
      if (position == start) {
        ++position;
      }
      continue;
    }

    const auto byte = u2b_.find(codepoint);
    if (byte != u2b_.end()) {
      result += static_cast<char>(byte->second);
    } else {
      result += codepoint;
    }
  }

  return result;
}

//------------------------------------------
// BPE 分词器工厂
//------------------------------------------

// 仅在三个配置文件成功加载后返回可用分词器，否则返回空指针
std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path) {
  auto tokenizer = std::make_unique<BpeTokenizer>();
  if (tokenizer->Load(vocab_path, merges_path, special_path)) {
    return tokenizer;
  }
  return nullptr;
}

}  // namespace mini_llama
