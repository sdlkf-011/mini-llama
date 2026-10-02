#include "mini_llama/gguf_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>

#include "mini_llama/gguf.h"

namespace mini_llama {
namespace {

// 从当前位置读取一个 UTF-8 码点；非法或不完整序列跳过一个字节
std::string NextUtf8Codepoint(const std::string& text, size_t* position) {
  if (*position >= text.size()) {
    return {};
  }

  const unsigned char first = text[*position];
  size_t length = 1;
  if ((first & 0x80) == 0) {
    length = 1;
  } else if ((first & 0xe0) == 0xc0) {
    length = 2;
  } else if ((first & 0xf0) == 0xe0) {
    length = 3;
  } else if ((first & 0xf8) == 0xf0) {
    length = 4;
  } else {
    ++(*position);
    return {};
  }
  if (*position + length > text.size()) {
    ++(*position);
    return {};
  }

  std::string codepoint = text.substr(*position, length);
  *position += length;
  return codepoint;
}

bool IsSpecial(const std::string& token) {
  // 尖括号 token 视为特殊字符串，十六进制字节 token 除外
  return token.size() >= 2 && token.front() == '<' && token.back() == '>' &&
         token.find("<0x") != 0;
}

std::vector<std::string> SplitText(const std::string& text) {
  // 使用轻量规则完成预分词，降低 GGUF tokenizer 的运行时依赖
  std::vector<std::string> chunks;
  size_t position = 0;

  while (position < text.size()) {
    size_t start = position;
    if (std::isspace(static_cast<unsigned char>(text[position]))) {
      // 保留连续空白，并将末尾单个空格附着到后续片段
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
      start = position - 1;
      position = start;
    }
    if (position >= text.size()) {
      break;
    }
    if (text[position] == ' ') {
      ++position;
    }
    if (position >= text.size()) {
      chunks.emplace_back(text.substr(start));
      break;
    }
    const unsigned char current = static_cast<unsigned char>(text[position]);
    if (std::isalnum(current) || current >= 0x80) {
      // 字母数字和 UTF-8 非 ASCII 字节作为普通文本连续收集
      ++position;
      while (position < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[position]);
        if (!(std::isalnum(c) || c >= 0x80)) {
          break;
        }
        ++position;
      }
    } else {
      // 连续标点独立成段，避免跨类型产生不符合预分词规则的合并
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

void GgufTokenizer::BuildByteMappings() {
  // 构造 GPT-2 风格的字节到 Unicode 映射及其逆映射
  b2u_.resize(256);
  u2b_.clear();
  std::vector<bool> direct(256, false);
  for (int c = '!'; c <= '~'; ++c) {
    direct[c] = true;
  }
  for (int c = 0xa1; c <= 0xac; ++c) {
    direct[c] = true;
  }
  for (int c = 0xae; c <= 0xff; ++c) {
    direct[c] = true;
  }

  int extra = 256;
  for (int c = 0; c < 256; ++c) {
    // 为全部 256 个字节生成唯一且可逆的 Unicode 表示
    int codepoint = direct[c] ? c : extra++;
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

bool GgufTokenizer::Load(const std::string& gguf_path) {
  // 加载新模型前清理旧元数据，确保失败或重载不会混入旧词表内容
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  bos_id_ = eos_id_ = unk_id_ = -1;
  // 只扫描 GGUF 元数据，不把可能数 GB 的张量数据读入内存
  std::vector<std::string> tokens;
  std::vector<std::string> merges;
  if (!GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.tokens", tokens) ||
      tokens.empty()) {
    return false;
  }
  // merges 对 unigram/字符词表可以不存在；BPE 模型则按元数据顺序使用
  GgufGetMetadataStringArray(gguf_path, "tokenizer.ggml.merges", merges);

  auto read_token_id = [&](const std::string& key, int* destination) {
    int64_t value = 0;
    if (!GgufGetMetadataInt(gguf_path, key, value)) {
      return false;
    }
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
      return false;
    }
    *destination = static_cast<int>(value);
    return true;
  };
  read_token_id("tokenizer.ggml.bos_token_id", &bos_id_);
  read_token_id("tokenizer.ggml.eos_token_id", &eos_id_);
  if (!read_token_id("tokenizer.ggml.unknown_token_id", &unk_id_)) {
    read_token_id("tokenizer.ggml.unk_token_id", &unk_id_);
  }

  // 建立 token 内容到 ID 的索引，并解析合并规则的优先级
  id_to_token_ = std::move(tokens);
  vocab_.clear();
  for (int id = 0; id < static_cast<int>(id_to_token_.size()); ++id) {
    vocab_[id_to_token_[id]] = id;
  }
  BuildByteMappings();
  merge_ranks_.clear();
  int rank = 0;
  for (const std::string& merge : merges) {
    std::istringstream stream(merge);
    std::string left, right;
    if (stream >> left >> right) {
      merge_ranks_[{left, right}] = rank++;
    }
  }
  special_tokens_.clear();

  for (int id = 0; id < static_cast<int>(id_to_token_.size()); ++id) {
    // 同时识别特殊 token 和常见控制 token 别名
    const std::string& token = id_to_token_[id];
    if (IsSpecial(token)) {
      special_tokens_.emplace_back(token, id);
    }
    if (unk_id_ < 0 && (token == "<unk>" || token == "<UNK>")) {
      unk_id_ = id;
    }
    if (bos_id_ < 0 && (token == "<bos>" || token == "<s>")) {
      bos_id_ = id;
    }
    if (eos_id_ < 0 && (token == "<eos>" || token == "</s>")) {
      eos_id_ = id;
    }
  }
  std::sort(
      special_tokens_.begin(), special_tokens_.end(),
      [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  return true;
}

std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  if (id_to_token_.empty() || b2u_.size() != 256) {
    return {};
  }
  std::vector<int> result;
  size_t offset = 0;
  while (offset < text.size()) {
    // GGUF 与独立 BPE 词表一样，必须先匹配特殊字符串
    bool matched_special = false;
    for (const auto& special : special_tokens_) {
      if (text.compare(offset, special.first.size(), special.first) == 0) {
        result.push_back(special.second);
        offset += special.first.size();
        matched_special = true;
        break;
      }
    }
    if (matched_special) {
      continue;
    }

    size_t end = offset + 1;
    // 普通文本段截止到下一个特殊 token，避免特殊内容参与 BPE
    while (end < text.size()) {
      bool special = false;
      for (const auto& item : special_tokens_) {
        if (text.compare(end, item.first.size(), item.first) == 0) {
          special = true;
          break;
        }
      }
      if (special) {
        break;
      }
      ++end;
    }

    for (const std::string& chunk : SplitText(text.substr(offset, end - offset))) {
      // 对预分词片段反复执行最高优先级合并，直到不存在可合并相邻符号
      std::vector<std::string> symbols;
      for (unsigned char byte : chunk) {
        symbols.push_back(b2u_[byte]);
      }

      while (symbols.size() > 1) {
        // 每轮仅合并 rank 最小的相邻符号对
        int best_rank = -1;
        size_t best = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
          auto it = merge_ranks_.find({symbols[i], symbols[i + 1]});
          if (it != merge_ranks_.end() &&
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
        // GGUF 词表不包含最终符号时回退到 UNK ID
        auto it = vocab_.find(symbol);
        result.push_back(it == vocab_.end() ? unk_id_ : it->second);
      }
    }
    offset = end;
  }
  return result;
}

std::string GgufTokenizer::DecodeToken(int token) const {
  if (token < 0 || token >= static_cast<int>(id_to_token_.size())) {
    return {};
  }
  return Decode({token});
}

std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  // 先拼接全部 token 的内部字符串，再整体执行 byte-to-unicode 逆映射。
  // 这样跨 token 的原始 UTF-8 字节会在最终字符串中保持正确顺序
  std::string text;
  for (int token : tokens) {
    if (token >= 0 && token < static_cast<int>(id_to_token_.size())) {
      text += id_to_token_[token];
    }
  }

  std::string result;
  size_t position = 0;
  while (position < text.size()) {
    const std::string codepoint = NextUtf8Codepoint(text, &position);
    if (codepoint.empty()) {
      continue;
    }
    const auto byte = u2b_.find(codepoint);
    if (byte != u2b_.end()) {
      result.push_back(static_cast<char>(byte->second));
    } else {
      result += codepoint;
    }
  }
  return result;
}

std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path) {
  // GGUF 校验或元数据解析失败时返回 nullptr
  auto tokenizer = std::make_unique<GgufTokenizer>();
  if (!tokenizer->Load(gguf_path)) {
    return nullptr;
  }
  return tokenizer;
}

}  // namespace mini_llama
