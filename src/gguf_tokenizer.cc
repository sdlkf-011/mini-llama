#include "mini_llama/gguf_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

namespace mini_llama {
namespace {

// GGUF 文件采用小端二进制布局，此读取器集中处理边界检查和游标移动。
class Reader {
 public:
  explicit Reader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

  size_t remaining() const { return bytes_.size() - position_; }

  bool Read(void* destination, size_t size) {
    // 每次读取前检查剩余长度，防止损坏文件造成越界访问。
    if (size > remaining()) {
      return false;
    }
    std::memcpy(destination, bytes_.data() + position_, size);
    position_ += size;
    return true;
  }

  template <typename T>
  bool Scalar(T* value) {
    return Read(value, sizeof(T));
  }

  bool String(std::string* value) {
    // GGUF 字符串由 uint64 长度和紧随其后的原始字节组成。
    uint64_t length = 0;
    if (!Scalar(&length) || length > remaining()) {
      return false;
    }
    value->assign(reinterpret_cast<const char*>(bytes_.data() + position_),
                  static_cast<size_t>(length));
    position_ += static_cast<size_t>(length);
    return true;
  }

  bool Skip(size_t size) {
    if (size > remaining()) {
      return false;
    }
    position_ += size;
    return true;
  }

  bool ArrayHeader(uint32_t* type, uint64_t* count) {
    // 数组头记录元素类型和元素数量。
    return Scalar(type) && Scalar(count);
  }

 private:
  const std::vector<uint8_t>& bytes_;
  size_t position_ = 0;
};

enum ValueType : uint32_t {
  kUint8 = 0,
  kInt8 = 1,
  kUint16 = 2,
  kInt16 = 3,
  kUint32 = 4,
  kInt32 = 5,
  kFloat32 = 6,
  kBool = 7,
  kString = 8,
  kArray = 9,
  kUint64 = 10,
  kInt64 = 11,
  kFloat64 = 12
};

// 读取 GGUF 元数据中的整数，并统一转换为接口所需的 int。
bool ReadInteger(Reader* reader, uint32_t type, int* value) {
  switch (type) {
    case kUint8: {
      uint8_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = v;
      return true;
    }
    case kInt8: {
      int8_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = v;
      return true;
    }
    case kUint16: {
      uint16_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = static_cast<int>(v);
      return true;
    }
    case kInt16: {
      int16_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = v;
      return true;
    }
    case kUint32: {
      uint32_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = static_cast<int>(v);
      return true;
    }
    case kInt32: {
      int32_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = v;
      return true;
    }
    case kUint64: {
      uint64_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = static_cast<int>(v);
      return true;
    }
    case kInt64: {
      int64_t v;
      if (!reader->Scalar(&v)) {
        return false;
      }
      *value = static_cast<int>(v);
      return true;
    }
    default:
      return false;
  }
}

bool SkipValue(Reader* reader, uint32_t type) {
  // 跳过暂时不关心的元数据，保证后续键值仍从正确偏移读取。
  switch (type) {
    case kUint8:
    case kInt8:
    case kBool:
      return reader->Skip(1);
    case kUint16:
    case kInt16:
      return reader->Skip(2);
    case kUint32:
    case kInt32:
    case kFloat32:
      return reader->Skip(4);
    case kUint64:
    case kInt64:
    case kFloat64:
      return reader->Skip(8);
    case kString: {
      std::string ignored;
      return reader->String(&ignored);
    }
    case kArray: {
      // 数组元素可能仍是复合类型，因此递归跳过每个元素。
      uint32_t element_type;
      uint64_t count;
      if (!reader->ArrayHeader(&element_type, &count) ||
          count > reader->remaining()) {
        return false;
      }
      for (uint64_t i = 0; i < count; ++i) {
        if (!SkipValue(reader, element_type)) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}

bool IsSpecial(const std::string& token) {
  // 尖括号 token 视为特殊字符串，十六进制字节 token 除外。
  return token.size() >= 2 && token.front() == '<' && token.back() == '>' &&
         token.find("<0x") != 0;
}

std::vector<std::string> SplitText(const std::string& text) {
  // 使用轻量规则完成预分词，降低 GGUF tokenizer 的运行时依赖。
  std::vector<std::string> chunks;
  size_t position = 0;

  while (position < text.size()) {
    size_t start = position;
    if (std::isspace(static_cast<unsigned char>(text[position]))) {
      // 保留连续空白，并将末尾单个空格附着到后续片段。
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
      // 字母数字和 UTF-8 非 ASCII 字节作为普通文本连续收集。
      ++position;
      while (position < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[position]);
        if (!(std::isalnum(c) || c >= 0x80)) {
          break;
        }
        ++position;
      }
    } else {
      // 连续标点独立成段，避免跨类型产生不符合预分词规则的合并。
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
  // 构造 GPT-2 风格的字节到 Unicode 映射及其逆映射。
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
    // 为全部 256 个字节生成唯一且可逆的 Unicode 表示。
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
  // 加载新模型前清理旧元数据，确保失败或重载不会混入旧词表内容。
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  special_tokens_.clear();
  bos_id_ = eos_id_ = unk_id_ = -1;
  std::ifstream input(gguf_path, std::ios::binary);
  if (!input) {
    return false;
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  Reader reader(bytes);
  char magic[4];
  uint32_t version;
  uint64_t tensor_count, metadata_count;
  if (!reader.Read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0 ||
      !reader.Scalar(&version) || version < 1 || version > 3 ||
      !reader.Scalar(&tensor_count) || !reader.Scalar(&metadata_count)) {
    return false;
  }

  // 依次读取 GGUF 元数据；词表和合并表是 BPE 编码的必要数据。
  (void)tensor_count;
  std::vector<std::string> tokens, merges;
  for (uint64_t index = 0; index < metadata_count; ++index) {
    std::string key;
    uint32_t type;
    if (!reader.String(&key) || !reader.Scalar(&type)) {
      return false;
    }
    if (key == "tokenizer.ggml.tokens" && type == kArray) {
      // tokenizer.ggml.tokens 按数组下标直接定义 token ID。
      uint32_t element_type;
      uint64_t count;
      if (!reader.ArrayHeader(&element_type, &count) || element_type != kString) {
        return false;
      }
      tokens.reserve(static_cast<size_t>(count));
      for (uint64_t i = 0; i < count; ++i) {
        std::string token;
        if (!reader.String(&token)) {
          return false;
        }
        tokens.push_back(std::move(token));
      }
    } else if (key == "tokenizer.ggml.merges" && type == kArray) {
      // tokenizer.ggml.merges 的数组顺序即 BPE 合并优先级。
      uint32_t element_type;
      uint64_t count;
      if (!reader.ArrayHeader(&element_type, &count) || element_type != kString) {
        return false;
      }
      merges.reserve(static_cast<size_t>(count));
      for (uint64_t i = 0; i < count; ++i) {
        std::string merge;
        if (!reader.String(&merge)) {
          return false;
        }
        merges.push_back(std::move(merge));
      }
    } else if (key == "tokenizer.ggml.bos_token_id" ||
               key == "tokenizer.ggml.eos_token_id" ||
               key == "tokenizer.ggml.unknown_token_id" ||
               key == "tokenizer.ggml.unk_token_id") {
      // 控制 token ID 在不同 GGUF 生成器中可能采用多种整数宽度。
      int value;
      if (!ReadInteger(&reader, type, &value)) {
        return false;
      }
      if (key.find("bos") != std::string::npos) {
        bos_id_ = value;
      } else if (key.find("eos") != std::string::npos) {
        eos_id_ = value;
      } else {
        unk_id_ = value;
      }
    } else if (!SkipValue(&reader, type)) {
      // 张量描述之外的其他元数据与分词无关，但仍需正确跳过。
      return false;
    }
  }
  if (tokens.empty()) {
    return false;
  }

  // 建立 token 内容到 ID 的索引，并解析合并规则的优先级。
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
    // 同时识别特殊 token 和常见控制 token 别名。
    const std::string& token = id_to_token_[id];
    if (IsSpecial(token)) {
      special_tokens_.emplace_back(token, id);
    }
    if (token == "<unk>" || token == "<UNK>") {
      unk_id_ = id;
    }
    if (token == "<bos>" || token == "<s>") {
      bos_id_ = id;
    }
    if (token == "<eos>" || token == "</s>") {
      eos_id_ = id;
    }
  }
  std::sort(
      special_tokens_.begin(), special_tokens_.end(),
      [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
  return true;
}

std::vector<int> GgufTokenizer::Encode(const std::string& text) const {
  std::vector<int> result;
  size_t offset = 0;
  while (offset < text.size()) {
    // GGUF 与独立 BPE 词表一样，必须先匹配特殊字符串。
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
    // 普通文本段截止到下一个特殊 token，避免特殊内容参与 BPE。
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
      // 对预分词片段反复执行最高优先级合并，直到不存在可合并相邻符号。
      std::vector<std::string> symbols;
      for (unsigned char byte : chunk) {
        symbols.push_back(b2u_[byte]);
      }

      while (symbols.size() > 1) {
        // 每轮仅合并 rank 最小的相邻符号对。
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
        // GGUF 词表不包含最终符号时回退到 UNK ID。
        auto it = vocab_.find(symbol);
        result.push_back(it == vocab_.end() ? unk_id_ : it->second);
      }
    }
    offset = end;
  }
  return result;
}

std::string GgufTokenizer::DecodeToken(int token) const {
  // 将 GPT-2 Unicode 字节符号还原为原始 UTF-8 字节。
  if (token < 0 || token >= static_cast<int>(id_to_token_.size())) {
    return {};
  }
  const std::string& value = id_to_token_[token];
  std::string result;

  for (size_t position = 0; position < value.size();) {
    // 从最长候选开始查询逆映射，未命中的普通 Unicode 内容原样追加。
    bool found = false;
    for (size_t length = std::min<size_t>(4, value.size() - position); length > 0;
         --length) {
      auto it = u2b_.find(value.substr(position, length));
      if (it != u2b_.end()) {
        result.push_back(static_cast<char>(it->second));
        position += length;
        found = true;
        break;
      }
    }
    if (!found) {
      result += value.substr(position);
      break;
    }
  }
  return result;
}

std::string GgufTokenizer::Decode(const std::vector<int>& tokens) const {
  // 顺序拼接 token 解码结果，恢复完整字符串。
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }
  return result;
}

std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path) {
  // GGUF 校验或元数据解析失败时返回 nullptr。
  auto tokenizer = std::make_unique<GgufTokenizer>();
  if (!tokenizer->Load(gguf_path)) {
    return nullptr;
  }
  return tokenizer;
}

}  // namespace mini_llama
