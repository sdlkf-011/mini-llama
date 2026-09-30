#include "mini_llama/tokenizer.h"

#include <string>
#include <vector>

namespace mini_llama {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
  // ASCII 字符直接以其字节值作为 token，非 ASCII 字节统一映射到 UNK
  std::vector<int> result;
  result.reserve(text.size());
  for (unsigned char c : text) {
    result.push_back(c < kVocabSize ? c : unk_id());
  }

  return result;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
  // 仅接受合法的 0..127 token，非法 token 解码为空串
  if (token < 0 || token >= kVocabSize) {
    return {};
  }
  return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
  // 按 token 顺序拼接字符，保持解码结果与输入序列一致
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }

  return result;
}

}  // namespace mini_llama
