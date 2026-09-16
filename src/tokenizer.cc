#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

std::vector<int> AsciiTokenizer::Encode(const std::string& text) const {
  // ASCII 字符直接以其字节值作为 token，非 ASCII 字节统一映射到 UNK。
  std::vector<int> result;
  result.reserve(text.size());
  for (unsigned char c : text) {
    result.push_back(c < kVocabSize ? c : unk_id());
  }

  return result;
}

std::string AsciiTokenizer::DecodeToken(int token) const {
  // 仅接受合法的 0..127 token，非法 token 解码为空串。
  if (token < 0 || token >= kVocabSize) {
    return {};
  }
  return std::string(1, static_cast<char>(token));
}

std::string AsciiTokenizer::Decode(const std::vector<int>& tokens) const {
  // 按 token 顺序拼接字符，保持解码结果与输入序列一致。
  std::string result;
  for (int token : tokens) {
    result += DecodeToken(token);
  }

  return result;
}

std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path) {
  // 词表路径为空或文件不存在时，使用无需外部资源的 ASCII 分词器。
  if (vocab_path.empty() || !std::filesystem::is_regular_file(vocab_path)) {
    return std::make_unique<AsciiTokenizer>();
  }
  return std::make_unique<JsonVocabTokenizer>(vocab_path);
}

std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path) {
  auto tokenizer = std::make_unique<BpeTokenizer>();
  // 三个文件必须全部成功加载；失败时返回空指针，避免暴露半初始化对象。
  if (!tokenizer->Load(vocab_path, merges_path, special_path)) {
    return nullptr;
  }
  return tokenizer;
}

}  // namespace mini_llama
