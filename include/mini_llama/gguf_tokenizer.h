#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "mini_llama/tokenizer.h"

namespace mini_llama {

// GgufTokenizer：直接从 GGUF 元数据加载分词器数据
// // 实现带有字节级回退的 GPT-2 风格 BPE
class GgufTokenizer : public ITokenizer {
 public:
  GgufTokenizer() = default;
  ~GgufTokenizer() override = default;

  // 从 GGUF 文件加载分词器。如果文件缺少所需的分词器元数据，则返回 false。
  bool Load(const std::string& gguf_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return static_cast<int>(id_to_token_.size()); }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  std::unordered_map<std::string, int> vocab_;
  std::vector<std::string> id_to_token_;
  std::map<std::pair<std::string, std::string>, int> merge_ranks_;
  std::vector<std::pair<std::string, int>> special_tokens_;  // 按长度降序排序

  // 字节 <-> Unicode 映射（GPT-2 风格）
  std::vector<std::string> b2u_;
  std::unordered_map<std::string, uint8_t> u2b_;

  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;

  void BuildByteMappings();
};

// ----------------------------------------------
// 工厂函数：从 GGUF 文件创建 GgufTokenizer。
// ----------------------------------------------

// 如果 GGUF 缺少分词器元数据，则返回 nullptr
std::unique_ptr<ITokenizer> CreateGgufTokenizer(const std::string& gguf_path);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_TOKENIZER_H_
