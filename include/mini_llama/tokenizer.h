#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_TOKENIZER_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_TOKENIZER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mini_llama {

// 统一抽象接口
class ITokenizer {
 public:
  virtual ~ITokenizer() = default;

  // 编码器
  virtual std::vector<int> Encode(const std::string& text) const = 0;
  // 解码 Token
  virtual std::string DecodeToken(int token) const = 0;
  // 解码器
  virtual std::string Decode(const std::vector<int>& tokens) const = 0;

  // 获取词表的基本信息
  virtual int vocab_size() const = 0;  // tokenizer 词表的容量
  virtual int bos_id() const = 0;      // 序列开始标记
  virtual int eos_id() const = 0;      // 序列结束时标记
  virtual int unk_id() const = 0;      // 未知标记
};

// ASCII 分词器：针对 0..127 的字符级映射
class AsciiTokenizer : public ITokenizer {
 public:
  static constexpr int kVocabSize = 128;

  AsciiTokenizer() = default;

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return kVocabSize; }
  int bos_id() const override { return 1; }
  int eos_id() const override { return 2; }
  int unk_id() const override { return 0; }
};

// JSON 词表分词器：从 JSON 文件加载词表
// 格式：[{"id": 0, "content": "<unk>", "special": true}, ...]
// 编码自动添加 BOS，并采用贪心最长匹配；未匹配字节回退到 UNK
// 文件无法读取或格式非法时构造函数抛出 std::runtime_error
class JsonVocabTokenizer : public ITokenizer {
 public:
  explicit JsonVocabTokenizer(const std::string& vocab_path);

  std::vector<int> Encode(const std::string& text) const override;
  std::string DecodeToken(int token) const override;
  std::string Decode(const std::vector<int>& tokens) const override;

  int vocab_size() const override { return vocab_size_; }
  int bos_id() const override { return bos_id_; }
  int eos_id() const override { return eos_id_; }
  int unk_id() const override { return unk_id_; }

 private:
  struct VocabEntry {
    int id = 0;
    std::string content;
    bool special = false;
  };

  int vocab_size_ = 0;
  int bos_id_ = 1;
  int eos_id_ = 2;
  int unk_id_ = 0;

  // id -> entry
  std::vector<VocabEntry> id_to_entry_;

  // content -> id
  std::vector<std::pair<std::string, int>> content_to_id_;
};

// BPE 分词器（GPT-2 风格）
// 加载 vocab.json + merges.txt + special_tokens.json
class BpeTokenizer : public ITokenizer {
 public:
  BpeTokenizer() = default;
  ~BpeTokenizer() override = default;

  bool Load(const std::string& vocab_path, const std::string& merges_path,
            const std::string& special_path);

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

  int vocab_size_ = -1;
  int bos_id_ = -1;
  int eos_id_ = -1;
  int unk_id_ = -1;

  // 特殊 token（例如 <|im_start|>）按长度降序排序，以便最长匹配
  std::vector<std::pair<std::string, int>> special_tokens_;

  std::vector<std::string> b2u_;                  // byte -> Unicode string
  std::unordered_map<std::string, uint8_t> u2b_;  // Unicode string -> byte

  void BuildByteMappings();
};

// -------------------------------------------------------
// 工厂函数
// -------------------------------------------------------

// 如果 vocab_path 非空且词表有效，则加载 JsonVocabTokenizer
// 路径不存在或 JSON 加载失败时回退到 AsciiTokenizer
std::unique_ptr<ITokenizer> CreateTokenizer(const std::string& vocab_path);

// 从 vocab.json + merges.txt + special_tokens.json 加载 BPE 分词器。
std::unique_ptr<ITokenizer> CreateBpeTokenizer(const std::string& vocab_path,
                                               const std::string& merges_path,
                                               const std::string& special_path);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_TOKENIZER_H_
