#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_LOADER_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_LOADER_H_

#include <cstddef>
#include <string>
#include <vector>

#include "mini_llama/model.h"

namespace mini_llama {

// model.json 中单个张量的物理布局信息
struct TensorInfo {
  std::string name;        // 全局唯一名称
  std::vector<int> shape;  // 逻辑形状，各维必须为正数
  std::string dtype;       // 当前阶段仅支持 float32
  size_t offset = 0;       // 在 model.bin 中的起始字节偏移
  size_t byte_size = 0;    // 张量连续占用的字节数
};

// 分词器描述。path 由装配层相对于清单目录解析
// 特殊 token ID 是模型输入约定，装配层须与实际分词器核对一致后再启动推理
struct TokenizerInfo {
  std::string type = "ascii";
  std::string path;
  int bos_id = 1;
  int eos_id = 2;
  int unk_id = 0;
};

// 一次解析和完整校验后的模型清单
struct ModelManifest {
  ModelConfig config;
  TokenizerInfo tokenizer;
  std::vector<TensorInfo> tensors;
};

// 将 model.json 解析为清单（配置、分词器和张量元数据）
// 当 JSON 非法、字段不完整或张量布局不一致时抛出 std::runtime_error
ModelManifest ParseManifest(const std::string& config_path);

// 按名称和偏移加载 F32 权重。失败时返回 loaded=false 并填写 load_error
MiniLlamaModel LoadModel(const std::string& config_path,
                         const std::string& weights_path);

// 在不读取 model.bin 的情况下校验并打印模型元数据
bool InspectModel(const std::string& config_path);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_LOADER_H_
