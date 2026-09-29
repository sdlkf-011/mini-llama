#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_LOADER_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_LOADER_H_

#include <string>

#include "mini_llama/model.h"

namespace mini_llama {

// 从 GGUF 文件加载模型。支持 F32、F16、Q8_0、Q4_0 和 Q4_1 张量，
// 加载时统一转换为现有 CPU 前向路径使用的 F32 Tensor。
// 失败时返回 loaded=false，并在 load_error 中提供原因。
MiniLlamaModel LoadGgufModel(const std::string& gguf_path);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_GGUF_LOADER_H_
