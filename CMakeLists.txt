cmake_minimum_required(VERSION 3.17)

# 项目配置
project(mini_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# mini_llama 核心库
add_library(mini_llama_core
  src/tensor.cc
  src/ops.cc
  src/matmul_optimized.cc
  src/gguf_tokenizer.cc
  src/json_vocab_tokenizer.cc
  src/bpe_tokenizer.cc
  src/tokenizer.cc
  src/context.cc
  src/kv_cache.cc
  src/batch.cc
  src/forward.cc
  src/request_context.cc
  src/radix_tree.cc
  src/sampler.cc
  src/main.cc
  src/chat.cc
  src/prompt_builder.cc
  src/terminal.cc
  src/loader.cc
)

target_include_directories(mini_llama_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 测试模块
include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
