cmake_minimum_required(VERSION 3.17)

# 项目配置
project(mini_llama LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MINI_LLAMA_CUDA "Enable the CUDA backend" OFF)
if(MINI_LLAMA_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# mini_llama 核心库
add_library(mini_llama_core
  src/backend.cc
  src/cuda_runtime.cc
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
  src/gguf.cc
  src/gguf_loader.cc
  src/quantized_tensor.cc
  src/quant.cc
  src/model.cc
  src/debug.cc
  src/thread_pool.cc
)

if(MINI_LLAMA_CUDA)
  target_compile_definitions(mini_llama_core PUBLIC MINI_LLAMA_USE_CUDA=1)
  target_link_libraries(mini_llama_core PRIVATE CUDA::cudart)
endif()

target_include_directories(mini_llama_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 命令行程序复用 src/main.cc；宏只在该目标中启用全局 main()
add_executable(mini-llama
  src/main.cc
)

target_compile_definitions(mini-llama
  PRIVATE
    MINI_LLAMA_BUILD_EXECUTABLE
)

target_link_libraries(mini-llama
  PRIVATE
    mini_llama_core
)

# 测试模块
include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME backend-cli-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            "MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>"
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_backend_cli.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(backend-cli-smoke PROPERTIES TIMEOUT 180)
  add_test(NAME golden-logits
    COMMAND ${CMAKE_COMMAND} -E env
      "MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>"
      "PYTHONDONTWRITEBYTECODE=1"
      ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/test_golden.py
  )
  set_tests_properties(golden-logits PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TIMEOUT 120
  )
endif()
