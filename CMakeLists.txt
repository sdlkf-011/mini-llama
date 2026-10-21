cmake_minimum_required(VERSION 3.17)

# 项目配置
project(mini_llama LANGUAGES CXX)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA 模块使用 Runtime API、cuBLAS 和自定义 Kernel；CPU 构建无需 CUDA Toolkit
option(MINI_LLAMA_CUDA "Enable the CUDA backend" OFF)
if(MINI_LLAMA_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# mini_llama 核心库
add_library(mini_llama_core
  src/backend.cc
  src/cuda_runtime.cc
  src/cuda_tensor.cc
  src/cuda_matmul.cc
  src/cuda_ops.cc
  src/cuda_quant.cc
  src/tensor.cc
  src/ops.cc
  src/matmul_optimized.cc
  src/gguf_tokenizer.cc
  src/json_vocab_tokenizer.cc
  src/bpe_tokenizer.cc
  src/tokenizer.cc
  src/context.cc
  src/cuda_attention.cc
  src/cuda_kv_cache.cc
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
  target_sources(mini_llama_core PRIVATE src/cuda_ops.cu src/cuda_quant.cu
                                               src/cuda_attention.cu)
  set_target_properties(mini_llama_core PROPERTIES CUDA_STANDARD 17
                                                  CUDA_STANDARD_REQUIRED ON
                                                  CUDA_RUNTIME_LIBRARY Shared)
  target_compile_definitions(mini_llama_core PUBLIC MINI_LLAMA_USE_CUDA=1)
  target_link_libraries(mini_llama_core PRIVATE CUDA::cudart CUDA::cublas)
endif()

target_include_directories(mini_llama_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(mini_llama_core PUBLIC Threads::Threads)

# 可选 POSIX 行编辑器；非交互输入不使用 libedit
set(MINI_LLAMA_LIBEDIT "AUTO" CACHE STRING "Use libedit for interactive input: AUTO, ON, or OFF")
set_property(CACHE MINI_LLAMA_LIBEDIT PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${MINI_LLAMA_LIBEDIT}" mini_llama_libedit_mode)
if(NOT mini_llama_libedit_mode MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR "MINI_LLAMA_LIBEDIT must be AUTO, ON, or OFF")
endif()

set(MINI_LLAMA_HAS_LIBEDIT OFF)
if(UNIX AND NOT mini_llama_libedit_mode STREQUAL "OFF")
  find_path(LIBEDIT_INCLUDE_DIR NAMES histedit.h)
  find_library(LIBEDIT_LIBRARY NAMES edit)
  if(LIBEDIT_INCLUDE_DIR AND LIBEDIT_LIBRARY)
    set(MINI_LLAMA_HAS_LIBEDIT ON)
    target_include_directories(mini_llama_core PRIVATE ${LIBEDIT_INCLUDE_DIR})
    target_link_libraries(mini_llama_core PRIVATE ${LIBEDIT_LIBRARY})
    target_compile_definitions(mini_llama_core PRIVATE MINI_LLAMA_USE_LIBEDIT=1)
  elseif(mini_llama_libedit_mode STREQUAL "ON")
    message(FATAL_ERROR "libedit development files are required (Ubuntu/Debian: apt install libedit-dev)")
  else()
    message(WARNING "libedit was not found; using the built-in UTF-8 terminal editor (optional: apt install libedit-dev).")
  endif()
elseif(mini_llama_libedit_mode STREQUAL "ON")
  message(FATAL_ERROR "MINI_LLAMA_LIBEDIT=ON requires a POSIX platform")
endif()
if(MINI_LLAMA_HAS_LIBEDIT)
  message(STATUS "Interactive line editing: libedit")
elseif(UNIX)
  message(STATUS "Interactive line editing: built-in UTF-8 editor")
else()
  message(STATUS "Interactive input: std::getline (UTF-8 terminal line editing disabled)")
endif()

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
  if(UNIX)
    add_test(NAME terminal-input
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_terminal_input.py
              --binary $<TARGET_FILE:mini-llama>
              --model ${CMAKE_SOURCE_DIR}/models/tiny)
    set_tests_properties(terminal-input PROPERTIES TIMEOUT 30)
  endif()
  add_test(
    NAME backend-cli-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            "MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>"
            "PYTHONDONTWRITEBYTECODE=1"
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_backend_cli.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(backend-cli-smoke PROPERTIES TIMEOUT 180)
  add_test(
    NAME cuda-chat-smoke
    COMMAND ${CMAKE_COMMAND} -E env
            "MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>"
            "PYTHONDONTWRITEBYTECODE=1"
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_cuda_chat_smoke.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(cuda-chat-smoke PROPERTIES
    TIMEOUT 600 SKIP_RETURN_CODE 77 LABELS "cuda;smoke")
  add_test(
    NAME threaded-cli-smoke
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/scripts/test_threaded_cli.py
            --binary $<TARGET_FILE:mini-llama>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(threaded-cli-smoke PROPERTIES TIMEOUT 180)
  add_test(NAME golden-logits
    COMMAND ${CMAKE_COMMAND} -E env
      "MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>"
      "PYTHONDONTWRITEBYTECODE=1"
      ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/scripts/test_golden.py
  )
  set_tests_properties(golden-logits PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TIMEOUT 120
    SKIP_RETURN_CODE 77
  )
endif()
