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
)

target_include_directories(mini_llama_core
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# 模块测试
file(GLOB UNIT_TEST_SOURCES
  CONFIGURE_DEPENDS
  ${PROJECT_SOURCE_DIR}/tests/unit/*.cc
)

add_executable(mini-llama-tests
  ${UNIT_TEST_SOURCES}
)

target_link_libraries(mini-llama-tests
  PRIVATE
    mini_llama_core
)
