find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

foreach(google_test_target IN ITEMS gtest gtest_main gmock gmock_main)
  if(TARGET ${google_test_target})
    set_property(TARGET ${google_test_target} PROPERTY EXPORT_COMPILE_COMMANDS OFF)
  endif()
endforeach()

include(GoogleTest)

# 每个源码注册独立 GTest 目标；可选第三参数作为 CTest 名称前缀
function(add_unit_test test_name test_source)
  add_executable(${test_name}
    ${test_source}
  )

  target_link_libraries(${test_name}
    PRIVATE
      mini_llama_core
      GTest::gtest_main
  )

  if(ARGC GREATER 2)
    gtest_discover_tests(${test_name} TEST_PREFIX "${ARGV2}")
  else()
    gtest_discover_tests(${test_name})
  endif()
endfunction()

add_unit_test(mini-llama-batch-tests unit/batch_test.cc)
add_unit_test(mini-llama-cuda-runtime-tests cuda_runtime_test.cc "cuda_runtime.")

add_unit_test(mini-llama-cuda-tensor-tests cuda_tensor_test.cc "cuda_tensor.")
add_unit_test(mini-llama-cuda-matmul-tests cuda_matmul_test.cc)
add_unit_test(mini-llama-cuda-forward-linear-tests cuda_forward_linear_test.cc
              "cuda_forward_linear.")
# 与文档中的直接运行路径一致；其余既有测试仍保留在 tests/
set_target_properties(mini-llama-cuda-matmul-tests
  mini-llama-cuda-forward-linear-tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
add_unit_test(mini-llama-chat-tests unit/chat_test.cc)
add_unit_test(mini-llama-debug-tests unit/debug_test.cc)
add_unit_test(mini-llama-forward-tests unit/forward_test.cc)
add_unit_test(mini-llama-gguf-tests unit/gguf_test.cc)
add_unit_test(mini-llama-gguf-loader-tests unit/gguf_loader_test.cc)
add_unit_test(mini-llama-gguf-tokenizer-tests unit/gguf_tokenizer_test.cc)
add_unit_test(mini-llama-kv-cache-tests unit/kv_cache_test.cc)
add_unit_test(mini-llama-loader-tests unit/loader_test.cc)
add_unit_test(mini-llama-ops-tests unit/ops_test.cc)
add_unit_test(mini-llama-quant-tests unit/quant_test.cc)
add_unit_test(mini-llama-sampler-tests unit/sampler_test.cc)
add_unit_test(mini-llama-tensor-tests unit/tensor_test.cc)
add_unit_test(mini-llama-tokenizer-tests unit/tokenizer_test.cc)

# 保留 tests 根目录的既有测试，单元测试目录按模块分别注册
add_unit_test(mini-llama-threaded-matmul-tests threaded_matmul_test.cc
              "mini-llama-tests.")
add_unit_test(mini-llama-chat-integration-tests chat_test.cc
              "mini-llama-tests.")
add_unit_test(mini-llama-gguf-tokenizer-integration-tests gguf_tokenizer_test.cc
              "mini-llama-tests.")
add_unit_test(mini-llama-gguf-integration-tests gguf_test.cc
              "mini-llama-tests.")
add_unit_test(mini-llama-debug-integration-tests debug_test.cc
              "mini-llama-tests.")
add_unit_test(mini-llama-main-tests main_test.cc)
add_unit_test(mini-llama-radix-tree-tests radix_tree_test.cc)
add_unit_test(mini-llama-request-context-tests request_context_test.cc)

target_compile_definitions(mini-llama-gguf-integration-tests
  PRIVATE
    MINI_LLAMA_TEST_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/tiny"
)

target_compile_definitions(mini-llama-loader-tests
  PRIVATE
    MINI_LLAMA_TEST_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/tiny"
)

target_compile_definitions(mini-llama-main-tests
  PRIVATE
    MINI_LLAMA_TEST_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/tiny"
)

target_compile_definitions(mini-llama-gguf-tokenizer-integration-tests
  PRIVATE
    MINI_LLAMA_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_compile_definitions(mini-llama-chat-integration-tests
  PRIVATE
    MINI_LLAMA_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

# 本阶段独立 CUDA Ops/Quant 测试，不包含后续设备前向模块
add_unit_test(mini-llama-cuda-quant-tests cuda_quant_test.cc)
add_unit_test(mini-llama-cuda-ops-tests cuda_ops_test.cc "cuda_ops.")
