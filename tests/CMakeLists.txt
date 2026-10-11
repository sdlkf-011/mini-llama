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

# 每个模块只注册根目录文件；同名的单元和集成用例已合并。
add_unit_test(mini-llama-batch-tests batch_test.cc)
add_unit_test(mini-llama-chat-tests chat_test.cc)
add_unit_test(mini-llama-cuda-attention-tests cuda_attention_test.cc "cuda_attention.")
add_unit_test(mini-llama-cuda-forward-buffer-tests cuda_forward_buffer_test.cc "cuda_forward_device_resident.")
add_unit_test(mini-llama-cuda-forward-linear-tests cuda_forward_linear_test.cc "cuda_forward_linear.")
add_unit_test(mini-llama-cuda-forward-tests cuda_forward_test.cc "cuda_forward.")
add_unit_test(mini-llama-cuda-kv-cache-tests cuda_kv_cache_test.cc "cuda_kv_cache.")
add_unit_test(mini-llama-cuda-matmul-tests cuda_matmul_test.cc)
add_unit_test(mini-llama-cuda-ops-tests cuda_ops_test.cc "cuda_ops.")
add_unit_test(mini-llama-cuda-quant-tests cuda_quant_test.cc)
add_unit_test(mini-llama-cuda-runtime-tests cuda_runtime_test.cc "cuda_runtime.")
add_unit_test(mini-llama-debug-tests debug_test.cc)
add_unit_test(mini-llama-forward-tests forward_test.cc)
add_unit_test(mini-llama-gguf-loader-tests gguf_loader_test.cc)
add_unit_test(mini-llama-gguf-tests gguf_test.cc)
add_unit_test(mini-llama-gguf-tokenizer-tests gguf_tokenizer_test.cc)
add_unit_test(mini-llama-kv-cache-tests kv_cache_test.cc)
add_unit_test(mini-llama-loader-tests loader_test.cc)
add_unit_test(mini-llama-main-tests main_test.cc)
add_unit_test(mini-llama-ops-tests ops_test.cc)
add_unit_test(mini-llama-quant-tests quant_test.cc)
add_unit_test(mini-llama-radix-tree-tests radix_tree_test.cc)
add_unit_test(mini-llama-request-context-tests request_context_test.cc)
add_unit_test(mini-llama-sampler-tests sampler_test.cc)
add_unit_test(mini-llama-tensor-tests tensor_test.cc)
add_unit_test(mini-llama-threaded-matmul-tests threaded_matmul_test.cc "mini-llama-tests.")
add_unit_test(mini-llama-tokenizer-tests tokenizer_test.cc)

# 与文档中的直接运行路径一致。
set_target_properties(mini-llama-cuda-matmul-tests
  mini-llama-cuda-forward-linear-tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

foreach(model_test IN ITEMS gguf loader main)
  target_compile_definitions(mini-llama-${model_test}-tests PRIVATE
    MINI_LLAMA_TEST_MODEL_DIR="${PROJECT_SOURCE_DIR}/models/tiny")
endforeach()

foreach(source_test IN ITEMS chat gguf-tokenizer)
  target_compile_definitions(mini-llama-${source_test}-tests PRIVATE
    MINI_LLAMA_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
endforeach()
