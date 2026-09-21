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

include(GoogleTest)

function(add_unit_test test_name test_source)
  add_executable(${test_name}
    ${test_source}
  )

  target_link_libraries(${test_name}
    PRIVATE
      mini_llama_core
      GTest::gtest_main
  )

  gtest_discover_tests(${test_name})
endfunction()

add_unit_test(mini-llama-tensor-tests unit/tensor_test.cc)
add_unit_test(mini-llama-tokenizer-tests unit/tokenizer_test.cc)
add_unit_test(mini-llama-kv-cache-tests unit/kv_cache_test.cc)
add_unit_test(mini-llama-radix-tree-tests unit/radix_tree_test.cc)
add_unit_test(mini-llama-chat-tests unit/chat_test.cc)
add_unit_test(mini-llama-request-context-tests unit/request_context_test.cc)
add_unit_test(mini-llama-forward-tests unit/forward_test.cc)
add_unit_test(mini-llama-sampler-tests unit/sampler_test.cc)
