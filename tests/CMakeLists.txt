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

add_unit_test(
  mini-llama-tensor-tests
  unit/tensor_test.cc
)

add_unit_test(
  mini-llama-tokenizer-tests
  unit/tokenizer_test.cc
)
