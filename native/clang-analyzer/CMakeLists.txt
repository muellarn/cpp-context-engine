cmake_minimum_required(VERSION 3.20)

project(cpp_context_clang_analyzer VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
find_package(ZLIB REQUIRED)

if(NOT LLVM_VERSION_MAJOR EQUAL 18)
  message(FATAL_ERROR "cpp-context-clang-analyzer requires LLVM and Clang 18")
endif()

add_executable(cpp-context-clang-analyzer src/main.cpp)
target_compile_features(cpp-context-clang-analyzer PRIVATE cxx_std_17)
target_include_directories(cpp-context-clang-analyzer SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
target_compile_definitions(cpp-context-clang-analyzer PRIVATE
  ${LLVM_DEFINITIONS_LIST}
  CPP_CONTEXT_ANALYZER_VERSION="${PROJECT_VERSION}"
)
target_link_libraries(cpp-context-clang-analyzer PRIVATE clang-cpp LLVM ZLIB::ZLIB)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(cpp-context-clang-analyzer PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
  add_executable(cpp-context-path-cache-test tests/path_cache_test.cpp)
  target_compile_features(cpp-context-path-cache-test PRIVATE cxx_std_17)
  target_include_directories(cpp-context-path-cache-test PRIVATE src)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(cpp-context-path-cache-test PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME path-cache COMMAND cpp-context-path-cache-test)
endif()

install(TARGETS cpp-context-clang-analyzer RUNTIME DESTINATION bin)
