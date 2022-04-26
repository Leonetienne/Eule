cmake_minimum_required(VERSION 3.16)
project(Eule)

## Library
set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(_EULE_NO_INTRINSICS_)

FILE(GLOB main_src src/*.cpp)
add_library(Eule
  ${main_src}
)

target_include_directories(Eule PRIVATE include)

## Tests
FILE(GLOB test_src test/*.cpp)
add_executable(Eule_tests
  test/Catch2.h
  ${test_src}
)
target_link_libraries(Eule_tests Eule)

target_include_directories(Eule_tests PRIVATE include)
