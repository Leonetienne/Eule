cmake_minimum_required(VERSION 3.16)
project(Tests)

set(CMAKE_CXX_STANDARD 17)

include_directories(..)
link_directories(../Eule/cmake-build-debug)

add_executable(Tests
        Catch2.h
        main.cpp
        Math__Mod.cpp
        Math__Oscillate.cpp
        Math__Abs.cpp
        Math__Clamp.cpp
        Math__Lerp.cpp
        Math__Max.cpp
)

target_link_libraries(Tests Eule)
