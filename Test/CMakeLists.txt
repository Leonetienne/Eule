cmake_minimum_required(VERSION 3.16)
project(Tests)

set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(_EULE_NO_INTRINSICS_)

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
        Math__Min.cpp
        Math__Similar.cpp
        Matrix4x4.cpp
        Vector2.cpp
        Vector3.cpp
        Vector4.cpp
)

target_link_libraries(Tests Eule)
