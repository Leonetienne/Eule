cmake_minimum_required(VERSION 3.16)
project(Eule)

set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(_EULE_NO_INTRINSICS_)

add_library(Eule
 src/Collider.cpp
 src/Math.cpp
 src/Matrix4x4.cpp
 src/Quaternion.cpp
 src/Random.cpp
 src/TrapazoidalPrismCollider.cpp
 src/Vector2.cpp
 src/Vector3.cpp
 src/Vector4.cpp
)

target_include_directories(Eule PRIVATE include)

