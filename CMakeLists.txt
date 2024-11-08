cmake_minimum_required(VERSION 3.29)
project(serena)

set(CMAKE_CXX_STANDARD 20)

find_package(glfw3 REQUIRED)

add_library(glad glad/glad.c)
target_compile_options(glad PRIVATE "-O3" "-march=native")

add_executable(serena sandbox/main.cpp serena.hpp)
target_include_directories(serena PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(serena PUBLIC glfw glad)
target_compile_definitions(serena PRIVATE ROOT="${CMAKE_CURRENT_SOURCE_DIR}/sandbox/")
