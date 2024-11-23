cmake_minimum_required(VERSION 3.20)
project(serena)

set(CMAKE_CXX_STANDARD 20)

find_package(glfw3 REQUIRED)

add_library(glad glad/glad.c)
target_compile_options(glad PRIVATE "-O3" "-march=native")

add_library(serena INTERFACE)
target_link_libraries(serena INTERFACE glad glfw)
target_include_directories(serena INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(sandbox)
