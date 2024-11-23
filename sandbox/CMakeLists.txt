add_executable(basic basic.cpp)
target_link_libraries(basic PUBLIC serena)
target_compile_definitions(basic PRIVATE ROOT="${CMAKE_CURRENT_SOURCE_DIR}/")

add_executable(cloth cloth.cpp)
target_link_libraries(cloth PUBLIC serena)