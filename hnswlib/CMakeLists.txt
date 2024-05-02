cmake_minimum_required(VERSION 3.2)

project(hnsw-profile)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -Og -g3 -fno-omit-frame-pointer")
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(profile_hnsw profile_hnsw.cc)
target_link_libraries(profile_hnsw)
