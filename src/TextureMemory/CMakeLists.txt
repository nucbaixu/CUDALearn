cmake_minimum_required(VERSION 3.18)
project(TextureMemory)

set(CMAKE_CUDA_STANDARD 11)

include_directories(../../common/inc/)

add_executable(TextureMemory main.cu)

set_target_properties(TextureMemory PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
