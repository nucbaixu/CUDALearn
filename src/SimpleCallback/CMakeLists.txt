cmake_minimum_required(VERSION 3.18)
project(SimpleCallback)

set(CMAKE_CUDA_STANDARD 11)

include_directories(../../common/inc/)

add_executable(SimpleCallback simpleCallback.cu Multithreading.h Multithreading.cpp)

set_target_properties(SimpleCallback PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)