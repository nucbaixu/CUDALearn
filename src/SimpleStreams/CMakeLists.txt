cmake_minimum_required(VERSION 3.18)
project(SimpleStreams)

set(CMAKE_CUDA_STANDARD 11)

include_directories(../../common/inc/)

add_executable(SimpleStreams SimpleStreams.cu)

set_target_properties(SimpleStreams PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)