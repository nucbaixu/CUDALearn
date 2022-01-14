cmake_minimum_required(VERSION 3.18)
project(UnifiedMemoryStreams)

set(CMAKE_CUDA_STANDARD 11)

link_libraries(cublas.lib)
include_directories(../../common/inc/)

add_executable(UnifiedMemoryStreams UnifiedMemoryStreams.cu )

set_target_properties(UnifiedMemoryStreams PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)