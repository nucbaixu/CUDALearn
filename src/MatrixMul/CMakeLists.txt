cmake_minimum_required(VERSION 3.18)
project(MatrixMul)

set(CMAKE_CUDA_STANDARD 11)

include_directories(../../common/inc/)

add_executable(MatrixMul  MatrixMul.cu )

set_target_properties(MatrixMul PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
