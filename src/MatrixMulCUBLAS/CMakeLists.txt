cmake_minimum_required(VERSION 3.18)
project(MatrixMulCUBLAS)

set(CMAKE_CUDA_STANDARD 11)

include_directories(../../common/inc/)

link_libraries(cublas.lib)
add_executable(MatrixMulCUBLAS MatrixMulCUBLAS.cu)

set_target_properties(MatrixMulCUBLAS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)