cmake_minimum_required(VERSION 3.18)
project(TestCUDADemo CUDA)

set(CMAKE_CUDA_STANDARD 11)

# 引入子目录
ADD_SUBDIRECTORY(src/SimplePitchLinearTexture)
ADD_SUBDIRECTORY(src/TextureMemory)
ADD_SUBDIRECTORY(src/MatrixMul)
add_subdirectory(src/MatrixMulCUBLAS)
add_subdirectory(src/SimpleAtomicIntrinsics)
add_subdirectory(src/SimpleCallback)
add_subdirectory(src/SimpleStreams)
add_subdirectory(src/UnifiedMemoryStreams)