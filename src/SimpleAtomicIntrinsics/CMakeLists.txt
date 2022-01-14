cmake_minimum_required(VERSION 3.18)
project(simpleAtomicIntrinsics)

set(CMAKE_CUDA_STANDARD 11)

include_directories(../../common/inc/)

add_executable(simpleAtomicIntrinsics
        simpleAtomicIntrinsics.cu
        simpleAtomicIntrinsics_cpu.cpp
        simpleAtomicIntrinsics_kernel.cuh)

set_target_properties(simpleAtomicIntrinsics PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)