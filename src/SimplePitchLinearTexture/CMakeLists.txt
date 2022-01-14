cmake_minimum_required(VERSION 3.18)
project(simplePitchLinearTexture)

set(CMAKE_CUDA_STANDARD 11)

include_directories(../../common/inc/)

add_executable(simplePitchLinearTexture  simplePitchLinearTexture.cu)

set_target_properties(simplePitchLinearTexture PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
