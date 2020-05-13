set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(horovod_cuda_kernels)

find_package(CUDA)

MESSAGE(STATUS "HVD_NVCC_COMPILE_FLAGS = ${HVD_NVCC_COMPILE_FLAGS}")

list(APPEND CUDA_NVCC_FLAGS ${HVD_NVCC_COMPILE_FLAGS})

cuda_add_library(horovod_cuda_kernels
    cuda_kernels.cu
)
