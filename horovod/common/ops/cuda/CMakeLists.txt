cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(horovod_cuda_kernels)

find_package(CUDA)

# Remove C++ std flags (--std=c++11 included in CUDA_NVCC_FLAGS)
string(REGEX REPLACE  "-std=[^ ]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Set ar
if(DEFINED ENV{AR})
  set(CMAKE_AR $ENV{AR})
endif()

MESSAGE(STATUS "HVD_NVCC_COMPILE_FLAGS = ${HVD_NVCC_COMPILE_FLAGS}")

list(APPEND CUDA_NVCC_FLAGS ${HVD_NVCC_COMPILE_FLAGS})

cuda_add_library(horovod_cuda_kernels
    cuda_kernels.cu
)

set(CMAKE_VERBOSE_MAKEFILE ON)
