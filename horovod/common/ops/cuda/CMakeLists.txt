cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(adasum_cuda_kernel)

find_package(CUDA)

list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC -D_FORCE_INLINES -arch=sm_60")

cuda_add_library(adasum_cuda_kernels
adasum_cuda_kernels.cu
)
