message(STATUS "Build Horovod for SYCL")

set(CMAKE_CXX_STANDARD 17)

add_library(horovod_sycl_kernels sycl_kernels.cc)
target_compile_options(horovod_sycl_kernels PRIVATE ${SYCL_FLAGS})