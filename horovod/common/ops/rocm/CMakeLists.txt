message(STATUS "Build Horovod for ROCm")
set(HCC_PATH "${ROCM_ROOT}/bin/hipcc" CACHE PATH "Path to which HCC has been set")

set(HCC_HOME "{HCC_PATH}")

list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT})
list(APPEND CMAKE_MODULE_PATH "${ROCM_ROOT}/lib/cmake/hip")
set(HIP_CLANG_PATH "${ROCM_ROOT}/llvm/bin")
set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS};-D__HIP_PLATFORM_HIPCC__=1;-fPIC)
set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS};-fPIC)
list(APPEND HIP_HCC_FLAGS_RELEASE -O3 -fPIC)
list(APPEND HIP_HCC_FLAGS_DEBUG -G -fPIC)

list(APPEND HIP_HIPCC_FLAGS -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC)
hip_add_library(horovod_cuda_kernels STATIC hip_kernels.cu)
target_compile_definitions(horovod_cuda_kernels PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
hip_add_library(compatible_horovod_cuda_kernels STATIC hip_kernels.cu)
target_compile_definitions(compatible_horovod_cuda_kernels PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
