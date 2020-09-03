set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(ENV{PYTHONPATH} "${PROJECT_SOURCE_DIR}/cmake:$ENV{PYTHONPATH}")
execute_process(COMMAND ${PY_EXE} -c "import build_utils; print(' '.join(build_utils.get_nvcc_flags()))"
                OUTPUT_VARIABLE HVD_NVCC_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

MESSAGE(STATUS "HVD_NVCC_COMPILE_FLAGS = ${HVD_NVCC_COMPILE_FLAGS}")

list(APPEND CUDA_NVCC_FLAGS "${HVD_NVCC_COMPILE_FLAGS}")

cuda_add_library(horovod_cuda_kernels cuda_kernels.cu OPTIONS -D_GLIBCXX_USE_CXX11_ABI=1)

# if we need compatible c++ abi, build a compatible version
cuda_add_library(compatible_horovod_cuda_kernels cuda_kernels.cu OPTIONS -D_GLIBCXX_USE_CXX11_ABI=0)