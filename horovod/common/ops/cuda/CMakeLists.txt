set(ENV{PYTHONPATH} "${PROJECT_SOURCE_DIR}/cmake:$ENV{PYTHONPATH}")
execute_process(COMMAND ${PY_EXE} -c "import build_utils; print(' '.join(build_utils.get_nvcc_flags()))"
                OUTPUT_VARIABLE HVD_NVCC_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

MESSAGE(STATUS "HVD_NVCC_COMPILE_FLAGS = ${HVD_NVCC_COMPILE_FLAGS}")

# If we don't set CMAKE_CUDA_STANDARD, it will default to ${CMAKE_CXX_STANDARD} ("14" at this time). nvcc may fail if
# the --std=c++... argument is passed multiple times.
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(horovod_cuda_kernels cuda_kernels.cu)
target_compile_options(horovod_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       SHELL:${HVD_NVCC_COMPILE_FLAGS}
                       -D_GLIBCXX_USE_CXX11_ABI=1
                       >)

# if we need compatible c++ abi, build a compatible version
add_library(compatible_horovod_cuda_kernels cuda_kernels.cu)
target_compile_options(compatible_horovod_cuda_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       SHELL:${HVD_NVCC_COMPILE_FLAGS}
                       -D_GLIBCXX_USE_CXX11_ABI=0
                       >)
