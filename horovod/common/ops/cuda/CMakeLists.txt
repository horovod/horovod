set(ENV{PYTHONPATH} "${PROJECT_SOURCE_DIR}/cmake:$ENV{PYTHONPATH}")
execute_process(COMMAND ${PY_EXE} -c "import build_utils; print(';'.join(get_nvcc_flags()))"
                OUTPUT_VARIABLE HVD_NVCC_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

MESSAGE(STATUS "HVD_NVCC_COMPILE_FLAGS = ${HVD_NVCC_COMPILE_FLAGS}")

list(APPEND CUDA_NVCC_FLAGS ${HVD_NVCC_COMPILE_FLAGS})

cuda_add_library(horovod_cuda_kernels cuda_kernels.cu)
