set(HIP_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(ENV{PYTHONPATH} "${PROJECT_SOURCE_DIR}/cmake:$ENV{PYTHONPATH}")
execute_process(COMMAND ${PY_EXE} -c "import build_utils; print(' '.join(build_utils.get_hcc_flags()))"
                OUTPUT_VARIABLE HVD_HCC_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

MESSAGE(STATUS "HVD_HCC_COMPILE_FLAGS = ${HVD_HCC_COMPILE_FLAGS}")

list(APPEND HIP_HCC_FLAGS "${HVD_HCC_COMPILE_FLAGS}")

hip_add_library(horovod_hip_kernels hip_kernels.cu OPTIONS -D_GLIBCXX_USE_CXX11_ABI=1)

# if we need compatible c++ abi, build a compatible version
hip_add_library(compatible_horovod_hip_kernels hip_kernels.cu OPTIONS -D_GLIBCXX_USE_CXX11_ABI=0)
