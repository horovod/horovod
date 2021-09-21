if (${HAVE_CUDA})
  message("Build GPU Library with CUDA")
  set(ENV{PYTHONPATH} "${PROJECT_SOURCE_DIR}/cmake:$ENV{PYTHONPATH}")
  execute_process(COMMAND ${PY_EXE} -c "import build_utils; print(' '.join(build_utils.get_nvcc_flags()))"
                  OUTPUT_VARIABLE HVD_NVCC_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  MESSAGE(STATUS "HVD_NVCC_COMPILE_FLAGS = ${HVD_NVCC_COMPILE_FLAGS}")
  
  list(APPEND CUDA_NVCC_FLAGS "${HVD_NVCC_COMPILE_FLAGS}")
  
  cuda_add_library(horovod_gpu_kernels gpu_kernels.cu OPTIONS -D_GLIBCXX_USE_CXX11_ABI=1)
  
  # if we need compatible c++ abi, build a compatible version
  cuda_add_library(compatible_horovod_gpu_kernels gpu_kernels.cu OPTIONS -D_GLIBCXX_USE_CXX11_ABI=0)
elseif (${HAVE_ROCM})
  message("Build GPU Library with ROCM")
  if (NOT DEFINED HCC_APTH)
    if (DEFINED ENV{HCC_PATH})
      set(HIP_PATH ${HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
      set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been set")
    endif()
    set(HCC_HOME "{HCC_PATH}")
  endif()

  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin")
  set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  set(HIP_HCC_FLAGS ${HIP_HCC_FLAGS};-D__HIP_PLATFORM_HIPCC__=1;-fPIC)
  find_package(HIP QUIET REQUIRED)
  set(HIP_HIPCC_FLAGS ${HIP_HIPCC_FLAGS};-fPIC)
  list(APPEND HIP_HCC_FLAGS_RELEASE -O3 -fPIC)
  list(APPEND HIP_HCC_FLAGS_DEBUG -G -fPIC)
  
  list(APPEND HIP_HIPCC_FLAGS -D_GLIBCXX_USE_CXX11_ABI=1 -fPIC)
  hip_add_library(horovod_gpu_kernels STATIC gpu_kernels.cu)
endif()
