cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
if(POLICY CMP0074)
    # 1. Introduced with 3.12.4.
    # 2. *_ROOT variables will be checked
    cmake_policy(SET CMP0074 NEW)
endif()

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "Could not find CCache. Consider installing CCache to speed up compilation.")
endif()

project(horovod CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Configure path to modules (for find_package)
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        "${PROJECT_SOURCE_DIR}/cmake/Modules/"
        "${PROJECT_SOURCE_DIR}/cmake/upstream/")
include(cmake/Utilities.cmake)

create_metadata()

# 3rd-parties
include_directories("third_party/HTTPRequest/include"
        "third_party/boost/assert/include"
        "third_party/boost/config/include"
        "third_party/boost/core/include"
        "third_party/boost/detail/include"
        "third_party/boost/iterator/include"
        "third_party/boost/lockfree/include"
        "third_party/boost/mpl/include"
        "third_party/boost/parameter/include"
        "third_party/boost/predef/include"
        "third_party/boost/preprocessor/include"
        "third_party/boost/static_assert/include"
        "third_party/boost/type_traits/include"
        "third_party/boost/utility/include"
        "third_party/lbfgs/include")

# Predefined Eigen and Flatbuffers headers path, they could be replaced by tensorflow headers path
set(EIGEN_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/third_party/eigen")
set(FLATBUFFERS_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/third_party/flatbuffers/include")

# Sources
list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/common.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/controller.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/fusion_buffer_manager.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/group_table.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/half.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/logging.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/message.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/operations.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/parameter_manager.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/process_set.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/response_cache.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/stall_inspector.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/thread_pool.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/timeline.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/tensor_queue.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/ops/collective_operations.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/ops/operation_manager.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/optim/bayesian_optimization.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/optim/gaussian_process.cc"
        "${PROJECT_SOURCE_DIR}/horovod/common/utils/env_parser.cc")

# Default Macro
add_definitions(-DEIGEN_MPL2_ONLY=1)

# Remove platform default std
string(REGEX REPLACE  "-std=[^ ]+" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Pickup ar from environmental variable if available
if(DEFINED ENV{AR})
    set(CMAKE_AR $ENV{AR})
endif()

# Add default project CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC -Wall -ftree-vectorize")

# RelWithDebInfo uses -O2, prefer performance over debug info in RelWithDebInfo build type
string(REPLACE "-O2" "-O3" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-O2" "-O3" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

# Add architecture specific optimization flags
set(ARCH_FLAGS "-mf16c" "-mavx" "-mfma")
set_build_arch_flags("${ARCH_FLAGS}")
# Specify Horovod exports
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup -Wl,-exported_symbols_list,${CMAKE_SOURCE_DIR}/horovod.exp")
    set(CMAKE_MACOSX_RPATH TRUE)
else()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--version-script=${CMAKE_SOURCE_DIR}/horovod.lds -Wl,-Bsymbolic-functions -Wl,-z,relro,-z,now")
endif()
# GPU Operations
set(HOROVOD_GPU $ENV{HOROVOD_GPU})
set(HOROVOD_GPU_OPERATIONS $ENV{HOROVOD_GPU_OPERATIONS})
if(DEFINED HOROVOD_GPU_OPERATIONS AND NOT "${HOROVOD_GPU_OPERATIONS}" MATCHES "^(MPI|NCCL)$")
    message(FATAL_ERROR "HOROVOD_GPU_OPERATIONS=${HOROVOD_GPU_OPERATIONS} is invalid, supported values are '', 'MPI', and 'NCCL'.")
endif()
set_gpu_op(HOROVOD_GPU_ALLREDUCE "MPI;NCCL;DDL")
set_gpu_op(HOROVOD_GPU_ALLGATHER "MPI;NCCL")
set_gpu_op(HOROVOD_GPU_BROADCAST "MPI;NCCL")
set_gpu_op(HOROVOD_GPU_ALLTOALL "MPI;NCCL")
set_gpu_op(HOROVOD_GPU_REDUCESCATTER "MPI;NCCL")

foreach(VAR in ITEMS HOROVOD_GPU_ALLREDUCE HOROVOD_GPU_ALLGATHER HOROVOD_GPU_BROADCAST HOROVOD_GPU_ALLTOALL HOROVOD_GPU_REDUCESCATTER)
    if(DEFINED ${VAR})
        string(SUBSTRING ${${VAR}} 0 1 ${VAR})
        convert_to_ascii_dec(ASCII_DEC ${${VAR}})
        add_definitions(-D${VAR}=${ASCII_DEC})
    endif()
endforeach()

# PYTHON
if(NOT PYTHON_EXECUTABLE)
    find_package(Python 3.6 COMPONENTS Interpreter REQUIRED)
    set(PY_EXE ${Python_EXECUTABLE})
else()
    set(PY_EXE ${PYTHON_EXECUTABLE})
endif()
message(STATUS "Using command ${PY_EXE}")

# MPI
if (NOT "$ENV{HOROVOD_WITHOUT_MPI}" STREQUAL "1")
    set(MPI_REQUIRED "")
    if ("$ENV{HOROVOD_WITH_MPI}" STREQUAL "1")
        set(MPI_REQUIRED "REQUIRED")
    endif ()
    find_package(MPI ${MPI_REQUIRED})
    if(MPI_FOUND)
        include_directories(SYSTEM ${MPI_INCLUDE_PATH})
        list(APPEND LINKER_LIBS ${MPI_LIBRARIES})
        list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/mpi/mpi_context.cc"
                "${PROJECT_SOURCE_DIR}/horovod/common/mpi/mpi_controller.cc"
                "${PROJECT_SOURCE_DIR}/horovod/common/ops/mpi_operations.cc"
                "${PROJECT_SOURCE_DIR}/horovod/common/ops/adasum/adasum_mpi.cc"
                "${PROJECT_SOURCE_DIR}/horovod/common/ops/adasum_mpi_operations.cc")
        add_definitions(-DHAVE_MPI=1)
        set(HAVE_MPI TRUE)
    endif()
endif()

# CUDA and ROCM
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")  # Set to "Static" or "Shared"
endif()
if(DEFINED ENV{HOROVOD_CUDA_HOME})
    set(CMAKE_CUDA_COMPILER "$ENV{HOROVOD_CUDA_HOME}/bin/nvcc")
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
    if(CUDAToolkit_BIN_DIR)
        message("CUDA compiler was not found in $PATH, but searching again in CUDA Toolkit binary directory")
        unset(CMAKE_CUDA_COMPILER CACHE)  # need to clear this from cache, else some versions of CMake go into an infinite loop
        set(CMAKE_CUDA_COMPILER "${CUDAToolkit_BIN_DIR}/nvcc")
        check_language(CUDA)
    endif()
endif()
if(CMAKE_CUDA_COMPILER)
    if((CMAKE_CXX_COMPILER_ID MATCHES GNU) AND (CMAKE_SYSTEM_PROCESSOR MATCHES ppc64le))
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")
        endif()
    endif()
    enable_language(CUDA)
endif()

macro(ADD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    string(TOLOWER "${CMAKE_CUDA_RUNTIME_LIBRARY}" lowercase_CMAKE_CUDA_RUNTIME_LIBRARY)
    if (lowercase_CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "static")
        list(APPEND LINKER_LIBS CUDA::cudart_static)
    elseif (lowercase_CMAKE_CUDA_RUNTIME_LIBRARY STREQUAL "shared")
        list(APPEND LINKER_LIBS CUDA::cudart)
    endif()
    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/cuda_operations.cc"
                        "${PROJECT_SOURCE_DIR}/horovod/common/ops/gpu_operations.cc")
    # CUDA + MPI
    if(HAVE_MPI)
        list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/mpi_gpu_operations.cc")
    endif()
    add_definitions(-DHAVE_CUDA=1 -DHAVE_GPU=1)
    set(HAVE_CUDA TRUE)
    if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
        set(HAVE_SUB_PROJECT_CUDA TRUE PARENT_SCOPE)
    endif()
endmacro()

if(DEFINED HOROVOD_GPU_ALLREDUCE OR DEFINED HOROVOD_GPU_ALLGATHER OR DEFINED HOROVOD_GPU_BROADCAST OR DEFINED HOROVOD_GPU_ALLTOALL OR DEFINED HOROVOD_GPU_REDUCESCATTER)
    if(NOT DEFINED HOROVOD_GPU OR HOROVOD_GPU STREQUAL "CUDA")
        add_cuda()
    elseif(HOROVOD_GPU STREQUAL "ROCM")
        find_package(ROCM REQUIRED)
        include_directories(SYSTEM ${ROCM_INCLUDE_DIRS})
        list(APPEND LINKER_LIBS ${ROCM_LIBRARIES})
        set(CMAKE_CXX_FLAGS "${ROCM_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
        list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/hip_operations.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/common/ops/gpu_operations.cc")
        if(HAVE_MPI)
            list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/mpi_gpu_operations.cc")
        endif()
        add_definitions(-DHAVE_ROCM=1 -DHAVE_GPU=1)
        set(HAVE_ROCM TRUE)
    else()
        message(FATAL_ERROR "Unknown HOROVOD_GPU type: ${HOROVOD_GPU}")
    endif()
endif()

# NCCL
if(HOROVOD_GPU_ALLREDUCE STREQUAL "N" OR HOROVOD_GPU_ALLGATHER STREQUAL "N" OR HOROVOD_GPU_BROADCAST STREQUAL "N" OR HOROVOD_GPU_ALLTOALL STREQUAL "N" OR HOROVOD_GPU_REDUCESCATTER STREQUAL "N")
    if(HAVE_ROCM)
        find_package(rccl REQUIRED)
        include_directories(SYSTEM ${RCCL_INCLUDE_DIRS})
        list(APPEND LINKER_LIBS roc::rccl)
    else()
        find_package(NCCL REQUIRED)
        if (NCCL_MAJOR_VERSION LESS "2")
            message(FATAL_ERROR "Horovod requires NCCL 2.0 or later version please upgrade.")
        endif()
        include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
        list(APPEND LINKER_LIBS ${NCCL_LIBRARIES})
    endif()
    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/nccl_operations.cc")
    add_definitions(-DHAVE_NCCL=1)
    set(HAVE_NCCL TRUE)
endif()

# DDL
if(HOROVOD_GPU_ALLREDUCE STREQUAL "D")
    message(DEPRECATION "DDL backend has been deprecated. Please, start using the NCCL backend by building Horovod with "
                        "'HOROVOD_GPU_OPERATIONS=NCCL'. Will be removed in v0.21.0.")
    list(APPEND LINKER_LIBS "${CUDAToolkit_LIBRARY_DIR}/libddl.so" "${CUDAToolkit_LIBRARY_DIR}/libddl_pack.so")
    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/mpi/ddl_mpi_context_manager.cc"
                        "${PROJECT_SOURCE_DIR}/horovod/common/ops/ddl_operations.cc")
    add_definitions(-DHAVE_DDL=1)
    set(HAVE_DDL TRUE)
endif()

# oneCCL
set(CCL_ROOT $ENV{CCL_ROOT})
if(DEFINED CCL_ROOT)
    set(CCL_CONFIGURATION $ENV{CCL_CONFIGURATION})
    include_directories(${CCL_ROOT}/include/${CCL_CONFIGURATION})
    list(APPEND LINKER_LIBS "${CCL_ROOT}/lib/${CCL_CONFIGURATION}/libccl.so")
    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/ccl_operations.cc")
    add_definitions(-DHAVE_CCL=1)
    set(HAVE_CCL TRUE)
endif()

set(HOROVOD_ALLOW_MIXED_GPU_IMPL $ENV{HOROVOD_ALLOW_MIXED_GPU_IMPL})
if(HOROVOD_GPU_ALLREDUCE STREQUAL "N" AND (HOROVOD_GPU_ALLGATHER STREQUAL "M" OR HOROVOD_GPU_BROADCAST STREQUAL "M" OR HOROVOD_GPU_ALLTOALL STREQUAL "M" OR HOROVOD_GPU_REDUCESCATTER STREQUAL "M") AND
   NOT HOROVOD_ALLOW_MIXED_GPU_IMPL STREQUAL "1")
message(FATAL_ERROR "You should not mix NCCL and MPI GPU due to a possible deadlock.\n"
                    "If you are sure you want to mix them, set the "
                    "HOROVOD_ALLOW_MIXED_GPU_IMPL environment variable to '1'.")
endif()

# NVTX
if (NOT "$ENV{HOROVOD_WITHOUT_NVTX}" STREQUAL "1")
    set(NVTX_REQUIRED "")
    if ("$ENV{HOROVOD_WITH_NVTX}" STREQUAL "1")
        set(NVTX_REQUIRED "REQUIRED")
    endif ()
    find_package(NVTX ${NVTX_REQUIRED})
    if(NVTX_FOUND)
        include_directories(SYSTEM ${NVTX_INCLUDE_DIRS})
        list(APPEND LINKER_LIBS ${NVTX_LIBRARIES})
        add_definitions(-DHAVE_NVTX=1)
        list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/nvtx_op_range.cc")
        set(HAVE_NVTX TRUE)
    endif()
endif()

# Gloo
if (NOT "$ENV{HOROVOD_WITHOUT_GLOO}" STREQUAL "1")
    if(HAVE_MPI)
        set(USE_MPI TRUE)
    else()
        set(USE_MPI FALSE)
    endif()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(USE_LIBUV ON CACHE BOOL "use libuv for gloo transport" FORCE)
    endif()
    set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
    add_subdirectory(third_party/gloo)
    include_directories(third_party/gloo)
    target_compile_definitions(gloo PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/gloo/gloo_context.cc"
                        "${PROJECT_SOURCE_DIR}/horovod/common/gloo/gloo_controller.cc"
                        "${PROJECT_SOURCE_DIR}/horovod/common/gloo/http_store.cc"
                        "${PROJECT_SOURCE_DIR}/horovod/common/gloo/memory_store.cc"
                        "${PROJECT_SOURCE_DIR}/horovod/common/ops/gloo_operations.cc")
    add_definitions(-DHAVE_GLOO=1)
    set(HAVE_GLOO TRUE)
endif()

# NCCL + MPI
if (HAVE_NCCL AND HAVE_MPI)
    list(APPEND SOURCES "${PROJECT_SOURCE_DIR}/horovod/common/ops/adasum_gpu_operations.cc")
endif()

set(HOROVOD_CPU_OPERATIONS $ENV{HOROVOD_CPU_OPERATIONS})
if(DEFINED HOROVOD_CPU_OPERATIONS)
    message(STATUS "Set default CPU operation to " ${HOROVOD_CPU_OPERATIONS})
    if(HOROVOD_CPU_OPERATIONS STREQUAL "MPI")
        if(NOT HAVE_MPI)
            message(FATAL_ERROR "MPI is not installed, try changing HOROVOD_CPU_OPERATIONS.")
        endif()
        add_definitions(-DHOROVOD_CPU_OPERATIONS_DEFAULT=M)
    elseif(HOROVOD_CPU_OPERATIONS STREQUAL "MLSL")
        message(FATAL_ERROR "Intel(R) MLSL was removed. Upgrade to oneCCL and set HOROVOD_CPU_OPERATIONS=CCL.")
    elseif(HOROVOD_CPU_OPERATIONS STREQUAL "CCL")
        if(NOT HAVE_CCL)
            message(FATAL_ERROR "oneCCL is not installed, try changing HOROVOD_CPU_OPERATIONS.")
        endif()
        add_definitions(-DHOROVOD_CPU_OPERATIONS_DEFAULT=C)
    elseif(HOROVOD_CPU_OPERATIONS STREQUAL "GLOO")
        if(NOT HAVE_GLOO)
            message(FATAL_ERROR "Cannot set both HOROVOD_WITHOUT_GLOO and HOROVOD_CPU_OPERATIONS=GLOO.")
        endif()
        add_definitions(-DHOROVOD_CPU_OPERATIONS_DEFAULT=G)
    endif()
endif()

# Get Python suffix
execute_process(COMMAND ${PY_EXE} -c "import sysconfig; print(next(x for x in [sysconfig.get_config_var('EXT_SUFFIX'), sysconfig.get_config_var('SO'), '.so'] if x))"
                OUTPUT_VARIABLE Python_SUFFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

# TF
add_subdirectory(horovod/tensorflow)
# PyTorch
add_subdirectory(horovod/torch)
#MXNet
add_subdirectory(horovod/mxnet)

# Correctly wrap up json format
file(APPEND "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_ROOT}/metadata.json" "\"dummy\": \"none\"\n}")

# CUDA kernels
if(HAVE_CUDA OR HAVE_SUB_PROJECT_CUDA)
    add_subdirectory(horovod/common/ops/cuda)
endif()

if(HAVE_ROCM)
    add_subdirectory(horovod/common/ops/rocm)
endif()

# if we need compatible c++ abi
# Duplicate gloo folder and add it as a new sub-project
if(HAVE_GLOO AND ((DEFINED Tensorflow_CXX11 AND NOT Tensorflow_CXX11) OR (DEFINED Pytorch_CXX11 AND NOT Pytorch_CXX11) OR (DEFINED Mxnet_CXX11 AND NOT Mxnet_CXX11)))
    file(COPY ${PROJECT_SOURCE_DIR}/third_party/gloo/ DESTINATION ${PROJECT_SOURCE_DIR}/third_party/compatible_gloo)
    file(READ ${PROJECT_SOURCE_DIR}/third_party/compatible_gloo/gloo/CMakeLists.txt GLOO_CMAKE)
    string(REPLACE "gloo " "compatible_gloo " GLOO_CMAKE "${GLOO_CMAKE}")
    file(WRITE ${PROJECT_SOURCE_DIR}/third_party/compatible_gloo/gloo/CMakeLists.txt "${GLOO_CMAKE}")
    add_subdirectory(third_party/compatible_gloo)
    target_compile_definitions(compatible_gloo PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

# Gloo for c++17 TF
if(HAVE_GLOO AND (DEFINED Tensorflow_CXX17 AND Tensorflow_CXX17))
    file(COPY ${PROJECT_SOURCE_DIR}/third_party/gloo/ DESTINATION ${PROJECT_SOURCE_DIR}/third_party/compatible17_gloo)
    file(READ ${PROJECT_SOURCE_DIR}/third_party/compatible17_gloo/gloo/CMakeLists.txt GLOO_CMAKE)
    string(REPLACE "gloo " "compatible17_gloo " GLOO_CMAKE "${GLOO_CMAKE}")
    file(WRITE ${PROJECT_SOURCE_DIR}/third_party/compatible17_gloo/gloo/CMakeLists.txt "${GLOO_CMAKE}")
    file(READ ${PROJECT_SOURCE_DIR}/third_party/compatible17_gloo/CMakeLists.txt GLOO_CMAKE)
    string(REPLACE "-std=c++11" "-std=c++17" GLOO_CMAKE "${GLOO_CMAKE}")
    string(PREPEND GLOO_CMAKE "set(CMAKE_CXX_STANDARD 17)\n")
    file(WRITE ${PROJECT_SOURCE_DIR}/third_party/compatible17_gloo/CMakeLists.txt "${GLOO_CMAKE}")
    add_subdirectory(third_party/compatible17_gloo)
    if (Tensorflow_CXX11)
        target_compile_definitions(compatible17_gloo PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
    else()
       target_compile_definitions(compatible17_gloo PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
    endif()
endif()
