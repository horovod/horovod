if("$ENV{HOROVOD_WITHOUT_MXNET}" STREQUAL "1")
    return()
endif()

set(Mxnet_TARGET_LIB "mxnet")

# Find MXNet
set(Mxnet_REQUIRED "")
if ("$ENV{HOROVOD_WITH_MXNET}" STREQUAL "1")
    set(Mxnet_REQUIRED "REQUIRED")
endif ()
find_package(Mxnet "1.4.1" ${Mxnet_REQUIRED})
if(NOT MXNET_FOUND)
    return()
endif()
set(Mxnet_CXX11 ${Mxnet_CXX11} PARENT_SCOPE)

# Append version number into metadata
file(APPEND "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_ROOT}/metadata.json" "\"mxnet\": \"${Mxnet_VERSION}\",\n")

if (HAVE_CUDA AND NOT Mxnet_USE_CUDA)
    message(FATAL_ERROR "Horovod build with GPU support was requested but this MXNet installation does not support CUDA.")
elseif (Mxnet_USE_CUDA AND NOT HAVE_CUDA)
    add_cuda()
endif()
include_directories(SYSTEM ${Mxnet_INCLUDE_DIRS})
list(APPEND Mxnet_LINKER_LIBS ${Mxnet_LIBRARIES})
if(HAVE_GLOO)
    if (Mxnet_CXX11)
        list(APPEND Mxnet_LINKER_LIBS gloo)
    else()
        list(APPEND Mxnet_LINKER_LIBS compatible_gloo)
    endif()
endif()
if(HAVE_CUDA)
    if (Mxnet_CXX11)
        list(APPEND Mxnet_LINKER_LIBS horovod_cuda_kernels)
    else()
        list(APPEND Mxnet_LINKER_LIBS compatible_horovod_cuda_kernels)
    endif()
endif()
set(CMAKE_CXX_FLAGS "${Mxnet_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMSHADOW_USE_MKL=0 -DMSHADOW_USE_F16C=0")

# MXNet SOURCES
list(APPEND Mxnet_SOURCES "${PROJECT_SOURCE_DIR}/horovod/mxnet/mpi_ops.cc"
                          "${PROJECT_SOURCE_DIR}/horovod/mxnet/tensor_util.cc"
                          "${PROJECT_SOURCE_DIR}/horovod/mxnet/cuda_util.cc"
                          "${PROJECT_SOURCE_DIR}/horovod/mxnet/adapter.cc")

# Create library
set_output_dir()
add_library(${Mxnet_TARGET_LIB} SHARED ${SOURCES} ${Mxnet_SOURCES})
target_include_directories(${Mxnet_TARGET_LIB} PRIVATE "${EIGEN_INCLUDE_PATH}")
target_include_directories(${Mxnet_TARGET_LIB} PRIVATE "${FLATBUFFERS_INCLUDE_PATH}")
target_link_libraries(${Mxnet_TARGET_LIB} ${LINKER_LIBS} ${Mxnet_LINKER_LIBS})
set_target_properties(${Mxnet_TARGET_LIB} PROPERTIES SUFFIX "${Python_SUFFIX}")
set_target_properties(${Mxnet_TARGET_LIB} PROPERTIES PREFIX "")
set_target_properties(${Mxnet_TARGET_LIB} PROPERTIES OUTPUT_NAME "mpi_lib")
