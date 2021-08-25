if("$ENV{HOROVOD_WITHOUT_TENSORFLOW}" STREQUAL "1")
    return()
endif()

set(TF_TARGET_LIB "tensorflow")

# Find TF
set(TF_REQUIRED "")
if ("$ENV{HOROVOD_WITH_TENSORFLOW}" STREQUAL "1")
    set(TF_REQUIRED "REQUIRED")
endif ()
find_package(Tensorflow "1.15.0" ${TF_REQUIRED})
if(NOT TENSORFLOW_FOUND)
    return()
endif()

# Append version number into metadata
file(APPEND "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_ROOT}/metadata.json" "\"tensorflow\": \"${Tensorflow_VERSION}\",\n")

# Overwite Eigen and Flatbuffers headers path if provided in tensorflow or system
# Search tensorflow path
find_path(TF_EIGEN_INCLUDE_PATH Eigen/src/Core/EigenBase.h PATHS ${Tensorflow_INCLUDE_DIRS} NO_DEFAULT_PATH)
if (TF_EIGEN_INCLUDE_PATH STREQUAL "TF_EIGEN_INCLUDE_PATH-NOTFOUND")
    # If not found, search system path and submodule path
    find_path(TF_EIGEN_INCLUDE_PATH Eigen/src/Core/EigenBase.h PATHS ${EIGEN_INCLUDE_PATH} REQUIRED)
endif()
set(EIGEN_INCLUDE_PATH "${TF_EIGEN_INCLUDE_PATH}" PARENT_SCOPE)
set(EIGEN_INCLUDE_PATH "${TF_EIGEN_INCLUDE_PATH}")

# Search tensorflow path
find_path(TF_FLATBUFFERS_INCLUDE_PATH flatbuffers/flatbuffers.h PATHS ${Tensorflow_INCLUDE_DIRS} NO_DEFAULT_PATH)
if (TF_FLATBUFFERS_INCLUDE_PATH STREQUAL "TF_FLATBUFFERS_INCLUDE_PATH-NOTFOUND")
    # If not found, search system path and submodule path
    find_path(TF_FLATBUFFERS_INCLUDE_PATH flatbuffers/flatbuffers.h PATHS ${FLATBUFFERS_INCLUDE_PATH} REQUIRED)
endif()
set(FLATBUFFERS_INCLUDE_PATH "${TF_FLATBUFFERS_INCLUDE_PATH}" PARENT_SCOPE)
set(FLATBUFFERS_INCLUDE_PATH "${TF_FLATBUFFERS_INCLUDE_PATH}")

include_directories(SYSTEM ${Tensorflow_INCLUDE_DIRS})
list(APPEND TF_LINKER_LIBS ${Tensorflow_LIBRARIES})
if(HAVE_GLOO)
    if (Tensorflow_CXX11)
        list(APPEND TF_LINKER_LIBS gloo)
    else()
        list(APPEND TF_LINKER_LIBS compatible_gloo)
    endif()
endif()
if(HAVE_CUDA)
    if (Tensorflow_CXX11)
        list(APPEND TF_LINKER_LIBS horovod_cuda_kernels)
    else()
        list(APPEND TF_LINKER_LIBS compatible_horovod_cuda_kernels)
    endif()
endif()
set(CMAKE_CXX_FLAGS "${Tensorflow_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
parse_version(${Tensorflow_VERSION} VERSION_DEC)
add_definitions(-DTENSORFLOW_VERSION=${VERSION_DEC})
set(Tensorflow_CXX11 ${Tensorflow_CXX11} PARENT_SCOPE)

# TF SOURCES
list(APPEND TF_SOURCES "${PROJECT_SOURCE_DIR}/horovod/tensorflow/mpi_ops.cc")
list(APPEND TF_SOURCES "${PROJECT_SOURCE_DIR}/horovod/tensorflow/xla_mpi_ops.cc")

# Create library
set_output_dir()
add_library(${TF_TARGET_LIB} SHARED ${SOURCES} ${TF_SOURCES})
if (NOT EIGEN_INCLUDE_PATH STREQUAL Tensorflow_INCLUDE_DIRS)
    target_include_directories(${TF_TARGET_LIB} PRIVATE ${EIGEN_INCLUDE_PATH})
endif()
if (NOT FLATBUFFERS_INCLUDE_PATH STREQUAL Tensorflow_INCLUDE_DIRS)
    target_include_directories(${TF_TARGET_LIB} PRIVATE ${FLATBUFFERS_INCLUDE_PATH})
endif()
target_link_libraries(${TF_TARGET_LIB} ${LINKER_LIBS} ${TF_LINKER_LIBS})
set_target_properties(${TF_TARGET_LIB} PROPERTIES SUFFIX "${Python_SUFFIX}")
set_target_properties(${TF_TARGET_LIB} PROPERTIES PREFIX "")
set_target_properties(${TF_TARGET_LIB} PROPERTIES OUTPUT_NAME "mpi_lib")
