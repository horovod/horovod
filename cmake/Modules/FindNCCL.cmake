# Try to find NCCL
#
# The following variables are optionally searched for defaults
#  HOROVOD_NCCL_HOME: Base directory where all NCCL components are found
#  HOROVOD_NCCL_INCLUDE: Directory where NCCL header is found
#  HOROVOD_NCCL_LIB: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#  NCCL_MAJOR_VERSION
#
# The path hints include CUDAToolkit_* seeing as some folks
# install NCCL in the same location as the CUDA toolkit.

set(HOROVOD_NCCL_HOME $ENV{HOROVOD_NCCL_HOME} CACHE PATH "Folder contains NVIDIA NCCL")
set(HOROVOD_NCCL_INCLUDE $ENV{HOROVOD_NCCL_INCLUDE} CACHE PATH "Folder contains NVIDIA NCCL headers")
set(HOROVOD_NCCL_LIB $ENV{HOROVOD_NCCL_LIB} CACHE PATH "Folder contains NVIDIA NCCL libraries")

list(APPEND NCCL_ROOT ${HOROVOD_NCCL_HOME} ${CUDAToolkit_LIBRARY_ROOT})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${HOROVOD_NCCL_INCLUDE} ${CUDAToolkit_INCLUDE_DIRS})

set(HOROVOD_NCCL_LINK $ENV{HOROVOD_NCCL_LINK})
if (HOROVOD_NCCL_LINK STREQUAL "SHARED")
    set(NCCL_LIBNAME "nccl")
    message(STATUS "Linking against shared NCCL library")
else()
    set(NCCL_LIBNAME "libnccl_static.a")
    message(STATUS "Linking against static NCCL library")
endif()

find_library(NCCL_LIBRARY
        NAMES ${NCCL_LIBNAME}
        HINTS ${HOROVOD_NCCL_LIB} ${CUDAToolkit_LIBRARY_DIR})

if (NCCL_LIBRARY STREQUAL "NCCL_LIBRARY-NOTFOUND" AND NCCL_LIBNAME MATCHES "static" AND
    NOT HOROVOD_NCCL_LINK STREQUAL "STATIC")
    message(STATUS "Could not find static NCCL library. Trying to find shared lib instead.")
    find_library(NCCL_LIBRARY
            NAMES "nccl"
            HINTS ${HOROVOD_NCCL_LIB} ${CUDAToolkit_LIBRARY_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if (NCCL_FOUND)
    set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIR}/nccl.h")
    message(STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}")
    file (STRINGS ${NCCL_HEADER_FILE} NCCL_MAJOR_VERSION_DEFINED
            REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
    if (NCCL_MAJOR_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
                NCCL_MAJOR_VERSION ${NCCL_MAJOR_VERSION_DEFINED})
        message(STATUS "NCCL_MAJOR_VERSION: ${NCCL_MAJOR_VERSION}")
    endif()
    set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
    set(NCCL_LIBRARIES ${NCCL_LIBRARY})
    message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
    mark_as_advanced(NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()
