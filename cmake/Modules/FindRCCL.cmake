# Try to find RCCL
#
# The following variables are optionally searched for defaults
#  HOROVOD_RCCL_HOME: Base directory where all RCCL components are found
#  HOROVOD_RCCL_INCLUDE: Directory where RCCL header is found
#  HOROVOD_RCCL_LIB: Directory where RCCL library is found
#
# The following are set after configuration is done:
#  RCCL_FOUND
#  RCCL_INCLUDE_DIRS
#  RCCL_LIBRARIES
#  RCCL_MAJOR_VERSION

set(HOROVOD_RCCL_HOME $ENV{HOROVOD_NCCL_HOME} CACHE PATH "Folder contains AMD RCCL")
set(HOROVOD_RCCL_INCLUDE $ENV{HOROVOD_NCCL_INCLUDE} CACHE PATH "Folder contains AMD RCCL headers")
set(HOROVOD_RCCL_LIB $ENV{HOROVOD_NCCL_LIB} CACHE PATH "Folder contains AMD RCCL libraries")

list(APPEND RCCL_ROOT ${HOROVOD_RCCL_HOME})
# Compatible layer for CMake <3.12. RCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${RCCL_ROOT})

find_path(RCCL_INCLUDE_DIR
        NAMES rccl.h
        HINTS ${HOROVOD_RCCL_INCLUDE})

find_library(rccl
        NAMES ${RCCL_LIBNAME}
        HINTS ${HOROVOD_RCCL_LIB})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RCCL DEFAULT_MSG RCCL_INCLUDE_DIR RCCL_LIBRARY)

if (RCCL_FOUND)
    set(RCCL_HEADER_FILE "${RCCL_INCLUDE_DIR}/rccl.h")
    message(STATUS "Determining RCCL version from the header file: ${RCCL_HEADER_FILE}")
    file (STRINGS ${RCCL_HEADER_FILE} RCCL_MAJOR_VERSION_DEFINED
            REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
    if (RCCL_MAJOR_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
                RCCL_MAJOR_VERSION ${RCCL_MAJOR_VERSION_DEFINED})
        message(STATUS "RCCL_MAJOR_VERSION: ${RCCL_MAJOR_VERSION}")
    endif()
    set(RCCL_INCLUDE_DIRS ${RCCL_INCLUDE_DIR})
    set(RCCL_LIBRARIES ${RCCL_LIBRARY})
    message(STATUS "Found RCCL (include: ${RCCL_INCLUDE_DIRS}, library: ${RCCL_LIBRARIES})")
    mark_as_advanced(RCCL_INCLUDE_DIRS RCCL_LIBRARIES)
endif()
