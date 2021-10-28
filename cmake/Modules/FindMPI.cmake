# Try to find MPI
#
# The following variables are optionally searched for defaults
#  HOROVOD_MPI_HOME: Base directory where all MPI components are found
#  HOROVOD_MPI_INCLUDE: Directory where MPI header is found
#  HOROVOD_MPI_LIB: Directory where MPI library is found
#
# The following are set after configuration is done:
#  MPI_FOUND
#  MPI_INCLUDE_DIRS
#  MPI_LIBRARIES
#  MPI_MAJOR_VERSION
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install MPI in the same location as the CUDA toolkit.

set(HOROVOD_MPI_HOME $ENV{HOROVOD_MPI_HOME} CACHE PATH "Folder contains MPI")
if(NOT DEFINED HOROVOD_MPI_HOME)
    set(HOROVOD_MPI_HOME "/usr/lib64/openmpi")
endif()

list(APPEND MPI_ROOT ${HOROVOD_MPI_HOME})
# Compatible layer for CMake <3.12. MPI_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${MPI_ROOT})

find_path(MPI_INCLUDE_DIR
        NAMES mpi.h
        HINTS ${HOROVOD_MPI_HOME}/include)

find_library(MPI_LIBRARY
        NAMES mpi
        HINTS ${HOROVOD_MPI_HOME}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPI DEFAULT_MSG MPI_INCLUDE_DIR MPI_LIBRARY)

if (MPI_FOUND)
    set(MPI_HEADER_FILE "${MPI_INCLUDE_DIR}/mpi.h")
    message(STATUS "Determining MPI version from the header file: ${MPI_HEADER_FILE}")
    file (STRINGS ${MPI_HEADER_FILE} MPI_MAJOR_VERSION_DEFINED
            REGEX "^[ \t]*#define[ \t]+MPI_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
    if (MPI_MAJOR_VERSION_DEFINED)
        string (REGEX REPLACE "^[ \t]*#define[ \t]+MPI_MAJOR[ \t]+" ""
                MPI_MAJOR_VERSION ${MPI_MAJOR_VERSION_DEFINED})
        message(STATUS "MPI_MAJOR_VERSION: ${MPI_MAJOR_VERSION}")
    endif()
    set(MPI_INCLUDE_DIRS ${MPI_INCLUDE_DIR})
    set(MPI_LIBRARIES ${MPI_LIBRARY})
    message(STATUS "Found MPI (include: ${MPI_INCLUDE_DIRS}, library: ${MPI_LIBRARIES})")
    mark_as_advanced(MPI_INCLUDE_DIRS MPI_LIBRARIES)
endif()
