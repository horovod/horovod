# Try to find ROCM
#
# The following variables are optionally searched for defaults
#  HOROVOD_ROCM_HOME: Base directory where all ROCM components are found
#
# The following are set after configuration is done:
#  ROCM_FOUND
#  ROCM_INCLUDE_DIRS
#  ROCM_LIBRARIES
#  ROCM_COMPILE_FLAGS

set(HOROVOD_ROCM_HOME $ENV{HOROVOD_ROCM_HOME} CACHE PATH "Folder containing ROCM")
if(NOT DEFINED HOROVOD_ROCM_HOME)
    set(HOROVOD_ROCM_HOME "/opt/rocm")
endif()

list(APPEND ROCM_ROOT ${HOROVOD_ROCM_HOME})
# Compatible layer for CMake <3.12. ROCM_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT})

set(ROCM_INCLUDE_DIRS "${HOROVOD_ROCM_HOME}/include" "${HOROVOD_ROCM_HOME}/hcc/include"
                      "${HOROVOD_ROCM_HOME}/hip/include" "${HOROVOD_ROCM_HOME}/hsa/include")

find_library(ROCM_LIBRARIES
             NAMES hip_hcc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCM DEFAULT_MSG ROCM_LIBRARIES)

set(ROCM_COMPILE_FLAGS "-D__HIP_PLATFORM_HCC__=1")

mark_as_advanced(ROCM_INCLUDE_DIRS ROCM_LIBRARIES ROCM_COMPILE_FLAGS)