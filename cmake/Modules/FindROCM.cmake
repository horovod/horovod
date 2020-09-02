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

set(HOROVOD_ROCM_HOME $ENV{HOROVOD_ROCM_HOME})
if(NOT DEFINED HOROVOD_ROCM_HOME)
    set(HOROVOD_ROCM_HOME "/opt/rocm")
endif()
set(HIP_PATH "${HOROVOD_ROCM_HOME}/hip")

list(APPEND ROCM_ROOT ${HOROVOD_ROCM_HOME})
# Compatible layer for CMake <3.12. ROCM_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT})

find_package(HIP REQUIRED)

if(HIP_FOUND)
  if(HIP_COMPILER STREQUAL clang)
    set(hip_library_name amdhip64)
  else()
    set(hip_library_name hip_hcc)
  endif()
  message(STATUS "HIP library name: ${hip_library_name}")
  find_library(ROCM_LIBRARIES NAMES ${hip_library_name} HINTS ${HIP_PATH}/lib)
endif()

set(ROCM_INCLUDE_DIRS ${HIP_INCLUDE_DIRS})
set(ROCM_COMPILE_FLAGS "-D__HIP_PLATFORM_HCC__=1")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ROCM DEFAULT_MSG ROCM_LIBRARIES)
