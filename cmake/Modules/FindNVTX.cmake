# Try to find NVTX
#
# NVTX comes with the CUDA toolkit so we use that root dir to search for the header-only variation of NVTX.
# Alternatively an explicit path can be given via the variable HOROVOD_NVTX_INCLUDE
#
# The following are set after configuration is done:
#  NVTX_FOUND
#  NVTX_INCLUDE_DIRS
#  NVTX_LIBRARIES

set(HOROVOD_NVTX_INCLUDE $ENV{HOROVOD_NVTX_INCLUDE} CACHE PATH "Folder containing NVIDIA NVTX3 headers")

list(APPEND NVTX_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12:
list(APPEND CMAKE_PREFIX_PATH ${NVTX_ROOT})

find_path(NVTX_INCLUDE_DIR
        NAMES nvtx3/nvToolsExt.h
        HINTS ${HOROVOD_NVTX_INCLUDE})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVTX DEFAULT_MSG NVTX_INCLUDE_DIR)

if (NVTX_FOUND)
    set(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR})
    # -ldl for dlopen, dlclose:
    set(NVTX_LIBRARIES ${CMAKE_DL_LIBS})
    message(STATUS "Found NVTX (include: ${NVTX_INCLUDE_DIRS}, library: ${NVTX_LIBRARIES})")
    mark_as_advanced(NVTX_INCLUDE_DIRS NVTX_LIBRARIES)
endif()
