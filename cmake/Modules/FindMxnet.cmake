# Try to find Mxnet
#
# The following are set after configuration is done:
#  MXNET_FOUND
#  Mxnet_INCLUDE_DIRS
#  Mxnet_LIBRARIES
#  Mxnet_COMPILE_FLAGS
#  Mxnet_USE_MKLDNN
#  Mxnet_USE_CUDA
#  Mxnet_VERSION

# Compatible layer for CMake <3.12. Mxnet_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${Mxnet_ROOT})
set(Mxnet_COMPILE_FLAGS "")

set(ENV{PYTHONPATH} "${PROJECT_SOURCE_DIR}/cmake:$ENV{PYTHONPATH}")
execute_process(COMMAND ${PY_EXE} -c "import os; import mxnet as mx; import build_utils; print(mx.__version__); print(mx.libinfo.find_include_path()); print(' '.join(mx.libinfo.find_lib_path())); print(build_utils.is_mx_mkldnn()); print(build_utils.is_mx_cuda())"
                OUTPUT_VARIABLE Mxnet_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX REPLACE "\n" ";" Mxnet_OUTPUT "${Mxnet_OUTPUT}")
list(LENGTH Mxnet_OUTPUT LEN)
if (LEN EQUAL "5")
    list(GET Mxnet_OUTPUT 0 Mxnet_VERSION)
    list(GET Mxnet_OUTPUT 1 Mxnet_INCLUDE_DIRS)
    list(GET Mxnet_OUTPUT 2 Mxnet_LIBRARIES)
    string(REPLACE " " ";" Mxnet_LIBRARIES "${Mxnet_LIBRARIES}")
    list(GET Mxnet_OUTPUT 3 Mxnet_USE_MKLDNN)
    list(GET Mxnet_OUTPUT 4 Mxnet_USE_CUDA)
    string(TOUPPER ${Mxnet_USE_MKLDNN} Mxnet_USE_MKLDNN)
    string(TOUPPER ${Mxnet_USE_CUDA} Mxnet_USE_CUDA)
    if (Mxnet_USE_MKLDNN AND EXISTS ${Mxnet_INCLUDE_DIRS}/mkldnn)
        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMXNET_USE_MKLDNN=1")
        list(APPEND Mxnet_INCLUDE_DIRS "${Mxnet_INCLUDE_DIRS}/mkldnn")
    else()
        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMXNET_USE_MKLDNN=0")
    endif()
    if (Mxnet_USE_CUDA)
        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMSHADOW_USE_CUDA=1")
    else()
        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMSHADOW_USE_CUDA=0")
    endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Mxnet REQUIRED_VARS Mxnet_LIBRARIES VERSION_VAR Mxnet_VERSION)

mark_as_advanced(Mxnet_INCLUDE_DIRS Mxnet_LIBRARIES Mxnet_COMPILE_FLAGS Mxnet_USE_MKLDNN Mxnet_USE_CUDA Mxnet_VERSION)
