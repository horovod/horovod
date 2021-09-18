# Try to find Mxnet
#
# The following are set after configuration is done:
#  MXNET_FOUND
#  Mxnet_INCLUDE_DIRS
#  Mxnet_LIBRARIES
#  Mxnet_COMPILE_FLAGS
#  Mxnet_USE_MKLDNN
#  Mxnet_USE_ONEDNN
#  Mxnet_USE_CUDA
#  Mxnet_VERSION
#  Mxnet_CXX11

# Compatible layer for CMake <3.12. Mxnet_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${Mxnet_ROOT})
set(Mxnet_COMPILE_FLAGS "")

set(ENV{PYTHONPATH} "${PROJECT_SOURCE_DIR}/cmake:$ENV{PYTHONPATH}")
execute_process(COMMAND ${PY_EXE} -c "import os; import mxnet as mx; import build_utils; print(mx.__version__); print(mx.libinfo.find_include_path()); print(' '.join(mx.libinfo.find_lib_path())); print(build_utils.is_mx_mkldnn()); print(build_utils.is_mx_onednn()); print(build_utils.is_mx_cuda())"
                OUTPUT_VARIABLE Mxnet_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX REPLACE "\n" ";" Mxnet_OUTPUT "${Mxnet_OUTPUT}")
list(LENGTH Mxnet_OUTPUT LEN)
if (LEN EQUAL "6")
    list(GET Mxnet_OUTPUT 0 Mxnet_VERSION)
    list(GET Mxnet_OUTPUT 1 Mxnet_INCLUDE_DIRS)
    list(GET Mxnet_OUTPUT 2 Mxnet_LIBRARIES)
    string(REPLACE " " ";" Mxnet_LIBRARIES "${Mxnet_LIBRARIES}")
    list(GET Mxnet_OUTPUT 3 Mxnet_USE_MKLDNN)
    list(GET Mxnet_OUTPUT 4 Mxnet_USE_ONEDNN)
    list(GET Mxnet_OUTPUT 5 Mxnet_USE_CUDA)
    string(TOUPPER ${Mxnet_USE_MKLDNN} Mxnet_USE_MKLDNN)
    string(TOUPPER ${Mxnet_USE_ONEDNN} Mxnet_USE_ONEDNN)
    string(TOUPPER ${Mxnet_USE_CUDA} Mxnet_USE_CUDA)
    if (Mxnet_USE_MKLDNN OR Mxnet_USE_ONEDNN)
        if (Mxnet_USE_MKLDNN AND EXISTS ${Mxnet_INCLUDE_DIRS}/mkldnn)
            set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMXNET_USE_MKLDNN=1 -DMXNET_USE_ONEDNN=0")
            list(APPEND Mxnet_INCLUDE_DIRS "${Mxnet_INCLUDE_DIRS}/mkldnn")
        elseif (Mxnet_USE_ONEDNN AND EXISTS ${Mxnet_INCLUDE_DIRS}/onednn)
            set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMXNET_USE_MKLDNN=0 -DMXNET_USE_ONEDNN=1")
            list(APPEND Mxnet_INCLUDE_DIRS "${Mxnet_INCLUDE_DIRS}/onednn")
        else()
            if (Mxnet_FIND_REQUIRED)
                set(MSG_LEVEL "FATAL_ERROR")
            else()
                set(MSG_LEVEL "WARNING")
            endif()
            set(MXNET_FOUND FALSE)
            if (Mxnet_USE_MKLDNN)
                message(${MSG_LEVEL} "MXNet was found with mkl-dnn support but mkldnn header files are missing. Please, install MXNet with mkldnn header files.")
            elseif (Mxnet_USE_ONEDNN)
                message(${MSG_LEVEL} "MXNet was found with onednn support but onednn header files are missing. Please, install MXNet with onednn header files.")
            endif()
            return()
        endif()
    else()
        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMXNET_USE_MKLDNN=0 -DMXNET_USE_ONEDNN=0")
    endif()
    if (Mxnet_USE_CUDA)
        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMSHADOW_USE_CUDA=1")
    else()
        set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -DMSHADOW_USE_CUDA=0")
    endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Mxnet REQUIRED_VARS Mxnet_LIBRARIES VERSION_VAR Mxnet_VERSION)
if(NOT MXNET_FOUND)
    return()
endif()


execute_process(COMMAND ${PY_EXE} -c "import mxnet; print(mxnet.library.compiled_with_gcc_cxx11_abi() if hasattr(mxnet, 'library') and hasattr(mxnet.library, 'compiled_with_gcc_cxx11_abi') else 1)"
  OUTPUT_VARIABLE Mxnet_CXX11 OUTPUT_STRIP_TRAILING_WHITESPACE)
string(TOUPPER ${Mxnet_CXX11} Mxnet_CXX11)
if (Mxnet_CXX11)
  set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
  set(Mxnet_COMPILE_FLAGS "${Mxnet_COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mark_as_advanced(Mxnet_INCLUDE_DIRS Mxnet_LIBRARIES Mxnet_COMPILE_FLAGS Mxnet_USE_MKLDNN Mxnet_USE_ONEDNN Mxnet_USE_CUDA Mxnet_VERSION)
