# Try to find Pytorch
#
# The following are set after configuration is done:
#  PYTORCH_FOUND
#  Pytorch_INCLUDE_DIRS
#  Pytorch_LIBRARY_DIRS
#  Pytorch_LIBRARIES
#  Pytorch_COMPILE_FLAGS
#  Pytorch_VERSION
#  Pytorch_CUDA
#  Pytorch_ROCM
#  Pytorch_CXX11

# Compatible layer for CMake <3.12. Pytorch_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${Pytorch_ROOT})

execute_process(COMMAND ${PY_EXE} -c "import torch; print(torch.__version__)"
                OUTPUT_VARIABLE Pytorch_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pytorch REQUIRED_VARS Pytorch_VERSION VERSION_VAR Pytorch_VERSION)
if(NOT PYTORCH_FOUND)
    return()
endif()

execute_process(COMMAND ${PY_EXE} -c "import torch; from torch.utils.cpp_extension import CUDA_HOME; print(True if ((torch.version.cuda is not None) and (CUDA_HOME is not None)) else False)"
                OUTPUT_VARIABLE Pytorch_CUDA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX REPLACE "No CUDA runtime[^\n]*\n?" "" Pytorch_CUDA "${Pytorch_CUDA}")
string(TOUPPER "${Pytorch_CUDA}" Pytorch_CUDA)

execute_process(COMMAND ${PY_EXE} -c "import torch; from torch.utils.cpp_extension import ROCM_HOME; print(True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False)"
                OUTPUT_VARIABLE Pytorch_ROCM OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX REPLACE "No CUDA runtime[^\n]*\n?" "" Pytorch_ROCM "${Pytorch_ROCM}")
string(TOUPPER "${Pytorch_ROCM}" Pytorch_ROCM)

if(Pytorch_ROCM)
    execute_process(COMMAND ${PY_EXE} -c "from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS; print(' '.join(COMMON_HIPCC_FLAGS))"
                    OUTPUT_VARIABLE _Pytorch_ROCM_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX REPLACE "No CUDA runtime[^\n]*\n?" "" _Pytorch_ROCM_FLAGS "${_Pytorch_ROCM_FLAGS}")
    set(Pytorch_COMPILE_FLAGS "${_Pytorch_ROCM_FLAGS}")
endif()

if (Pytorch_CUDA OR Pytorch_ROCM)
    set(Pytorch_EXT "CUDAExtension")
else()
    set(Pytorch_EXT "CppExtension")
endif()

execute_process(COMMAND ${PY_EXE} -c "import os,sys; sys.stdout=sys.stderr=open(os.devnull,'w'); from torch.utils.cpp_extension import ${Pytorch_EXT} as ext; e = ext('', []); print(';'.join(e.include_dirs), file=sys.__stdout__)"
                OUTPUT_VARIABLE Pytorch_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PY_EXE} -c "import os,sys; sys.stdout=sys.stderr=open(os.devnull,'w'); from torch.utils.cpp_extension import ${Pytorch_EXT} as ext; e = ext('', []); print(';'.join(e.library_dirs), file=sys.__stdout__)"
                OUTPUT_VARIABLE Pytorch_LIBRARY_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PY_EXE} -c "import os,sys; sys.stdout=sys.stderr=open(os.devnull,'w'); from torch.utils.cpp_extension import ${Pytorch_EXT} as ext; e = ext('', []); print(';'.join(e.libraries), file=sys.__stdout__)"
                OUTPUT_VARIABLE _Pytorch_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)

foreach(_LIB IN LISTS _Pytorch_LIBRARIES)
    find_library(FOUND_LIB_${_LIB}
            NAMES ${_LIB}
            HINTS ${Pytorch_LIBRARY_DIRS})
    list(APPEND Pytorch_LIBRARIES ${FOUND_LIB_${_LIB}})
endforeach()

execute_process(COMMAND ${PY_EXE} -c "import torch; print(torch.torch.compiled_with_cxx11_abi())"
                OUTPUT_VARIABLE Pytorch_CXX11 OUTPUT_STRIP_TRAILING_WHITESPACE)
string(TOUPPER ${Pytorch_CXX11} Pytorch_CXX11)
if (Pytorch_CXX11)
    set(Pytorch_COMPILE_FLAGS "${Pytorch_COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(Pytorch_COMPILE_FLAGS "${Pytorch_COMPILE_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mark_as_advanced(Pytorch_INCLUDE_DIRS Pytorch_LIBRARY_DIRS Pytorch_LIBRARIES Pytorch_COMPILE_FLAGS Pytorch_VERSION Pytorch_CUDA Pytorch_ROCM Pytorch_CXX11)
