if("$ENV{HOROVOD_WITHOUT_PYTORCH}" STREQUAL "1")
    return()
endif()

set(PYTORCH_TARGET_LIB "pytorch")

# Find PyTorch
set(PYTORCH_REQUIRED "")
if ("$ENV{HOROVOD_WITH_PYTORCH}" STREQUAL "1")
    set(PYTORCH_REQUIRED "REQUIRED")
endif ()
find_package(Pytorch "1.2.0" ${PYTORCH_REQUIRED})
if(NOT PYTORCH_FOUND)
    return()
endif()

# Append version number into metadata
file(APPEND "${CMAKE_LIBRARY_OUTPUT_DIRECTORY_ROOT}/metadata.json" "\"pytorch\": \"${Pytorch_VERSION}\",\n")

if (HAVE_CUDA AND NOT Pytorch_CUDA)
    message(FATAL_ERROR "Horovod build with GPU support was requested but this PyTorch installation does not support CUDA.")
elseif (Pytorch_CUDA AND NOT HAVE_CUDA)
    add_cuda()
endif()
if (HAVE_ROCM AND NOT Pytorch_ROCM)
    message(FATAL_ERROR "Horovod build with GPU support was requested but this PyTorch installation does not support ROCm.")
elseif (Pytorch_ROCM AND NOT HAVE_ROCM)
    add_definitions(-DHAVE_ROCM=1 -DHAVE_GPU=1)
endif()
if (Pytorch_ROCM)
    # Hipify extension code
    execute_process(COMMAND ${PY_EXE} -c "from torch.utils.hipify import hipify_python; hipify_python.hipify(project_directory='${PROJECT_SOURCE_DIR}/horovod', output_directory='${PROJECT_SOURCE_DIR}/horovod', includes=('torch/*.cc','torch/*.h'), show_detailed=True, is_pytorch_extension=True)")
    execute_process(COMMAND ${PY_EXE} -c "from torch.utils import hipify; print(True) if hasattr(hipify, '__version__') else print(False)" OUTPUT_VARIABLE HIPIFY_HAS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

include_directories(SYSTEM ${Pytorch_INCLUDE_DIRS})
execute_process(COMMAND ${PY_EXE} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
                OUTPUT_VARIABLE PYTHON_INCLUDE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(SYSTEM ${PYTHON_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${Pytorch_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
list(APPEND PYTORCH_LINKER_LIBS ${Pytorch_LIBRARIES})
if(HAVE_GLOO)
    if (Pytorch_CXX11)
        list(APPEND PYTORCH_LINKER_LIBS gloo)
    else()
        list(APPEND PYTORCH_LINKER_LIBS compatible_gloo)
    endif()
endif()
if(HAVE_CUDA)
    if (Pytorch_CXX11)
        list(APPEND PYTORCH_LINKER_LIBS horovod_cuda_kernels)
    else()
        list(APPEND PYTORCH_LINKER_LIBS compatible_horovod_cuda_kernels)
    endif()
endif()
parse_version(${Pytorch_VERSION} VERSION_DEC)
add_definitions(-DTORCH_VERSION=${VERSION_DEC} -DTORCH_API_INCLUDE_EXTENSION_H=1)
set(Pytorch_CXX11 ${Pytorch_CXX11} PARENT_SCOPE)
if(NOT Pytorch_VERSION VERSION_LESS "1.3.0")
    set(CMAKE_CXX_STANDARD 14)
endif()

# PyTorch SOURCES
# Later versions of PyTorch that use ROCm's hipify step will rename files.
if(Pytorch_ROCM AND "${HIPIFY_HAS_VERSION}" STREQUAL "True")
list(APPEND PYTORCH_SOURCES "${PROJECT_SOURCE_DIR}/horovod/torch/handle_manager.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/ready_event_hip.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/hip_util.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/mpi_ops_v2_hip.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/adapter_v2.cc")
else()
list(APPEND PYTORCH_SOURCES "${PROJECT_SOURCE_DIR}/horovod/torch/handle_manager.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/ready_event.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/cuda_util.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/mpi_ops_v2.cc"
                            "${PROJECT_SOURCE_DIR}/horovod/torch/adapter_v2.cc")
endif()

# Create library
set_output_dir()
add_library(${PYTORCH_TARGET_LIB} SHARED ${SOURCES} ${PYTORCH_SOURCES})
target_include_directories(${PYTORCH_TARGET_LIB} PRIVATE "${EIGEN_INCLUDE_PATH}")
target_include_directories(${PYTORCH_TARGET_LIB} PRIVATE "${FLATBUFFERS_INCLUDE_PATH}")
target_link_libraries(${PYTORCH_TARGET_LIB} ${LINKER_LIBS} ${PYTORCH_LINKER_LIBS})
set_target_properties(${PYTORCH_TARGET_LIB} PROPERTIES SUFFIX "${Python_SUFFIX}")
set_target_properties(${PYTORCH_TARGET_LIB} PROPERTIES PREFIX "")
set_target_properties(${PYTORCH_TARGET_LIB} PROPERTIES OUTPUT_NAME "mpi_lib_v2")
