# Try to find Tensorflow
#
# The following are set after configuration is done:
#  TENSORFLOW_FOUND
#  Tensorflow_INCLUDE_DIRS
#  Tensorflow_LIBRARIES
#  Tensorflow_COMPILE_FLAGS
#  Tensorflow_VERSION
#  Tensorflow_CXX11

# Compatible layer for CMake <3.12. Tensorflow_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${Tensorflow_ROOT})

execute_process(COMMAND ${PY_EXE} -c "import tensorflow as tf; print(tf.__version__); print(tf.sysconfig.get_include()); print(' '.join(tf.sysconfig.get_link_flags())); print(' '.join(tf.sysconfig.get_compile_flags()))"
                OUTPUT_VARIABLE Tensorflow_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX REPLACE "\n" ";" Tensorflow_OUTPUT "${Tensorflow_OUTPUT}")
list(LENGTH Tensorflow_OUTPUT LEN)
if (LEN EQUAL "4")
    list(GET Tensorflow_OUTPUT 0 Tensorflow_VERSION)
    list(GET Tensorflow_OUTPUT 1 Tensorflow_INCLUDE_DIRS)
    list(GET Tensorflow_OUTPUT 2 Tensorflow_LIBRARIES)
    string(REPLACE " " ";" Tensorflow_LIBRARIES_LIST "${Tensorflow_LIBRARIES}")
    list(GET Tensorflow_LIBRARIES_LIST 0 Tensorflow_LIB_PATH_ARGUMENT)
    string(REGEX REPLACE "^-L" "" Tensorflow_LIB_PATH ${Tensorflow_LIB_PATH_ARGUMENT})
    if (Tensorflow_VERSION VERSION_GREATER "2.6" OR Tensorflow_VERSION VERSION_EQUAL "2.6")
#        # XLA implementations are in _pywrap_tensorflow_internal.so
        set(Tensorflow_LIBRARIES "${Tensorflow_LIBRARIES} ${Tensorflow_LIB_PATH}/python/_pywrap_tensorflow_internal.so")
    endif()
    message("Tensorflow_LIBRARIES := ${Tensorflow_LIBRARIES}")
    list(GET Tensorflow_OUTPUT 3 Tensorflow_COMPILE_FLAGS)
    if("${Tensorflow_COMPILE_FLAGS}" MATCHES "-D_GLIBCXX_USE_CXX11_ABI=1")
        set(Tensorflow_CXX11 TRUE)
    else()
        set(Tensorflow_CXX11 FALSE)
    endif()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tensorflow REQUIRED_VARS Tensorflow_LIBRARIES VERSION_VAR Tensorflow_VERSION)

mark_as_advanced(Tensorflow_INCLUDE_DIRS Tensorflow_LIBRARIES Tensorflow_COMPILE_FLAGS Tensorflow_VERSION Tensorflow_CXX11)
