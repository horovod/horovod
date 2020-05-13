# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright Microsoft
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

import os
import re
import shlex
import subprocess
import sys
import textwrap
import traceback
import pipes
import warnings
from copy import deepcopy
from distutils.errors import CompileError, DistutilsError, \
    DistutilsPlatformError, LinkError
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from horovod import __version__
from horovod.common.util import env


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


tensorflow_mpi_lib = Extension('horovod.tensorflow.mpi_lib', [])
torch_mpi_lib = Extension('horovod.torch.mpi_lib', [])
torch_mpi_lib_impl = Extension('horovod.torch.mpi_lib_impl', [])
torch_mpi_lib_v2 = Extension('horovod.torch.mpi_lib_v2', [])
mxnet_mpi_lib = Extension('horovod.mxnet.mpi_lib', [])
gloo_lib = CMakeExtension('gloo', cmake_lists_dir='third_party/gloo',
                          sources=[])

ccl_root = os.environ.get('CCL_ROOT')
have_ccl = ccl_root is not None


def is_build_action():
    if len(sys.argv) <= 1:
        return False

    if sys.argv[1].startswith('build'):
        return True

    if sys.argv[1].startswith('bdist'):
        return True

    if sys.argv[1].startswith('install'):
        return True


def check_tf_version():
    try:
        import tensorflow as tf
        if LooseVersion(tf.__version__) < LooseVersion('1.1.0'):
            raise DistutilsPlatformError(
                'Your TensorFlow version %s is outdated.  '
                'Horovod requires tensorflow>=1.1.0' % tf.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import tensorflow failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        # This means that tf.__version__ was not exposed, which makes it *REALLY* old.
        raise DistutilsPlatformError(
            'Your TensorFlow version is outdated.  Horovod requires tensorflow>=1.1.0')


def check_mx_version():
    try:
        import mxnet as mx
        if mx.__version__ < '1.4.0':
            raise DistutilsPlatformError(
                'Your MXNet version %s is outdated.  '
                'Horovod requires mxnet>=1.4.0' % mx.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import mxnet failed, is it installed?\n\n%s' % traceback.format_exc())
    except AttributeError:
        raise DistutilsPlatformError(
            'Your MXNet version is outdated.  Horovod requires mxnet>1.3.0')


def get_supported_instruction_set_flags(flags_to_check):
    supported = []
    try:
        flags_output = subprocess.check_output(
            'gcc -march=native -E -v - </dev/null 2>&1 | grep cc1',
            shell=True, universal_newlines=True).strip()
        flags = shlex.split(flags_output)
        supported = [x for x in flags_to_check if x in flags or x.replace('-m', '+') in flags]
    except subprocess.CalledProcessError:
        # Fallback to no advanced instruction set flags if were not able to get flag information.
        pass
    return supported


def get_cpp_flags(build_ext):
    last_err = None
    default_flags = ['-std=c++11', '-fPIC', '-O3', '-Wall', '-fassociative-math', '-ffast-math', '-ftree-vectorize', '-funsafe-math-optimizations']
    build_arch_flags_env = os.environ.get('HOROVOD_BUILD_ARCH_FLAGS')
    build_arch_flags = get_supported_instruction_set_flags(['-mf16c', '-mavx', '-mfma']) if build_arch_flags_env is None else build_arch_flags_env.split()
    if sys.platform == 'darwin':
        # Darwin most likely will have Clang, which has libc++.
        flags_to_try = [default_flags + ['-stdlib=libc++'] + build_arch_flags,
                        default_flags + build_arch_flags,
                        default_flags + ['-stdlib=libc++'],
                        default_flags]
    else:
        flags_to_try = [default_flags + build_arch_flags,
                        default_flags + ['-stdlib=libc++'] + build_arch_flags,
                        default_flags,
                        default_flags + ['-stdlib=libc++']]
    for cpp_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cpp_flags',
                         extra_compile_preargs=cpp_flags,
                         code=textwrap.dedent('''\
                    #include <unordered_map>
                    void test() {
                    }
                    '''))

            return cpp_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ compilation flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ compilation flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_link_flags(build_ext):
    last_err = None
    libtool_flags = ['-Wl,-exported_symbols_list,horovod.exp']
    ld_flags = ['-Wl,--version-script=horovod.lds']
    if sys.platform == 'darwin':
        flags_to_try = [libtool_flags, ld_flags]
    else:
        flags_to_try = [ld_flags, libtool_flags]
    for link_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_link_flags',
                         extra_link_preargs=link_flags,
                         code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            return link_flags
        except (CompileError, LinkError):
            last_err = 'Unable to determine C++ link flags (see error above).'
        except Exception:
            last_err = 'Unable to determine C++ link flags.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_include_dirs():
    import tensorflow as tf
    tf_inc = tf.sysconfig.get_include()
    return [tf_inc, '%s/external/nsync/public' % tf_inc]


def get_tf_lib_dirs():
    import tensorflow as tf
    tf_lib = tf.sysconfig.get_lib()
    return [tf_lib]


def get_tf_libs(build_ext, lib_dirs, cpp_flags):
    last_err = None
    for tf_libs in [['tensorflow_framework'], []]:
        try:
            lib_file = test_compile(build_ext, 'test_tensorflow_libs',
                                    library_dirs=lib_dirs, libraries=tf_libs,
                                    extra_compile_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return tf_libs
        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine -l link flags to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_abi(build_ext, include_dirs, lib_dirs, libs, cpp_flags):
    last_err = None
    cxx11_abi_macro = '_GLIBCXX_USE_CXX11_ABI'
    for cxx11_abi in ['0', '1']:
        try:
            lib_file = test_compile(build_ext, 'test_tensorflow_abi',
                                    macros=[(cxx11_abi_macro, cxx11_abi)],
                                    include_dirs=include_dirs,
                                    library_dirs=lib_dirs,
                                    libraries=libs,
                                    extra_compile_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                #include <string>
                #include "tensorflow/core/framework/op.h"
                #include "tensorflow/core/framework/op_kernel.h"
                #include "tensorflow/core/framework/shape_inference.h"
                void test() {
                    auto ignore = tensorflow::strings::StrCat("a", "b");
                }
                '''))

            from tensorflow.python.framework import load_library
            load_library.load_op_library(lib_file)

            return cxx11_abi_macro, cxx11_abi
        except (CompileError, LinkError):
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow (see error above).'
        except Exception:
            last_err = 'Unable to determine CXX11 ABI to use with TensorFlow.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_tf_flags(build_ext, cpp_flags):
    import tensorflow as tf
    try:
        return tf.sysconfig.get_compile_flags(), tf.sysconfig.get_link_flags()
    except AttributeError:
        # fallback to the previous logic
        tf_include_dirs = get_tf_include_dirs()
        tf_lib_dirs = get_tf_lib_dirs()
        tf_libs = get_tf_libs(build_ext, tf_lib_dirs, cpp_flags)
        tf_abi = get_tf_abi(build_ext, tf_include_dirs,
                            tf_lib_dirs, tf_libs, cpp_flags)

        compile_flags = []
        for include_dir in tf_include_dirs:
            compile_flags.append('-I%s' % include_dir)
        if tf_abi:
            compile_flags.append('-D%s=%s' % tf_abi)

        link_flags = []
        for lib_dir in tf_lib_dirs:
            link_flags.append('-L%s' % lib_dir)
        for lib in tf_libs:
            link_flags.append('-l%s' % lib)

        return compile_flags, link_flags


def get_mx_include_dirs():
    import mxnet as mx
    return [mx.libinfo.find_include_path()]


def get_mx_lib_dirs():
    import mxnet as mx
    mx_libs = mx.libinfo.find_lib_path()
    mx_lib_dirs = [os.path.dirname(mx_lib) for mx_lib in mx_libs]
    return mx_lib_dirs


def get_mx_libs(build_ext, lib_dirs, cpp_flags):
    last_err = None
    for mx_libs in [['mxnet'], []]:
        try:
            lib_file = test_compile(build_ext, 'test_mx_libs',
                                    library_dirs=lib_dirs, libraries=mx_libs,
                                    extra_compile_preargs=cpp_flags,
                                    code=textwrap.dedent('''\
                    void test() {
                    }
                    '''))

            return mx_libs
        except (CompileError, LinkError):
            last_err = 'Unable to determine -l link flags to use with MXNet (see error above).'
        except Exception:
            last_err = 'Unable to determine -l link flags to use with MXNet.  ' \
                       'Last error:\n\n%s' % traceback.format_exc()

    raise DistutilsPlatformError(last_err)


def get_mx_flags(build_ext, cpp_flags):
    mx_include_dirs = get_mx_include_dirs()
    mx_lib_dirs = get_mx_lib_dirs()
    mx_libs = get_mx_libs(build_ext, mx_lib_dirs, cpp_flags)

    compile_flags = []
    has_mkldnn = is_mx_mkldnn()
    for include_dir in mx_include_dirs:
        compile_flags.append('-I%s' % include_dir)
        if has_mkldnn:
            mkldnn_include = os.path.join(include_dir, 'mkldnn')
            compile_flags.append('-I%s' % mkldnn_include)

    link_flags = []
    for lib_dir in mx_lib_dirs:
        link_flags.append('-Wl,-rpath,%s' % lib_dir)
        link_flags.append('-L%s' % lib_dir)

    for lib in mx_libs:
        link_flags.append('-l%s' % lib)

    return compile_flags, link_flags


def get_mpi_flags():
    show_command = os.environ.get('HOROVOD_MPICXX_SHOW', 'mpicxx -show')
    try:
        mpi_show_output = subprocess.check_output(
            shlex.split(show_command), universal_newlines=True).strip()
        mpi_show_args = shlex.split(mpi_show_output)
        if not mpi_show_args[0].startswith('-'):
            # Open MPI and MPICH print compiler name as a first word, skip it
            mpi_show_args = mpi_show_args[1:]
        # strip off compiler call portion and always escape each arg
        return ' '.join(['"' + arg.replace('"', '"\'"\'"') + '"'
                         for arg in mpi_show_args])
    except Exception:
        raise DistutilsPlatformError(
            '%s failed (see error below), is MPI in $PATH?\n'
            'Note: If your version of MPI has a custom command to show compilation flags, '
            'please specify it with the HOROVOD_MPICXX_SHOW environment variable.\n\n'
            '%s' % (show_command, traceback.format_exc()))


def test_compile(build_ext, name, code, libraries=None, include_dirs=None,
                 library_dirs=None,
                 macros=None, extra_compile_preargs=None,
                 extra_link_preargs=None):
    test_compile_dir = os.path.join(build_ext.build_temp, 'test_compile')
    if not os.path.exists(test_compile_dir):
        os.makedirs(test_compile_dir)

    source_file = os.path.join(test_compile_dir, '%s.cc' % name)
    with open(source_file, 'w') as f:
        f.write(code)

    compiler = build_ext.compiler
    [object_file] = compiler.object_filenames([source_file])
    shared_object_file = compiler.shared_object_filename(
        name, output_dir=test_compile_dir)

    compiler.compile([source_file], extra_preargs=extra_compile_preargs,
                     include_dirs=include_dirs, macros=macros)
    compiler.link_shared_object(
        [object_file], shared_object_file, libraries=libraries,
        library_dirs=library_dirs,
        extra_preargs=extra_link_preargs)

    return shared_object_file


def get_cuda_dirs(build_ext, cpp_flags):
    cuda_include_dirs = []
    cuda_lib_dirs = []

    cuda_home = os.environ.get('HOROVOD_CUDA_HOME')
    if cuda_home:
        cuda_include_dirs += ['%s/include' % cuda_home]
        cuda_lib_dirs += ['%s/lib' % cuda_home, '%s/lib64' % cuda_home]

    cuda_include = os.environ.get('HOROVOD_CUDA_INCLUDE')
    if cuda_include:
        cuda_include_dirs += [cuda_include]

    cuda_lib = os.environ.get('HOROVOD_CUDA_LIB')
    if cuda_lib:
        cuda_lib_dirs += [cuda_lib]

    if not cuda_include_dirs and not cuda_lib_dirs:
        # default to /usr/local/cuda
        cuda_include_dirs += ['/usr/local/cuda/include']
        cuda_lib_dirs += ['/usr/local/cuda/lib', '/usr/local/cuda/lib64']

    try:
        test_compile(build_ext, 'test_cuda', libraries=['cudart'],
                     include_dirs=cuda_include_dirs,
                     library_dirs=cuda_lib_dirs,
                     extra_compile_preargs=cpp_flags,
                     code=textwrap.dedent('''\
            #include <cuda_runtime.h>
            void test() {
                cudaSetDevice(0);
            }
            '''))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'CUDA library was not found (see error above).\n'
            'Please specify correct CUDA location with the HOROVOD_CUDA_HOME '
            'environment variable or combination of HOROVOD_CUDA_INCLUDE and '
            'HOROVOD_CUDA_LIB environment variables.\n\n'
            'HOROVOD_CUDA_HOME - path where CUDA include and lib directories can be found\n'
            'HOROVOD_CUDA_INCLUDE - path to CUDA include directory\n'
            'HOROVOD_CUDA_LIB - path to CUDA lib directory')

    return cuda_include_dirs, cuda_lib_dirs


def get_rocm_dirs(build_ext, cpp_flags):
    rocm_include_dirs = []
    rocm_lib_dirs = []
    rocm_libs = ['hip_hcc']
    rocm_macros = [('__HIP_PLATFORM_HCC__',1)]

    rocm_path = os.environ.get('HOROVOD_ROCM_HOME', '/opt/rocm')
    rocm_include_dirs += [
            '%s/include' % rocm_path,
            '%s/hcc/include' % rocm_path,
            '%s/hip/include' % rocm_path,
            '%s/hsa/include' % rocm_path,
            ]
    rocm_lib_dirs += [
            '%s/lib' % rocm_path,
            ]

    try:
        test_compile(build_ext, 'test_hip', libraries=rocm_libs, include_dirs=rocm_include_dirs,
                     library_dirs=rocm_lib_dirs, extra_compile_preargs=cpp_flags, macros=rocm_macros,
                     code=textwrap.dedent('''\
            #include <hip/hip_runtime.h>
            void test() {
                hipSetDevice(0);
            }
            '''))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'HIP library and/or ROCm header files not found (see error above).\n'
            'Please specify correct ROCm location with the HOROVOD_ROCM_HOME environment variable')

    return rocm_include_dirs, rocm_lib_dirs, rocm_macros


def get_nccl_vals(build_ext, gpu_include_dirs, gpu_lib_dirs, gpu_macros, cpp_flags, have_rocm):
    nccl_include_dirs = []
    nccl_lib_dirs = []
    nccl_libs = []

    nccl_home = os.environ.get('HOROVOD_NCCL_HOME')
    if nccl_home:
        nccl_include_dirs += ['%s/include' % nccl_home]
        nccl_lib_dirs += ['%s/lib' % nccl_home, '%s/lib64' % nccl_home]

    nccl_include_dir = os.environ.get('HOROVOD_NCCL_INCLUDE')
    if nccl_include_dir:
        nccl_include_dirs += [nccl_include_dir]

    nccl_lib_dir = os.environ.get('HOROVOD_NCCL_LIB')
    if nccl_lib_dir:
        nccl_lib_dirs += [nccl_lib_dir]

    nccl_link_mode = os.environ.get('HOROVOD_NCCL_LINK', 'SHARED' if have_rocm else 'STATIC')
    if nccl_link_mode.upper() == 'SHARED':
        if have_rocm:
            nccl_libs += ['rccl']
        else:
            nccl_libs += ['nccl']
    else:
        nccl_libs += ['nccl_static']
        if have_rocm:
            raise DistutilsPlatformError('RCCL must be a shared library')

    try:
        test_compile(build_ext, 'test_nccl', libraries=nccl_libs,
                     include_dirs=nccl_include_dirs + gpu_include_dirs,
                     library_dirs=nccl_lib_dirs + gpu_lib_dirs,
                     extra_compile_preargs=cpp_flags,
                     macros=gpu_macros,
                     code=textwrap.dedent('''\
            #include <%s>
            #if NCCL_MAJOR < 2
            #error Horovod requires NCCL 2.0 or later version, please upgrade.
            #endif
            void test() {
                ncclUniqueId nccl_id;
                ncclGetUniqueId(&nccl_id);
            }
            '''%('rccl.h' if have_rocm else 'nccl.h')))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'NCCL 2.0 library or its later version was not found (see error above).\n'
            'Please specify correct NCCL location with the HOROVOD_NCCL_HOME '
            'environment variable or combination of HOROVOD_NCCL_INCLUDE and '
            'HOROVOD_NCCL_LIB environment variables.\n\n'
            'HOROVOD_NCCL_HOME - path where NCCL include and lib directories can be found\n'
            'HOROVOD_NCCL_INCLUDE - path to NCCL include directory\n'
            'HOROVOD_NCCL_LIB - path to NCCL lib directory')

    return nccl_include_dirs, nccl_lib_dirs, nccl_libs


def get_ddl_dirs(build_ext, cuda_include_dirs, cuda_lib_dirs, cpp_flags):
    ddl_include_dirs = []
    ddl_lib_dirs = []

    ddl_home = os.environ.get('HOROVOD_DDL_HOME')
    if ddl_home:
        ddl_include_dirs += ['%s/include' % ddl_home]
        ddl_lib_dirs += ['%s/lib' % ddl_home, '%s/lib64' % ddl_home]

    ddl_include_dir = os.environ.get('HOROVOD_DDL_INCLUDE')
    if ddl_include_dir:
        ddl_include_dirs += [ddl_include_dir]

    ddl_lib_dir = os.environ.get('HOROVOD_DDL_LIB')
    if ddl_lib_dir:
        ddl_lib_dirs += [ddl_lib_dir]

    # Keep DDL legacy folders for backward compatibility
    if not ddl_include_dirs:
        ddl_include_dirs += ['/opt/DL/ddl/include']
    if not ddl_lib_dirs:
        ddl_lib_dirs += ['/opt/DL/ddl/lib']

    try:
        test_compile(build_ext, 'test_ddl', libraries=['ddl', 'ddl_pack'],
                     include_dirs=ddl_include_dirs + cuda_include_dirs,
                     library_dirs=ddl_lib_dirs + cuda_lib_dirs,
                     extra_compile_preargs=cpp_flags,
                     code=textwrap.dedent('''\
                     #include <ddl.hpp>
                     void test() {
                     }
                     '''))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'IBM PowerAI DDL library was not found (see error above).\n'
            'Please specify correct DDL location with the HOROVOD_DDL_HOME '
            'environment variable or combination of HOROVOD_DDL_INCLUDE and '
            'HOROVOD_DDL_LIB environment variables.\n\n'
            'HOROVOD_DDL_HOME - path where DDL include and lib directories can be found\n'
            'HOROVOD_DDL_INCLUDE - path to DDL include directory\n'
            'HOROVOD_DDL_LIB - path to DDL lib directory')

    return ddl_include_dirs, ddl_lib_dirs


def set_cuda_options(build_ext, COMPILE_FLAGS, MACROS, INCLUDES, SOURCES, BUILD_MPI, LIBRARY_DIRS, LIBRARIES, **kwargs):
    cuda_include_dirs, cuda_lib_dirs = get_cuda_dirs(build_ext, COMPILE_FLAGS)
    MACROS += [('HAVE_CUDA', '1'), ('HAVE_GPU', '1')]
    INCLUDES += cuda_include_dirs
    SOURCES += ['horovod/common/ops/cuda_operations.cc',
                'horovod/common/ops/gpu_operations.cc']
    if BUILD_MPI:
        SOURCES += ['horovod/common/ops/mpi_gpu_operations.cc']
    LIBRARY_DIRS += cuda_lib_dirs
    LIBRARIES += ['cudart']


def get_common_options(build_ext):
    cpp_flags = get_cpp_flags(build_ext)
    link_flags = get_link_flags(build_ext)

    is_mac = sys.platform == 'darwin'
    compile_without_gloo = os.environ.get('HOROVOD_WITHOUT_GLOO')
    if compile_without_gloo:
        print('INFO: HOROVOD_WITHOUT_GLOO detected, skip compiling Horovod with Gloo.')
        have_gloo = False
        have_cmake = False
    else:
        # determining if system has cmake installed
        compile_with_gloo = os.environ.get('HOROVOD_WITH_GLOO')
        try:
            cmake_bin = get_cmake_bin()
            subprocess.check_output([cmake_bin, '--version'])
            have_cmake = True
        except Exception:
            if compile_with_gloo:
                # Require Gloo to succeed, otherwise fail the install.
                raise RuntimeError('Cannot find CMake. CMake is required to build Horovod with Gloo.')

            print('INFO: Cannot find CMake, will skip compiling Horovod with Gloo.')
            have_cmake = False

        # TODO: remove system check if gloo support MacOX in the future
        #  https://github.com/facebookincubator/gloo/issues/182
        if is_mac:
            if compile_with_gloo:
                raise RuntimeError('Gloo cannot be compiled on MacOS. Unset HOROVOD_WITH_GLOO to use MPI.')
            print('INFO: Gloo cannot be compiled on MacOS, will skip compiling Horovod with Gloo.')

        have_gloo = not is_mac and have_cmake

    compile_without_mpi = os.environ.get('HOROVOD_WITHOUT_MPI')
    mpi_flags = ''
    if compile_without_mpi:
        print('INFO: HOROVOD_WITHOUT_MPI detected, skip compiling Horovod with MPI.')
        have_mpi = False
    else:
        # If without_mpi flag is not set by user, try to get mpi flags
        try:
            mpi_flags = get_mpi_flags()
            have_mpi = True
        except Exception:
            if os.environ.get('HOROVOD_WITH_MPI'):
                # Require MPI to succeed, otherwise fail the install.
                raise

            # If exceptions happen, will not use mpi during compilation.
            print(traceback.format_exc(), file=sys.stderr)
            print('INFO: Cannot find MPI compilation flags, will skip compiling with MPI.')
            have_mpi = False

    if not have_gloo and not have_mpi:
        raise RuntimeError('One of Gloo or MPI are required for Horovod to run. Check the logs above for more info.')

    gpu_allreduce = os.environ.get('HOROVOD_GPU_ALLREDUCE')
    if gpu_allreduce and gpu_allreduce != 'MPI' and gpu_allreduce != 'NCCL' and \
        gpu_allreduce != 'DDL':
        raise DistutilsError('HOROVOD_GPU_ALLREDUCE=%s is invalid, supported '
                             'values are "", "MPI", "NCCL", "DDL".' % gpu_allreduce)

    gpu_allgather = os.environ.get('HOROVOD_GPU_ALLGATHER')
    if gpu_allgather and gpu_allgather != 'MPI':
        raise DistutilsError('HOROVOD_GPU_ALLGATHER=%s is invalid, supported '
                             'values are "", "MPI".' % gpu_allgather)

    gpu_broadcast = os.environ.get('HOROVOD_GPU_BROADCAST')
    if gpu_broadcast and gpu_broadcast != 'MPI' and gpu_broadcast != 'NCCL':
        raise DistutilsError('HOROVOD_GPU_BROADCAST=%s is invalid, supported '
                             'values are "", "MPI", "NCCL".' % gpu_broadcast)

    have_cuda = False
    have_rocm = False
    gpu_include_dirs = gpu_lib_dirs = gpu_macros = []
    if gpu_allreduce or gpu_allgather or gpu_broadcast:
        gpu_type = os.environ.get('HOROVOD_GPU', 'CUDA')
        if gpu_type == 'CUDA':
            have_cuda = True
            gpu_include_dirs, gpu_lib_dirs = get_cuda_dirs(build_ext, cpp_flags)
        elif gpu_type == 'ROCM':
            have_rocm = True
            gpu_include_dirs, gpu_lib_dirs, gpu_macros = get_rocm_dirs(build_ext, cpp_flags)
        else:
            raise DistutilsError("Unknown HOROVOD_GPU type '%s'" % gpu_type)

    if gpu_allreduce == 'NCCL':
        have_nccl = True
        nccl_include_dirs, nccl_lib_dirs, nccl_libs = get_nccl_vals(
            build_ext, gpu_include_dirs, gpu_lib_dirs, gpu_macros, cpp_flags, have_rocm)
    else:
        have_nccl = False
        nccl_include_dirs = nccl_lib_dirs = nccl_libs = []

    if gpu_allreduce == 'DDL':
        warnings.warn('DDL backend has been deprecated. Please, start using the NCCL backend '
                      'by building Horovod with "HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL". '
                      'Will be removed in v0.21.0.',
                      DeprecationWarning)
        have_ddl = True
        ddl_include_dirs, ddl_lib_dirs = get_ddl_dirs(build_ext,
                                                      gpu_include_dirs,
                                                      gpu_lib_dirs, cpp_flags)
    else:
        have_ddl = False
        ddl_include_dirs = ddl_lib_dirs = []

    if gpu_allreduce == 'NCCL' \
            and (gpu_allgather == 'MPI' or gpu_broadcast == 'MPI') \
            and not os.environ.get('HOROVOD_ALLOW_MIXED_GPU_IMPL'):
        raise DistutilsError(
            'You should not mix NCCL and MPI GPU due to a possible deadlock.\n'
            'If you\'re sure you want to mix them, set the '
            'HOROVOD_ALLOW_MIXED_GPU_IMPL environment variable to \'1\'.')

    MACROS = [('EIGEN_MPL2_ONLY', 1)]
    INCLUDES = ['third_party/HTTPRequest/include',
                'third_party/boost/assert/include',
                'third_party/boost/config/include',
                'third_party/boost/core/include',
                'third_party/boost/detail/include',
                'third_party/boost/iterator/include',
                'third_party/boost/lockfree/include',
                'third_party/boost/mpl/include',
                'third_party/boost/parameter/include',
                'third_party/boost/predef/include',
                'third_party/boost/preprocessor/include',
                'third_party/boost/static_assert/include',
                'third_party/boost/type_traits/include',
                'third_party/boost/utility/include',
                'third_party/eigen',
                'third_party/flatbuffers/include',
                'third_party/lbfgs/include']
    SOURCES = ['horovod/common/common.cc',
               'horovod/common/controller.cc',
               'horovod/common/fusion_buffer_manager.cc',
               'horovod/common/logging.cc',
               'horovod/common/message.cc',
               'horovod/common/operations.cc',
               'horovod/common/parameter_manager.cc',
               'horovod/common/response_cache.cc',
               'horovod/common/stall_inspector.cc',
               'horovod/common/thread_pool.cc',
               'horovod/common/timeline.cc',
               'horovod/common/tensor_queue.cc',
               'horovod/common/ops/collective_operations.cc',
               'horovod/common/ops/operation_manager.cc',
               'horovod/common/optim/bayesian_optimization.cc',
               'horovod/common/optim/gaussian_process.cc',
               'horovod/common/utils/env_parser.cc'
               ]
    COMPILE_FLAGS = cpp_flags + shlex.split(mpi_flags)
    LINK_FLAGS = link_flags + shlex.split(mpi_flags)
    LIBRARY_DIRS = []
    LIBRARIES = []

    cpu_operation = os.environ.get('HOROVOD_CPU_OPERATIONS')
    if cpu_operation:
        print('INFO: Set default CPU operation to ' + cpu_operation)
        if cpu_operation.upper() == 'MPI':
            if not have_mpi:
                raise RuntimeError('MPI is not installed, try changing HOROVOD_CPU_OPERATIONS.')
            MACROS += [('HOROVOD_CPU_OPERATIONS_DEFAULT', "'M'")]
        elif cpu_operation.upper() == 'MLSL':
            raise RuntimeError('Intel(R) MLSL was removed. Upgrade to oneCCL and set HOROVOD_CPU_OPERATIONS=CCL.')
        elif cpu_operation.upper() == 'CCL':
            if not have_ccl:
                raise RuntimeError('oneCCL is not installed, try changing HOROVOD_CPU_OPERATIONS.')
            MACROS += [('HOROVOD_CPU_OPERATIONS_DEFAULT', "'C'")]
        elif cpu_operation.upper() == 'GLOO':
            if compile_without_gloo:
                raise ValueError('Cannot set both HOROVOD_WITHOUT_GLOO and HOROVOD_CPU_OPERATIONS=GLOO.')
            if is_mac:
                raise RuntimeError('Cannot compile Gloo on MacOS, try changing HOROVOD_CPU_OPERATIONS.')
            elif not have_cmake:
                raise RuntimeError('Cannot compile Gloo without CMake, try installing CMake first.')
            MACROS += [('HOROVOD_CPU_OPERATIONS_DEFAULT', "'G'")]

    if have_mpi:
        MACROS += [('HAVE_MPI', '1')]
        SOURCES += ['horovod/common/half.cc',
                    'horovod/common/mpi/mpi_context.cc',
                    'horovod/common/mpi/mpi_controller.cc',
                    'horovod/common/ops/mpi_operations.cc',
                    'horovod/common/ops/adasum/adasum_mpi.cc',
                    'horovod/common/ops/adasum_mpi_operations.cc']
        COMPILE_FLAGS += shlex.split(mpi_flags)
        LINK_FLAGS += shlex.split(mpi_flags)

    if have_gloo:
        MACROS += [('HAVE_GLOO', '1')]
        INCLUDES += ['third_party/gloo']
        SOURCES += ['horovod/common/gloo/gloo_context.cc',
                    'horovod/common/gloo/gloo_controller.cc',
                    'horovod/common/gloo/http_store.cc',
                    'horovod/common/gloo/memory_store.cc',
                    'horovod/common/ops/gloo_operations.cc']

    if have_ccl:
        MACROS += [('HAVE_CCL', '1')]
        INCLUDES += [ccl_root + '/include/']
        SOURCES += ['horovod/common/ops/ccl_operations.cc']
        LIBRARY_DIRS += [ccl_root + '/lib/']
        LINK_FLAGS += ['-lccl']

    if have_cuda:
        set_cuda_options(build_ext, COMPILE_FLAGS, MACROS, INCLUDES, SOURCES, have_mpi, LIBRARY_DIRS, LIBRARIES)
        INCLUDES += ['horovod/common/ops/cuda']

    if have_rocm:
        MACROS += [('HAVE_ROCM', '1'), ('HAVE_GPU', '1')] + gpu_macros
        INCLUDES += gpu_include_dirs
        SOURCES += ['horovod/common/ops/hip_operations.cc',
                    'horovod/common/ops/gpu_operations.cc']
        if have_mpi:
            SOURCES += ['horovod/common/ops/mpi_gpu_operations.cc']
        LIBRARY_DIRS += gpu_lib_dirs
        LIBRARIES += ['hip_hcc']

    if have_nccl:
        MACROS += [('HAVE_NCCL', '1')]
        INCLUDES += nccl_include_dirs
        SOURCES += ['horovod/common/ops/nccl_operations.cc']
        if have_mpi:
            SOURCES += ['horovod/common/ops/adasum_gpu_operations.cc']
        LIBRARY_DIRS += nccl_lib_dirs
        LIBRARIES += nccl_libs

    if have_ddl and have_mpi:
        MACROS += [('HAVE_DDL', '1')]
        INCLUDES += ddl_include_dirs
        SOURCES += ['horovod/common/mpi/ddl_mpi_context_manager.cc',
                    'horovod/common/ops/ddl_operations.cc']
        LIBRARY_DIRS += ddl_lib_dirs
        LIBRARIES += ['ddl', 'ddl_pack']

    if gpu_allreduce:
        MACROS += [('HOROVOD_GPU_ALLREDUCE', "'%s'" % gpu_allreduce[0])]

    if gpu_allgather:
        MACROS += [('HOROVOD_GPU_ALLGATHER', "'%s'" % gpu_allgather[0])]

    if gpu_broadcast:
        MACROS += [('HOROVOD_GPU_BROADCAST', "'%s'" % gpu_broadcast[0])]

    return dict(MACROS=MACROS,
                INCLUDES=INCLUDES,
                SOURCES=SOURCES,
                COMPILE_FLAGS=COMPILE_FLAGS,
                LINK_FLAGS=LINK_FLAGS,
                LIBRARY_DIRS=LIBRARY_DIRS,
                LIBRARIES=LIBRARIES,
                BUILD_GLOO=have_gloo,
                BUILD_MPI=have_mpi,
                )


def enumerate_binaries_in_path():
    for path_dir in os.getenv('PATH', '').split(':'):
        if os.path.isdir(path_dir):
            for bin_file in sorted(os.listdir(path_dir)):
                yield path_dir, bin_file


def determine_gcc_version(compiler):
    try:
        compiler_macros = subprocess.check_output(
            '%s -dM -E - </dev/null' % compiler,
            shell=True, universal_newlines=True).split('\n')
        for m in compiler_macros:
            version_match = re.match('^#define __VERSION__ "(.*?)"$', m)
            if version_match:
                return LooseVersion(version_match.group(1))
        print('INFO: Unable to determine version of the compiler %s.' % compiler)

    except subprocess.CalledProcessError:
        print('INFO: Unable to determine version of the compiler %s.\n%s'
              '' % (compiler, traceback.format_exc()))

    return None


def find_gxx_compiler_in_path():
    compilers = []

    for path_dir, bin_file in enumerate_binaries_in_path():
        if re.match('^g\\+\\+(?:-\\d+(?:\\.\\d+)*)?$', bin_file):
            # g++, or g++-7, g++-4.9, or g++-4.8.5
            compiler = os.path.join(path_dir, bin_file)
            compiler_version = determine_gcc_version(compiler)
            if compiler_version:
                compilers.append((compiler, compiler_version))

    return compilers


def find_matching_gcc_compiler_path(gxx_compiler_version):
    for path_dir, bin_file in enumerate_binaries_in_path():
        if re.match('^gcc(?:-\\d+(?:\\.\\d+)*)?$', bin_file):
            # gcc, or gcc-7, gcc-4.9, or gcc-4.8.5
            compiler = os.path.join(path_dir, bin_file)
            compiler_version = determine_gcc_version(compiler)
            if compiler_version and compiler_version == gxx_compiler_version:
                return compiler

    print('INFO: Unable to find gcc compiler (version %s).' % gxx_compiler_version)
    return None


def remove_offensive_gcc_compiler_options(compiler_version):
    offensive_replacements = dict()
    if compiler_version < LooseVersion('4.9'):
        offensive_replacements = {
            '-Wdate-time': '',
            '-fstack-protector-strong': '-fstack-protector'
        }

    if offensive_replacements:
        from sysconfig import get_config_var
        cflags = get_config_var('CONFIGURE_CFLAGS')
        cppflags = get_config_var('CONFIGURE_CPPFLAGS')
        ldshared = get_config_var('LDSHARED')

        for k, v in offensive_replacements.items():
            if cflags:
                cflags = cflags.replace(k, v)
            if cppflags:
                cppflags = cppflags.replace(k, v)
            if ldshared:
                ldshared = ldshared.replace(k, v)

        return cflags, cppflags, ldshared

    # Use defaults
    return None, None, None


# Filter out all the compiler macros (starts with -D)
# that need to be passed to compiler
def filter_compile_macros(compile_flags):
    res = []
    for flag in compile_flags:
        if flag.startswith('-D'):
            res.append(flag)
    return res


def build_tf_extension(build_ext, global_options):
    # Backup the options, preventing other plugins access libs that
    # compiled with compiler of this plugin
    options = deepcopy(global_options)

    check_tf_version()
    tf_compile_flags, tf_link_flags = get_tf_flags(
        build_ext, options['COMPILE_FLAGS'])

    gloo_compile_macros = filter_compile_macros(tf_compile_flags)

    tensorflow_mpi_lib.define_macros = options['MACROS']
    tensorflow_mpi_lib.include_dirs = options['INCLUDES']
    tensorflow_mpi_lib.sources = options['SOURCES'] + \
                                 ['horovod/tensorflow/mpi_ops.cc']
    tensorflow_mpi_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
                                            tf_compile_flags
    tensorflow_mpi_lib.extra_link_args = options['LINK_FLAGS'] + tf_link_flags

    tensorflow_mpi_lib.library_dirs = options['LIBRARY_DIRS']
    tensorflow_mpi_lib.libraries = options['LIBRARIES']

    cc_compiler = cxx_compiler = cflags = cppflags = ldshared = None
    if sys.platform.startswith('linux') and not os.getenv('CC') and not os.getenv('CXX'):
        # Determine g++ version compatible with this TensorFlow installation
        import tensorflow as tf
        if hasattr(tf, 'version'):
            # Since TensorFlow 1.13.0
            tf_compiler_version = LooseVersion(tf.version.COMPILER_VERSION)
        else:
            tf_compiler_version = LooseVersion(tf.COMPILER_VERSION)

        if tf_compiler_version.version[0] == 4:
            # g++ 4.x is ABI-incompatible with g++ 5.x+ due to std::function change
            # See: https://github.com/tensorflow/tensorflow/issues/27067
            maximum_compiler_version = LooseVersion('5')
        else:
            maximum_compiler_version = LooseVersion('999')

        # Find the compatible compiler of the highest version
        compiler_version = LooseVersion('0')
        for candidate_cxx_compiler, candidate_compiler_version in find_gxx_compiler_in_path():
            if candidate_compiler_version >= tf_compiler_version and \
                    candidate_compiler_version < maximum_compiler_version:
                candidate_cc_compiler = \
                    find_matching_gcc_compiler_path(candidate_compiler_version)
                if candidate_cc_compiler and candidate_compiler_version > compiler_version:
                    cc_compiler = candidate_cc_compiler
                    cxx_compiler = candidate_cxx_compiler
                    compiler_version = candidate_compiler_version
            else:
                print('INFO: Compiler %s (version %s) is not usable for this TensorFlow '
                      'installation. Require g++ (version >=%s, <%s).' %
                      (candidate_cxx_compiler, candidate_compiler_version,
                       tf_compiler_version, maximum_compiler_version))

        if cc_compiler:
            print('INFO: Compilers %s and %s (version %s) selected for TensorFlow plugin build.'
                  '' % (cc_compiler, cxx_compiler, compiler_version))
        else:
            raise DistutilsPlatformError(
                'Could not find compiler compatible with this TensorFlow installation.\n'
                'Please check the Horovod website for recommended compiler versions.\n'
                'To force a specific compiler version, set CC and CXX environment variables.')

        cflags, cppflags, ldshared = remove_offensive_gcc_compiler_options(compiler_version)

    try:
        with env(CC=cc_compiler, CXX=cxx_compiler, CFLAGS=cflags, CPPFLAGS=cppflags,
                 LDSHARED=ldshared):
            if options['BUILD_GLOO']:
                build_cmake(build_ext, gloo_lib, 'tf', gloo_compile_macros, options, tensorflow_mpi_lib)
            customize_compiler(build_ext.compiler)
            build_ext.build_extension(tensorflow_mpi_lib)
    finally:
        # Revert to the default compiler settings
        customize_compiler(build_ext.compiler)


def parse_version(version_str):
    if "dev" in version_str:
        return 9999999999
    m = re.match(r'^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?', version_str)
    if m is None:
        return None

    # turn version string to long integer
    version = int(m.group(1)) * 10 ** 9
    if m.group(2) is not None:
        version += int(m.group(2)) * 10 ** 6
    if m.group(3) is not None:
        version += int(m.group(3)) * 10 ** 3
    if m.group(4) is not None:
        version += int(m.group(4))
    return version


def is_mx_mkldnn():
    try:
        from mxnet import runtime
        features = runtime.Features()
        return features.is_enabled('MKLDNN')
    except Exception:
        msg = 'INFO: Cannot detect if MKLDNN is enabled in MXNet. Please \
            set MXNET_USE_MKLDNN=1 if MKLDNN is enabled in your MXNet build.'
        if 'linux' not in sys.platform:
            # MKLDNN is only enabled by default in MXNet Linux build. Return
            # False by default for non-linux build but still allow users to
            # enable it by using MXNET_USE_MKLDNN env variable.
            print(msg)
            return os.environ.get('MXNET_USE_MKLDNN', '0') == '1'
        else:
            try:
                import mxnet as mx
                mx_libs = mx.libinfo.find_lib_path()
                for mx_lib in mx_libs:
                    output = subprocess.check_output(['readelf', '-d', mx_lib])
                    if 'mkldnn' in str(output):
                        return True
                return False
            except Exception:
                print(msg)
                return os.environ.get('MXNET_USE_MKLDNN', '0') == '1'


def is_mx_cuda():
    try:
        from mxnet import runtime
        features = runtime.Features()
        return features.is_enabled('CUDA')
    except Exception:
        if 'linux' in sys.platform:
            try:
                import mxnet as mx
                mx_libs = mx.libinfo.find_lib_path()
                for mx_lib in mx_libs:
                    output = subprocess.check_output(['readelf', '-d', mx_lib])
                    if 'cuda' in str(output):
                        return True
                return False
            except Exception:
                return False
    return False


def build_mx_extension(build_ext, global_options):
    # Backup the options, preventing other plugins access libs that
    # compiled with compiler of this plugin
    options = deepcopy(global_options)

    # First build gloo
    if options['BUILD_GLOO']:
        build_cmake(build_ext, gloo_lib, 'mxnet', [], options=options)

    check_mx_version()
    mx_compile_flags, mx_link_flags = get_mx_flags(
        build_ext, options['COMPILE_FLAGS'])

    mx_have_cuda = is_mx_cuda()
    macro_have_cuda = check_macro(options['MACROS'], 'HAVE_CUDA')
    if not mx_have_cuda and macro_have_cuda:
        raise DistutilsPlatformError(
            'Horovod build with GPU support was requested, but this MXNet '
            'installation does not support CUDA.')

    # Update HAVE_CUDA to mean that MXNet supports CUDA. Internally, we will be checking
    # HOROVOD_GPU_(ALLREDUCE|ALLGATHER|BROADCAST) to decide whether we should use GPU
    # version or transfer tensors to CPU memory for those operations.
    if mx_have_cuda and not macro_have_cuda:
        set_cuda_options(build_ext, **options)

    mxnet_mpi_lib.define_macros = options['MACROS']
    if check_macro(options['MACROS'], 'HAVE_CUDA'):
        mxnet_mpi_lib.define_macros += [('MSHADOW_USE_CUDA', '1')]
    else:
        mxnet_mpi_lib.define_macros += [('MSHADOW_USE_CUDA', '0')]
    if is_mx_mkldnn():
        mxnet_mpi_lib.define_macros += [('MXNET_USE_MKLDNN', '1')]
    else:
        mxnet_mpi_lib.define_macros += [('MXNET_USE_MKLDNN', '0')]
    mxnet_mpi_lib.define_macros += [('MSHADOW_USE_MKL', '0')]
    mxnet_mpi_lib.define_macros += [('MSHADOW_USE_F16C', '0')]
    mxnet_mpi_lib.include_dirs = options['INCLUDES']
    mxnet_mpi_lib.sources = options['SOURCES'] + \
                            ['horovod/mxnet/mpi_ops.cc',
                             'horovod/mxnet/tensor_util.cc',
                             'horovod/mxnet/cuda_util.cc',
                             'horovod/mxnet/adapter.cc']
    mxnet_mpi_lib.extra_compile_args = options['COMPILE_FLAGS'] + \
                                       mx_compile_flags
    mxnet_mpi_lib.extra_link_args = options['LINK_FLAGS'] + mx_link_flags
    mxnet_mpi_lib.library_dirs = options['LIBRARY_DIRS']
    mxnet_mpi_lib.libraries = options['LIBRARIES']

    build_ext.build_extension(mxnet_mpi_lib)


def dummy_import_torch():
    try:
        import torch
    except:
        pass


def check_torch_version():
    try:
        import torch
        if LooseVersion(torch.__version__) < LooseVersion('0.4.0'):
            raise DistutilsPlatformError(
                'Your PyTorch version %s is outdated.  '
                'Horovod requires torch>=0.4.0' % torch.__version__)
    except ImportError:
        raise DistutilsPlatformError(
            'import torch failed, is it installed?\n\n%s' % traceback.format_exc())

    # parse version
    version = parse_version(torch.__version__)
    if version is None:
        raise DistutilsPlatformError(
            'Unable to determine PyTorch version from the version string \'%s\'' % torch.__version__)
    return version


def is_torch_cuda():
    try:
        from torch.utils.ffi import create_extension
        cuda_test_ext = create_extension(
            name='horovod.torch.test_cuda',
            headers=['horovod/torch/dummy.h'],
            sources=[],
            with_cuda=True,
            extra_compile_args=['-std=c11', '-fPIC', '-O3']
        )
        cuda_test_ext.build()
        return True
    except:
        print(
            'INFO: Above error indicates that this PyTorch installation does not support CUDA.')
        return False


def is_torch_cuda_v2(build_ext, include_dirs, extra_compile_args):
    try:
        from torch.utils.cpp_extension import include_paths
        test_compile(build_ext, 'test_torch_cuda',
                     include_dirs=include_dirs + include_paths(cuda=True),
                     extra_compile_preargs=extra_compile_args,
                     code=textwrap.dedent('''\
            #include <THC/THC.h>
            void test() {
            }
            '''))
        return True
    except (CompileError, LinkError, EnvironmentError):
        print(
            'INFO: Above error indicates that this PyTorch installation does not support CUDA.')
        return False


def get_torch_rocm_macros():
    try:
        from torch.utils.cpp_extension import COMMON_HIPCC_FLAGS
        pattern = re.compile(r'-D(\w+)=?(\w+)?')
        return [pattern.match(flag).groups() for flag in COMMON_HIPCC_FLAGS if pattern.match(flag)]
    except:
        return []


def is_torch_rocm_v2(build_ext, include_dirs, extra_compile_args):
    try:
        from torch.utils.cpp_extension import include_paths
        rocm_macros = get_torch_rocm_macros()
        test_compile(build_ext, 'test_torch_rocm',
                     include_dirs=include_dirs + include_paths(cuda=True),
                     extra_compile_preargs=extra_compile_args,
                     macros=rocm_macros,
                     code=textwrap.dedent('''\
            #include <THH/THH.h>
            void test() {
            }
            '''))
        return True
    except (CompileError, LinkError, EnvironmentError):
        print(
            'INFO: Above error indicates that this PyTorch installation does not support ROCm.')
        return False


def check_macro(macros, key):
    return any(k == key and v for k, v in macros)


def set_macro(macros, key, new_value):
    if any(k == key for k, _ in macros):
        return [(k, new_value if k == key else v) for k, v in macros]
    else:
        return macros + [(key, new_value)]


def set_flag(flags, flag, value):
    flag = '-' + flag
    if any(f.split('=')[0] == flag for f in flags):
        return [('{}={}'.format(flag, value) if f.split('=')[0] == flag else f) for f in flags]
    else:
        return flags + ['{}={}'.format(flag, value)]


class protect_files(object):
    def __init__(self, *files):
        self.files = files

    def __enter__(self):
        for file in self.files:
            os.rename(file, file + '.protected')

    def __exit__(self, type, value, traceback):
        for file in self.files:
            os.rename(file + '.protected', file)


def build_torch_extension(build_ext, global_options, torch_version):
    # Backup the options, preventing other plugins access libs that
    # compiled with compiler of this plugin
    options = deepcopy(global_options)

    have_cuda = is_torch_cuda()
    have_cuda_macro = check_macro(options['MACROS'], 'HAVE_CUDA')
    if not have_cuda and have_cuda_macro:
        raise DistutilsPlatformError(
            'Horovod build with GPU support was requested, but this PyTorch '
            'installation does not support CUDA.')

    # Build gloo
    if options['BUILD_GLOO']:
        build_cmake(build_ext, gloo_lib, 'torch', [], options)

    # Update HAVE_CUDA to mean that PyTorch supports CUDA. Internally, we will be checking
    # HOROVOD_GPU_(ALLREDUCE|ALLGATHER|BROADCAST) to decide whether we should use GPU
    # version or transfer tensors to CPU memory for those operations.
    if have_cuda and not have_cuda_macro:
        set_cuda_options(build_ext, **options)

    # Export TORCH_VERSION equal to our representation of torch.__version__. Internally it's
    # used for backwards compatibility checks.
    updated_macros = set_macro(
        options['MACROS'], 'TORCH_VERSION', str(torch_version))

    # Create_extension overwrites these files which are customized, we need to protect them.
    with protect_files('horovod/torch/mpi_lib/__init__.py',
                       'horovod/torch/mpi_lib_impl/__init__.py'):
        from torch.utils.ffi import create_extension
        ffi_iface = create_extension(
            name='horovod.torch.mpi_lib',
            headers=['horovod/torch/interface.h'] +
                    (['horovod/torch/interface_cuda.h'] if have_cuda else []),
            with_cuda=have_cuda,
            language='c',
            package=True,
            sources=[],
            extra_compile_args=['-std=c11', '-fPIC', '-O3']
        )
        ffi_impl = create_extension(
            name='horovod.torch.mpi_lib_impl',
            headers=[],
            with_cuda=have_cuda,
            language='c++',
            package=True,
            source_extension='.cc',
            define_macros=updated_macros,
            include_dirs=options['INCLUDES'],
            sources=options['SOURCES'] + ['horovod/torch/mpi_ops.cc',
                                          'horovod/torch/handle_manager.cc',
                                          'horovod/torch/ready_event.cc',
                                          'horovod/torch/tensor_util.cc',
                                          'horovod/torch/cuda_util.cc',
                                          'horovod/torch/adapter.cc'],
            extra_compile_args=options['COMPILE_FLAGS'],
            extra_link_args=options['LINK_FLAGS'],
            library_dirs=options['LIBRARY_DIRS'],
            libraries=options['LIBRARIES']
        )

    for ffi, setuptools_ext in [(ffi_iface, torch_mpi_lib),
                                (ffi_impl, torch_mpi_lib_impl)]:
        ffi_ext = ffi.distutils_extension()
        # ffi_ext is distutils Extension, not setuptools Extension
        for k, v in ffi_ext.__dict__.items():
            setuptools_ext.__dict__[k] = v
        build_ext.build_extension(setuptools_ext)


def build_torch_extension_v2(build_ext, global_options, torch_version):
    # Backup the options, preventing other plugins access libs that
    # compiled with compiler of this plugin
    options = deepcopy(global_options)

    # Versions of PyTorch > 1.3.0 require C++14
    import torch
    compile_flags = options['COMPILE_FLAGS']
    if LooseVersion(torch.__version__) >= LooseVersion('1.3.0'):
        compile_flags = set_flag(compile_flags, 'std', 'c++14')

    have_cuda = is_torch_cuda_v2(build_ext, include_dirs=options['INCLUDES'],
                                 extra_compile_args=compile_flags)
    have_cuda_macro = check_macro(options['MACROS'], 'HAVE_CUDA')
    if not have_cuda and have_cuda_macro:
        raise DistutilsPlatformError(
            'Horovod build with GPU support was requested, but this PyTorch '
            'installation does not support CUDA.')
    elif have_cuda and not have_cuda_macro:
        # Update HAVE_GPU to mean that PyTorch supports CUDA. Internally, we will be checking
        # HOROVOD_GPU_(ALLREDUCE|ALLGATHER|BROADCAST) to decide whether we should use GPU
        # version or transfer tensors to CPU memory for those operations.
        set_cuda_options(build_ext, **options)

    # hereafter, macros are maintained outside of options dict
    updated_macros = options['MACROS']

    have_rocm = is_torch_rocm_v2(build_ext, include_dirs=options['INCLUDES'],
                                 extra_compile_args=compile_flags)
    have_rocm_macro = check_macro(updated_macros, 'HAVE_ROCM')
    if not have_rocm and have_rocm_macro:
        raise DistutilsPlatformError(
            'Horovod build with GPU support was requested, but this PyTorch '
            'installation does not support ROCm.')
    elif have_rocm and not have_rocm_macro:
        # ROCm PyTorch requires extensions to be hipified with the provided utility.
        # The utility does not change 'HAVE_CUDA', so those were renamed 'HAVE_GPU'.
        # Update HAVE_GPU to mean that PyTorch supports ROCm. Internally, we will be checking
        # HOROVOD_GPU_(ALLREDUCE|ALLGATHER|BROADCAST) to decide whether we should use GPU
        # version or transfer tensors to CPU memory for those operations.
        updated_macros = set_macro(updated_macros, 'HAVE_ROCM', str(int(have_rocm)))
        updated_macros = set_macro(updated_macros, 'HAVE_GPU', str(int(have_rocm)))
        # ROCm PyTorch requires additional macros.
        for (k,v) in get_torch_rocm_macros():
            updated_macros = set_macro(updated_macros, k, v)

    # Export TORCH_VERSION equal to our representation of torch.__version__. Internally it's
    # used for backwards compatibility checks.
    updated_macros = set_macro(updated_macros, 'TORCH_VERSION', str(torch_version))

    # Always set _GLIBCXX_USE_CXX11_ABI, since PyTorch can only detect whether it was set to 1.
    updated_macros = set_macro(updated_macros, '_GLIBCXX_USE_CXX11_ABI',
                               str(int(torch.compiled_with_cxx11_abi())))

    gloo_abi_flag = ['-D_GLIBCXX_USE_CXX11_ABI=' + str(int(torch.compiled_with_cxx11_abi()))]

    # PyTorch requires -DTORCH_API_INCLUDE_EXTENSION_H
    updated_macros = set_macro(updated_macros, 'TORCH_API_INCLUDE_EXTENSION_H', '1')

    if have_rocm:
        from torch.utils.cpp_extension import CUDAExtension as TorchExtension
        from torch.utils.hipify import hipify_python
        this_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "horovod")
        hipify_python.hipify(
                project_directory=this_dir,
                output_directory=this_dir,
                includes=("torch/*.cc","torch/*.h"),
                show_detailed=True,
                is_pytorch_extension=True)
    elif have_cuda:
        from torch.utils.cpp_extension import CUDAExtension as TorchExtension
    else:
        # CUDAExtension fails with `ld: library not found for -lcudart` if CUDA is not present
        from torch.utils.cpp_extension import CppExtension as TorchExtension

    ext = TorchExtension(torch_mpi_lib_v2.name,
                         define_macros=updated_macros,
                         include_dirs=options['INCLUDES'],
                         sources=options['SOURCES'] + [
                            'horovod/torch/mpi_ops_v2.cc',
                            'horovod/torch/handle_manager.cc',
                            'horovod/torch/ready_event.cc',
                            'horovod/torch/cuda_util.cc',
                            'horovod/torch/adapter_v2.cc'],
                         extra_compile_args=compile_flags,
                         extra_link_args=options['LINK_FLAGS'],
                         library_dirs=options['LIBRARY_DIRS'],
                         libraries=options['LIBRARIES'])

    # Patch an existing torch_mpi_lib_v2 extension object.
    for k, v in ext.__dict__.items():
        torch_mpi_lib_v2.__dict__[k] = v

    cc_compiler = cxx_compiler = cflags = cppflags = ldshared = None
    if sys.platform.startswith('linux') and not os.getenv('CC') and not os.getenv('CXX'):
        from torch.utils.cpp_extension import check_compiler_abi_compatibility

        # Find the compatible compiler of the highest version
        compiler_version = LooseVersion('0')
        for candidate_cxx_compiler, candidate_compiler_version in find_gxx_compiler_in_path():
            if check_compiler_abi_compatibility(candidate_cxx_compiler):
                candidate_cc_compiler = \
                    find_matching_gcc_compiler_path(candidate_compiler_version)
                if candidate_cc_compiler and candidate_compiler_version > compiler_version:
                    cc_compiler = candidate_cc_compiler
                    cxx_compiler = candidate_cxx_compiler
                    compiler_version = candidate_compiler_version
            else:
                print('INFO: Compiler %s (version %s) is not usable for this PyTorch '
                      'installation, see the warning above.' %
                      (candidate_cxx_compiler, candidate_compiler_version))

        if cc_compiler:
            print('INFO: Compilers %s and %s (version %s) selected for PyTorch plugin build.'
                  '' % (cc_compiler, cxx_compiler, compiler_version))
        else:
            raise DistutilsPlatformError(
                'Could not find compiler compatible with this PyTorch installation.\n'
                'Please check the Horovod website for recommended compiler versions.\n'
                'To force a specific compiler version, set CC and CXX environment variables.')

        cflags, cppflags, ldshared = remove_offensive_gcc_compiler_options(compiler_version)

    try:
        with env(CC=cc_compiler, CXX=cxx_compiler, CFLAGS=cflags, CPPFLAGS=cppflags,
                 LDSHARED=ldshared):
            if options['BUILD_GLOO']:
                build_cmake(build_ext, gloo_lib, 'torchv2', gloo_abi_flag, options, torch_mpi_lib_v2)
            customize_compiler(build_ext.compiler)
            build_ext.build_extension(torch_mpi_lib_v2)
    finally:
        # Revert to the default compiler settings
        customize_compiler(build_ext.compiler)


def get_cmake_bin():
    return os.environ.get('HOROVOD_CMAKE', 'cmake')


def build_cmake(build_ext, ext, prefix, additional_flags, options, plugin_ext=None):
    cmake_bin = get_cmake_bin()

    # All statically linked libraries will be placed here
    lib_output_dir = os.path.abspath(os.path.join(build_ext.build_temp, 'lib', prefix))
    if not os.path.exists(lib_output_dir):
        os.makedirs(lib_output_dir)

    if plugin_ext:
        plugin_ext.library_dirs.append(lib_output_dir)
    options['LIBRARY_DIRS'].append(lib_output_dir)

    extdir = os.path.abspath(
        os.path.dirname(build_ext.get_ext_fullpath(ext.name)))
    config = 'Debug' if build_ext.debug else 'Release'

    # Pass additional compiler flags by setting CMAKE_CXX_FLAGS_[DEBUG/RELEASE]
    # so that cmake will append these flags to CMAKE_CXX_FLAGS
    additional_cxx_flags = pipes.quote(' '.join(additional_flags))
    cmake_cxx_flag = '-DCMAKE_CXX_FLAGS_{type}:STRING={flags}'.format(
        type=config.upper(), flags=additional_cxx_flags)

    use_mpi_flag = 'ON' if options['BUILD_MPI'] else 'OFF'
    cmake_args = ['-DUSE_MPI=' + use_mpi_flag,
                  '-DCMAKE_BUILD_TYPE=' + config,
                  cmake_cxx_flag,
                  '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(config.upper(), extdir),
                  '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_{}={}'.format(config.upper(),
                                                                  lib_output_dir),
                  ]

    cmake_build_args = [
        '--config', config,
        '--', '-j4',
    ]

    # Keep temp build files within a unique subdirectory
    build_temp = os.path.abspath(os.path.join(build_ext.build_temp, ext.name, prefix))
    if not os.path.exists(build_temp):
        os.makedirs(build_temp)

    # Config and build the extension
    try:
        subprocess.check_call([cmake_bin, ext.cmake_lists_dir] + cmake_args,
                              cwd=build_temp)
        subprocess.check_call([cmake_bin, '--build', '.'] + cmake_build_args,
                              cwd=build_temp)
    except OSError as e:
        raise RuntimeError('CMake failed: {}'.format(str(e)))

    # Add the library so the plugin will link against it during compilation
    options['LIBRARIES'].append(ext.name)
    if plugin_ext:
        plugin_ext.libraries.append(ext.name)


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        options = get_common_options(self)
        built_plugins = []

        # If PyTorch is installed, it must be imported before TensorFlow, otherwise
        # we may get an error: dlopen: cannot load any more object with static TLS
        if not os.environ.get('HOROVOD_WITHOUT_PYTORCH'):
            dummy_import_torch()
        if not os.environ.get('HOROVOD_WITHOUT_TENSORFLOW'):
            try:
                build_tf_extension(self, options)
                built_plugins.append(True)
            except:
                if not os.environ.get('HOROVOD_WITH_TENSORFLOW'):
                    print(
                        'INFO: Unable to build TensorFlow plugin, will skip it.\n\n'
                        '%s' % traceback.format_exc(), file=sys.stderr)
                    built_plugins.append(False)
                else:
                    raise
        if not os.environ.get('HOROVOD_WITHOUT_PYTORCH'):
            try:
                torch_version = check_torch_version()
                if torch_version >= 1000000000:
                    build_torch_extension_v2(self, options, torch_version)
                else:
                    build_torch_extension(self, options, torch_version)
                built_plugins.append(True)
            except:
                if not os.environ.get('HOROVOD_WITH_PYTORCH'):
                    print(
                        'INFO: Unable to build PyTorch plugin, will skip it.\n\n'
                        '%s' % traceback.format_exc(), file=sys.stderr)
                    built_plugins.append(False)
                else:
                    raise
        if not os.environ.get('HOROVOD_WITHOUT_MXNET'):
            try:
                build_mx_extension(self, options)
                built_plugins.append(True)
            except:
                if not os.environ.get('HOROVOD_WITH_MXNET'):
                    print(
                        'INFO: Unable to build MXNet plugin, will skip it.\n\n'
                        '%s' % traceback.format_exc(), file=sys.stderr)
                    built_plugins.append(False)
                else:
                    raise
        if not built_plugins:
            raise DistutilsError(
                'TensorFlow, PyTorch, and MXNet plugins were excluded from build. Aborting.')
        if not any(built_plugins):
            raise DistutilsError(
                'None of TensorFlow, PyTorch, or MXNet plugins were built. See errors above.')


require_list = ['cloudpickle', 'psutil', 'pyyaml', 'six']
test_require_list = ['mock', 'pytest', 'pytest-forked']

# framework dependencies
tensorflow_require_list = ['tensorflow']
tensorflow_cpu_require_list = ['tensorflow-cpu']
tensorflow_gpu_require_list = ['tensorflow-gpu']
keras_require_list = ['keras>=2.0.8,!=2.0.9,!=2.1.0,!=2.1.1']
pytorch_require_list = ['torch']
mxnet_require_list = ['mxnet>=1.4.1']
spark_require_list = ['h5py>=2.9', 'numpy', 'petastorm>=0.9.0', 'pyarrow>=0.15.0', 'pyspark>=2.3.2']
# all frameworks' dependencies
all_frameworks_require_list = tensorflow_require_list + \
                              tensorflow_gpu_require_list + \
                              keras_require_list + \
                              pytorch_require_list + \
                              mxnet_require_list + \
                              spark_require_list

# Skip cffi if pytorch extension explicitly disabled
if not os.environ.get('HOROVOD_WITHOUT_PYTORCH'):
    require_list.append('cffi>=1.4.0')


def get_package_version():
    return __version__ + "+" + os.environ['HOROVOD_LOCAL_VERSION'] if 'HOROVOD_LOCAL_VERSION' in os.environ else __version__


setup(name='horovod',
      version=get_package_version(),
      packages=find_packages(),
      description='Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.',
      author='The Horovod Authors',
      long_description=textwrap.dedent('''\
          Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
          The goal of Horovod is to make distributed Deep Learning fast and easy to use.'''),
      url='https://github.com/horovod/horovod',
      classifiers=[
          'License :: OSI Approved :: Apache Software License'
      ],
      ext_modules=[tensorflow_mpi_lib, torch_mpi_lib, torch_mpi_lib_impl,
                   torch_mpi_lib_v2, mxnet_mpi_lib, gloo_lib],
      cmdclass={'build_ext': custom_build_ext},
      # cffi is required for PyTorch
      # If cffi is specified in setup_requires, it will need libffi to be installed on the machine,
      # which is undesirable.  Luckily, `install` action will install cffi before executing build,
      # so it's only necessary for `build*` or `bdist*` actions.
      setup_requires=require_list if is_build_action() else [],
      install_requires=require_list,
      tests_require=test_require_list,
      extras_require={
          'all-frameworks': all_frameworks_require_list,
          'tensorflow': tensorflow_require_list,
          'tensorflow-cpu': tensorflow_cpu_require_list,
          'tensorflow-gpu': tensorflow_gpu_require_list,
          'keras': keras_require_list,
          'pytorch': pytorch_require_list,
          'mxnet': mxnet_require_list,
          'spark': spark_require_list
      },
      zip_safe=False,
      scripts=['bin/horovodrun'])
