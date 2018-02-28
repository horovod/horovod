# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
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
import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsError, DistutilsPlatformError, LinkError
import shlex
import subprocess
import sys
import textwrap
import traceback

from horovod import __version__


common_mpi_lib = Extension('horovod.common.mpi_lib', [])
tensorflow_mpi_lib = Extension('horovod.tensorflow.mpi_lib', [])


def check_tf_version():
    try:
        import tensorflow as tf
        if tf.__version__ < '1.1.0':
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


def get_cpp_flags(build_ext):
    last_err = None
    default_flags = ['-std=c++11', '-fPIC', '-O2']
    if sys.platform == 'darwin':
        # Darwin most likely will have Clang, which has libc++.
        flags_to_try = [default_flags + ['-stdlib=libc++'], default_flags]
    else:
        flags_to_try = [default_flags, default_flags + ['-stdlib=libc++']]
    for cpp_flags in flags_to_try:
        try:
            test_compile(build_ext, 'test_cpp_flags', extra_preargs=cpp_flags,
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
                                    extra_preargs=cpp_flags,
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
                                    include_dirs=include_dirs, library_dirs=lib_dirs,
                                    libraries=libs, extra_preargs=cpp_flags,
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
        tf_abi = get_tf_abi(build_ext, tf_include_dirs, tf_lib_dirs, tf_libs, cpp_flags)

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


def test_compile(build_ext, name, code, libraries=None, include_dirs=None, library_dirs=None, macros=None,
                 extra_preargs=None):
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

    compiler.compile([source_file], extra_preargs=extra_preargs,
                     include_dirs=include_dirs, macros=macros)
    compiler.link_shared_object(
        [object_file], shared_object_file, libraries=libraries, library_dirs=library_dirs)

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
        test_compile(build_ext, 'test_cuda', libraries=['cudart'], include_dirs=cuda_include_dirs,
                     library_dirs=cuda_lib_dirs, extra_preargs=cpp_flags, code=textwrap.dedent('''\
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


def get_nccl_dirs(build_ext, cuda_include_dirs, cuda_lib_dirs, cpp_flags):
    nccl_include_dirs = []
    nccl_lib_dirs = []

    nccl_home = os.environ.get('HOROVOD_NCCL_HOME')
    if nccl_home:
        nccl_include_dirs += ['%s/include' % nccl_home]
        nccl_lib_dirs += ['%s/lib' % nccl_home, '%s/lib64' % nccl_home]

    nccl_include = os.environ.get('HOROVOD_NCCL_INCLUDE')
    if nccl_include:
        nccl_include_dirs += [nccl_include]

    nccl_lib = os.environ.get('HOROVOD_NCCL_LIB')
    if nccl_lib:
        nccl_lib_dirs += [nccl_lib]

    try:
        test_compile(build_ext, 'test_nccl', libraries=['nccl_static'], include_dirs=nccl_include_dirs + cuda_include_dirs,
                     library_dirs=nccl_lib_dirs + cuda_lib_dirs, extra_preargs=cpp_flags, code=textwrap.dedent('''\
            #include <nccl.h>
            #if NCCL_MAJOR < 2
            #error Horovod requires NCCL 2.0 or later version, please upgrade.
            #endif
            void test() {
                ncclUniqueId nccl_id;
                ncclGetUniqueId(&nccl_id);
            }
            '''))
    except (CompileError, LinkError):
        raise DistutilsPlatformError(
            'NCCL 2.0 library or its later version was not found (see error above).\n'
            'Please specify correct NCCL location with the HOROVOD_NCCL_HOME '
            'environment variable or combination of HOROVOD_NCCL_INCLUDE and '
            'HOROVOD_NCCL_LIB environment variables.\n\n'
            'HOROVOD_NCCL_HOME - path where NCCL include and lib directories can be found\n'
            'HOROVOD_NCCL_INCLUDE - path to NCCL include directory\n'
            'HOROVOD_NCCL_LIB - path to NCCL lib directory')

    return nccl_include_dirs, nccl_lib_dirs


def get_common_options(build_ext):
    cpp_flags = get_cpp_flags(build_ext)
    mpi_flags = get_mpi_flags()

    gpu_allreduce = os.environ.get('HOROVOD_GPU_ALLREDUCE')
    if gpu_allreduce and gpu_allreduce != 'MPI' and gpu_allreduce != 'NCCL':
        raise DistutilsError('HOROVOD_GPU_ALLREDUCE=%s is invalid, supported '
                             'values are "", "MPI", "NCCL".' % gpu_allreduce)

    gpu_allgather = os.environ.get('HOROVOD_GPU_ALLGATHER')
    if gpu_allgather and gpu_allgather != 'MPI':
        raise DistutilsError('HOROVOD_GPU_ALLGATHER=%s is invalid, supported '
                             'values are "", "MPI".' % gpu_allgather)

    gpu_broadcast = os.environ.get('HOROVOD_GPU_BROADCAST')
    if gpu_broadcast and gpu_broadcast != 'MPI':
        raise DistutilsError('HOROVOD_GPU_BROADCAST=%s is invalid, supported '
                             'values are "", "MPI".' % gpu_broadcast)

    if gpu_allreduce or gpu_allgather or gpu_broadcast:
        have_cuda = True
        cuda_include_dirs, cuda_lib_dirs = get_cuda_dirs(build_ext, cpp_flags)
    else:
        have_cuda = False
        cuda_include_dirs = cuda_lib_dirs = []

    if gpu_allreduce == 'NCCL':
        have_nccl = True
        nccl_include_dirs, nccl_lib_dirs = get_nccl_dirs(
            build_ext, cuda_include_dirs, cuda_lib_dirs, cpp_flags)
    else:
        have_nccl = False
        nccl_include_dirs = nccl_lib_dirs = []

    MACROS = []
    INCLUDES = []
    SOURCES = []
    COMPILE_FLAGS = cpp_flags + shlex.split(mpi_flags)
    LINK_FLAGS = shlex.split(mpi_flags)
    LIBRARY_DIRS = []
    LIBRARIES = []

    if have_cuda:
        MACROS += [('HAVE_CUDA', '1')]
        INCLUDES += cuda_include_dirs
        LIBRARY_DIRS += cuda_lib_dirs
        LIBRARIES += ['cudart']

    if have_nccl:
        MACROS += [('HAVE_NCCL', '1')]
        INCLUDES += nccl_include_dirs
        LINK_FLAGS += ['-Wl,--version-script=hide_nccl.lds']
        LIBRARY_DIRS += nccl_lib_dirs
        LIBRARIES += ['nccl_static']

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
                LIBRARIES=LIBRARIES)


def build_common_extension(build_ext, options, abi_compile_flags):
    common_mpi_lib.define_macros = options['MACROS']
    common_mpi_lib.include_dirs = options['INCLUDES']
    common_mpi_lib.sources = options['SOURCES'] + ['horovod/common/common.cc',
                                                   'horovod/common/mpi_message.cc',
                                                   'horovod/common/operations.cc',
                                                   'horovod/common/timeline.cc']
    common_mpi_lib.extra_compile_args = options['COMPILE_FLAGS'] + abi_compile_flags
    common_mpi_lib.extra_link_args = options['LINK_FLAGS']
    common_mpi_lib.library_dirs = options['LIBRARY_DIRS']
    common_mpi_lib.libraries = options['LIBRARIES']

    build_ext.build_extension(common_mpi_lib)


def build_tf_extension(build_ext, options):
    check_tf_version()
    tf_compile_flags, tf_link_flags = get_tf_flags(build_ext, options['COMPILE_FLAGS'])

    tensorflow_mpi_lib.define_macros = options['MACROS']
    tensorflow_mpi_lib.include_dirs = options['INCLUDES']
    tensorflow_mpi_lib.sources = options['SOURCES'] + ['horovod/tensorflow/mpi_ops.cc']
    tensorflow_mpi_lib.extra_compile_args = options['COMPILE_FLAGS'] + tf_compile_flags
    tensorflow_mpi_lib.extra_link_args = options['LINK_FLAGS'] + tf_link_flags
    tensorflow_mpi_lib.library_dirs = options['LIBRARY_DIRS']
    tensorflow_mpi_lib.libraries = options['LIBRARIES']

    build_ext.build_extension(tensorflow_mpi_lib)

    # Return ABI flags used for TensorFlow compilation.  We will use this flag
    # to compile all the libraries.
    return [flag for flag in tf_compile_flags if '_GLIBCXX_USE_CXX11_ABI' in flag]


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        options = get_common_options(self)
        abi_compile_flags = build_tf_extension(self, options)
        build_common_extension(self, options, abi_compile_flags)


setup(name='horovod',
      version=__version__,
      packages=find_packages(),
      description='Distributed training framework for TensorFlow.',
      author='Uber Technologies, Inc.',
      long_description=textwrap.dedent('''\
          Horovod is a distributed training framework for TensorFlow. 
          The goal of Horovod is to make distributed Deep Learning
          fast and easy to use.'''),
      url='https://github.com/uber/horovod',
      classifiers=[
          'License :: OSI Approved :: Apache Software License'
      ],
      ext_modules=[common_mpi_lib, tensorflow_mpi_lib],
      cmdclass={'build_ext': custom_build_ext},
      zip_safe=False)
