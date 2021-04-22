# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright Microsoft
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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
import subprocess
import sys
import textwrap

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from horovod import __version__


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', sources=[], **kwa):
        Extension.__init__(self, name, sources=sources, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


tensorflow_mpi_lib = CMakeExtension('horovod.tensorflow.mpi_lib',
                                     cmake_lists_dir='.', sources=[])
torch_mpi_lib_v2 = CMakeExtension('horovod.torch.mpi_lib_v2',
                                     cmake_lists_dir='.', sources=[])
mxnet_mpi_lib = CMakeExtension('horovod.mxnet.mpi_lib',
                                     cmake_lists_dir='.', sources=[])

def is_build_action():
    if len(sys.argv) <= 1:
        return False

    if sys.argv[1].startswith('build'):
        return True

    if sys.argv[1].startswith('bdist'):
        return True

    if sys.argv[1].startswith('install'):
        return True


def get_cmake_bin():
    return os.environ.get('HOROVOD_CMAKE', 'cmake')


class custom_build_ext(build_ext):
    def build_extensions(self):
        if os.getenv('HOROVOD_SKIP_COMPILE') == '1':
            # Skip building extensions using CMake
            print("Horovod is being installed without native libraries")
            return

        cmake_bin = get_cmake_bin()

        config = 'Debug' if self.debug else 'RelWithDebInfo'

        ext_name = self.extensions[0].name
        build_dir = self.get_ext_fullpath(ext_name).replace(self.get_ext_filename(ext_name), '')
        build_dir = os.path.abspath(build_dir)

        cmake_args = ['-DCMAKE_BUILD_TYPE=' + config,
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(config.upper(), build_dir),
                      '-DPYTHON_EXECUTABLE:FILEPATH=' + sys.executable]

        make_args = []
        if self.verbose:
            make_args.append('VERBOSE=1')

        cmake_build_args = ['--config', config]
        if make_args:
            # -- specifies that these args are going to the native build tool: make
            cmake_build_args += ['--'] + make_args

        cmake_build_dir = os.path.join(self.build_temp, config)
        if not os.path.exists(cmake_build_dir):
            os.makedirs(cmake_build_dir)

        # Config and build the extension
        try:
            subprocess.check_call([cmake_bin, self.extensions[0].cmake_lists_dir] + cmake_args,
                                  cwd=cmake_build_dir)
            subprocess.check_call([cmake_bin, '--build', '.'] + cmake_build_args,
                                  cwd=cmake_build_dir)
        except OSError as e:
            raise RuntimeError('CMake failed: {}'.format(str(e)))


# python packages required to use horovod in general
require_list = ['cloudpickle', 'psutil', 'pyyaml', 'dataclasses;python_version<"3.7"']

# framework dependencies
tensorflow_require_list = ['tensorflow']
tensorflow_cpu_require_list = ['tensorflow-cpu']
tensorflow_gpu_require_list = ['tensorflow-gpu']
keras_require_list = ['keras>=2.0.8,!=2.0.9,!=2.1.0,!=2.1.1']
pytorch_require_list = ['torch', 'pytorch_lightning']
mxnet_require_list = ['mxnet>=1.4.1']
pyspark_require_list = ['pyspark>=2.3.2;python_version<"3.8"',
                        'pyspark>=3.0.0;python_version>="3.8"']
# Pin h5py: https://github.com/h5py/h5py/issues/1732
spark_require_list = ['h5py<3', 'numpy', 'petastorm>=0.9.8', 'pyarrow>=0.15.0']
ray_require_list = ['ray']
pytorch_spark_require_list = pytorch_require_list + \
                             spark_require_list + \
                             pyspark_require_list

# all frameworks' dependencies
all_frameworks_require_list = tensorflow_require_list + \
                              tensorflow_gpu_require_list + \
                              keras_require_list + \
                              pytorch_require_list + \
                              mxnet_require_list + \
                              spark_require_list + \
                              pyspark_require_list

# python packages required / recommended to develop horovod
# these are the earliest versions to work with Python 3.8
# keep in sync with Dockerfile.test.cpu
# NOTE: do not use versions with +cpu or +gpu here as users would need to add --find-links to pip
dev_require_list = ['tensorflow-cpu==2.2.0',
                    'keras==2.3.1',
                    'torch==1.4.0',
                    'torchvision==0.5.0',
                    'pytorch_lightning>=1.2.9',
                    'mxnet==1.5.0',
                    'pyspark==3.0.1'] + spark_require_list
# torchvision 0.5.0 depends on torch==1.4.0

# python packages required only to run tests
# Pin h5py: https://github.com/h5py/h5py/issues/1732
test_require_list = ['mock', 'pytest', 'pytest-forked', 'parameterized', 'h5py<3']

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
      license='Apache 2.0',
      long_description=textwrap.dedent('''\
          Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
          The goal of Horovod is to make distributed Deep Learning fast and easy to use.'''),
      url='https://github.com/horovod/horovod',
      keywords=['deep learning', 'tensorflow', 'keras', 'pytorch', 'mxnet', 'spark', 'AI'],
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      ext_modules=[tensorflow_mpi_lib, torch_mpi_lib_v2, mxnet_mpi_lib],
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
          'spark': spark_require_list + pyspark_require_list,
          'pytorch-spark': pytorch_spark_require_list,
          'ray': ray_require_list,
          'dev': dev_require_list,
          'test': test_require_list,
      },
      python_requires='>=3.6',
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'horovodrun = horovod.runner.launch:run_commandline'
          ]
      })
