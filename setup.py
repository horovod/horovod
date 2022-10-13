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

import atexit
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

from horovod import __version__

_FRAMEWORK_METADATA_FILE = 'horovod/metadata.json'

class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir='.', sources=None, **kwa):
        if sources is None:
            sources = []
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

    if sys.argv[1].startswith('develop'):
        return True

def get_cmake_bin():
    from packaging import version

    if 'HOROVOD_CMAKE' in os.environ:
        return os.environ['HOROVOD_CMAKE']

    cmake_bin = 'cmake'
    try:
        out = subprocess.check_output([cmake_bin, '--version'])
    except OSError:
        cmake_installed_version = version.parse("0.0")
    else:
        cmake_installed_version = version.parse(re.search(r'version\s*([\d.]+)', out.decode()).group(1))

    if cmake_installed_version < version.parse("3.13.0"):
        print("Could not find a recent CMake to build Horovod. "
              "Attempting to install CMake 3.13 to a temporary location via pip.", flush=True)
        cmake_temp_dir = tempfile.TemporaryDirectory(prefix="horovod-cmake-tmp")
        atexit.register(cmake_temp_dir.cleanup)
        try:
            _ = subprocess.check_output(["pip", "install", "--target", cmake_temp_dir.name, "cmake~=3.13.0"])
        except Exception:
            raise RuntimeError("Failed to install temporary CMake. "
                               "Please update your CMake to 3.13+ or set HOROVOD_CMAKE appropriately.")
        cmake_bin = os.path.join(cmake_temp_dir.name, "bin", "run_cmake")
        with io.open(cmake_bin, "w") as f_run_cmake:
            f_run_cmake.write(
                f"#!/bin/sh\nPYTHONPATH={cmake_temp_dir.name} {os.path.join(cmake_temp_dir.name, 'bin', 'cmake')} \"$@\"")
        os.chmod(cmake_bin, 0o755)

    return cmake_bin


class custom_build_ext(build_ext):
    def build_extensions(self):
        if os.getenv('HOROVOD_SKIP_COMPILE') == '1':
            # Skip building extensions using CMake
            print("Horovod is being installed without native libraries")
            return

        cmake_bin = get_cmake_bin()

        config = 'Debug' if self.debug or os.environ.get('HOROVOD_DEBUG') == "1" else 'RelWithDebInfo'

        ext_name = self.extensions[0].name
        build_dir = self.get_ext_fullpath(ext_name).replace(self.get_ext_filename(ext_name), '')
        build_dir = os.path.abspath(build_dir)

        cmake_args = ['-DCMAKE_BUILD_TYPE=' + config,
                      '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(config.upper(), build_dir),
                      '-DPYTHON_EXECUTABLE:FILEPATH=' + sys.executable]

        make_args = ['-j8'] if not os.environ.get('MAKEFLAGS') else []
        if self.verbose:
            make_args.append('VERBOSE=1')

        cmake_build_args = ['--config', config]
        if make_args:
            # -- specifies that these args are going to the native build tool: make
            cmake_build_args += ['--'] + make_args

        cmake_build_dir = os.path.join(self.build_temp, config)
        if not os.path.exists(cmake_build_dir):
            os.makedirs(cmake_build_dir)

        config_and_build_commands = [
            [cmake_bin, self.extensions[0].cmake_lists_dir] + cmake_args,
            [cmake_bin, '--build', '.'] + cmake_build_args
        ]

        if self.verbose:
            print(f"Running CMake in {cmake_build_dir}:")
            for command in config_and_build_commands:
                print(" ".join(command))
            sys.stdout.flush()

        # Config and build the extension
        try:
            for command in config_and_build_commands:
                subprocess.check_call(command, cwd=cmake_build_dir)
        except OSError as e:
            raise RuntimeError('CMake failed: {}'.format(str(e)))

        if sys.argv[1].startswith('develop'):
            # Copy over metadata.json file from build directory
            shutil.copyfile(os.path.join(build_dir, _FRAMEWORK_METADATA_FILE),
                            os.path.join(self.extensions[0].cmake_lists_dir, _FRAMEWORK_METADATA_FILE))
            # Remove unfound frameworks, otherwise develop mode will fail the install
            self.extensions = [x for x in self.extensions if os.path.exists(self.get_ext_fullpath(x.name))]


# python packages required to use horovod in general
require_list = ['cloudpickle', 'psutil', 'pyyaml', 'dataclasses;python_version<"3.7"', 'packaging']

# framework dependencies
tensorflow_require_list = ['tensorflow']
tensorflow_cpu_require_list = ['tensorflow-cpu']
tensorflow_gpu_require_list = ['tensorflow-gpu']
keras_require_list = ['keras>=2.0.8,!=2.0.9,!=2.1.0,!=2.1.1']
# pytorch-lightning 1.3.8 is a stable version to work with horovod
pytorch_require_list = ['torch']
mxnet_require_list = ['mxnet>=1.4.1']
pyspark_require_list = ['pyspark>=2.3.2;python_version<"3.8"',
                        'pyspark>=3.0.0;python_version>="3.8"']
spark_require_list = ['numpy', 'petastorm>=0.12.0', 'pyarrow>=0.15.0', 'fsspec>=2021.07.0']
# https://github.com/ray-project/ray/pull/17465
# google-api-core>=2.9.0 depends on protobuf<5.0.0dev,>=3.20.1, which conflicts with
#   tensorflow protobuf~=3.20 and pytorch-lightning protobuf<3.20,>=3.9.2
ray_require_list = ['ray', 'aioredis<2', 'google-api-core<2.9.0']
pytorch_spark_require_list = pytorch_require_list + \
                             spark_require_list + \
                             pyspark_require_list + \
                             ['pytorch_lightning>=1.3.8,<1.5.10']

# all frameworks' dependencies
all_frameworks_require_list = tensorflow_require_list + \
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
                    'pytorch_lightning>=1.3.8,<1.5.10',
                    'mxnet==1.5.0',
                    'pyspark==3.0.1'] + spark_require_list
# torchvision 0.5.0 depends on torch==1.4.0

# python packages required only to run tests
test_require_list = ['mock', 'pytest', 'pytest-forked', 'pytest-subtests', 'parameterized']

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
