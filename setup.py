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

import os
import textwrap

from setuptools import setup, find_packages

from horovod import __version__

# Launcher-only build: the TensorFlow/PyTorch/MXNet framework integrations and the
# shared C++ allreduce core have been removed. This package ships only the
# `horovodrun` CLI / `horovod.runner` process launcher. Distributed communication in
# the launched processes is expected to be handled by the user's framework of choice
# (e.g. PyTorch `torch.distributed`). There is nothing to compile, so `pip install`
# is a pure-Python install.

# python packages required to use the launcher
require_list = ['cloudpickle', 'psutil', 'pyyaml', 'dataclasses;python_version<"3.7"', 'packaging']

# python packages required only to run tests
test_require_list = ['mock', 'pytest<8', 'pytest-forked', 'pytest-subtests', 'parameterized']


def get_package_version():
    return __version__ + "+" + os.environ['HOROVOD_LOCAL_VERSION'] if 'HOROVOD_LOCAL_VERSION' in os.environ else __version__


setup(name='horovod',
      version=get_package_version(),
      packages=find_packages(),
      description='A pure-Python multi-process launcher (horovodrun) derived from Horovod.',
      author='The Horovod Authors',
      license='Apache 2.0',
      long_description=textwrap.dedent('''\
          A launcher-only build of Horovod. It provides the `horovodrun` CLI and the
          `horovod.runner` package for launching single-node and multi-node multi-GPU
          jobs; the TensorFlow/PyTorch/MXNet integrations and the C++ allreduce core
          have been removed. Distributed communication in the launched processes is
          expected to be handled by the user (e.g. PyTorch `torch.distributed`).'''),
      url='https://github.com/horovod/horovod',
      keywords=['deep learning', 'distributed training', 'launcher', 'pytorch', 'AI'],
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      install_requires=require_list,
      tests_require=test_require_list,
      extras_require={
          'test': test_require_list,
      },
      python_requires='>=3.6',
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'horovodrun = horovod.runner.launch:run_commandline'
          ]
      })
