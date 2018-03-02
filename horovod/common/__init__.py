# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
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
# =============================================================================

import ctypes
import os
import sysconfig


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


MPI_COMMON_LIB_CTYPES = \
    ctypes.CDLL(os.path.join(os.path.dirname(__file__),
                             'mpi_lib' + get_ext_suffix()), mode=ctypes.RTLD_GLOBAL)


def init():
    """A function that initializes Horovod.
    """
    return MPI_COMMON_LIB_CTYPES.horovod_init()


def size():
    """A function that returns the number of Horovod processes.

    Returns:
      An integer scalar containing the number of Horovod processes.
    """
    size = MPI_COMMON_LIB_CTYPES.horovod_size()
    if size == -1:
        raise ValueError(
            'Horovod has not been initialized; use hvd.init().')
    return size


def local_size():
    """A function that returns the number of Horovod processes within the
    node the current process is running on.

    Returns:
      An integer scalar containing the number of local Horovod processes.
    """
    local_size = MPI_COMMON_LIB_CTYPES.horovod_local_size()
    if local_size == -1:
        raise ValueError(
            'Horovod has not been initialized; use hvd.init().')
    return local_size


def rank():
    """A function that returns the Horovod rank of the calling process.

    Returns:
      An integer scalar with the Horovod rank of the calling process.
    """
    rank = MPI_COMMON_LIB_CTYPES.horovod_rank()
    if rank == -1:
        raise ValueError(
            'Horovod has not been initialized; use hvd.init().')
    return rank


def local_rank():
    """A function that returns the local Horovod rank of the calling process, within the
    node that it is running on. For example, if there are seven processes running
    on a node, their local ranks will be zero through six, inclusive.

    Returns:
      An integer scalar with the local Horovod rank of the calling process.
    """
    local_rank = MPI_COMMON_LIB_CTYPES.horovod_local_rank()
    if local_rank == -1:
        raise ValueError(
            'Horovod has not been initialized; use hvd.init().')
    return local_rank


def mpi_threads_supported():
    """A function that returns a flag indicating whether MPI multi-threading is supported.

    If MPI multi-threading is supported, users may mix and match Horovod usage with other
    MPI libraries, such as `mpi4py`.

    Returns:
      A boolean value indicating whether MPI multi-threading is supported.
    """
    mpi_threads_supported = MPI_COMMON_LIB_CTYPES.mpi_threads_supported()
    if mpi_threads_supported == -1:
        raise ValueError(
            'Horovod has not been initialized; use hvd.init().')
    return bool(mpi_threads_supported)
