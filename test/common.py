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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import shutil
import sys
import tempfile

from mock import patch


def mpi_env_rank_and_size():
    """Get MPI rank and size from environment variables and return them as a
    tuple of integers.

    Most MPI implementations have an `mpirun` or `mpiexec` command that will
    run an MPI executable and set up all communication necessary between the
    different processors. As part of that set up, they will set environment
    variables that contain the rank and size of the MPI_COMM_WORLD
    communicator. We can read those environment variables from Python in order
    to ensure that `hvd.rank()` and `hvd.size()` return the expected values.

    Since MPI is just a standard, not an implementation, implementations
    typically choose their own environment variable names. This function tries
    to support several different implementation, but really it only needs to
    support whatever implementation we want to use for the TensorFlow test
    suite.

    If this is not running under MPI, then defaults of rank zero and size one
    are returned. (This is appropriate because when you call MPI_Init in an
    application not started with mpirun, it will create a new independent
    communicator with only one process in it.)
    """
    rank_env = 'PMI_RANK OMPI_COMM_WORLD_RANK'.split()
    size_env = 'PMI_SIZE OMPI_COMM_WORLD_SIZE'.split()

    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)

    # Default to rank zero and size one if there are no environment variables
    return 0, 1


@contextlib.contextmanager
def tempdir():
    dirpath = tempfile.mkdtemp()
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


@contextlib.contextmanager
def temppath():
    path = tempfile.mktemp()
    try:
        yield path
    finally:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)


@contextlib.contextmanager
def override_args(tool=None, *args):
    old = sys.argv[:]
    try:
        if tool:
            sys.argv[0] = tool
        sys.argv[1:] = args
        yield
    finally:
        sys.argv = old


@contextlib.contextmanager
def override_env(env):
    old = os.environ
    try:
        os.environ = env
        yield
    finally:
        os.environ = old


@contextlib.contextmanager
def is_built(gloo_is_built, mpi_is_built):
    """
    Patches the gloo_built and mpi_built methods called from horovod.run.run.run_controller
    to return the given booleans.
    :param gloo_is_built: boolean returned by gloo_built
    :param mpi_is_built: boolean returned by mpi_built
    :return: patched context manager
    """
    with patch(target="horovod.run.run.gloo_built") as gloo_built_mock:
        gloo_built_mock.return_value = gloo_is_built
        with patch(target="horovod.run.run.mpi_built") as mpi_built_mock:
            mpi_built_mock.return_value = mpi_is_built
            yield


@contextlib.contextmanager
def js_installed(js_is_installed):
    """
    Patches the lsf.LSFUtils.using_lsf and is_jsrun_installed methods called from
    horovod.run.run.run_controller to return the given booleans.
    :param js_is_installed: boolean returned by lsf.LSFUtils.using_lsf and is_jsrun_installed
    :return: patched context manager
    """
    with patch(target="horovod.run.run.lsf.LSFUtils.using_lsf") as using_lsf_mock:
        using_lsf_mock.return_value = js_is_installed
        with patch(target="horovod.run.run.is_jsrun_installed") as is_jsrun_installed_mock:
            is_jsrun_installed_mock.return_value = js_is_installed
            yield
