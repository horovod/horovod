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
import time

import mock

from horovod.run.util.threads import in_thread


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


def delay(func, seconds):
    """Delays the execution of func in a separate thread by given seconds."""
    def fn():
        time.sleep(seconds)
        func()

    t = in_thread(target=fn)


def wait(func, timeout=None):
    """Wait for func to return True until timeout."""
    start = int(time.time())
    while not func():
        time.sleep(0.1)
        if timeout is not None and int(time.time()) - start > timeout:
            raise TimeoutError('Timed out waiting for func to return True')


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
def undo(fn):
    try:
        yield
    finally:
        fn()


@contextlib.contextmanager
def is_built(gloo_is_built, mpi_is_built):
    """
    Patches the gloo_built and mpi_built methods called from horovod.run.run.run_controller
    to return the given booleans. That method is used by horovod.spark.run to determine which
    controller to use. Patching these methods allows to test horovod.spark.run without an MPI
    implementation to be installed.

    :param gloo_is_built: boolean returned by gloo_built
    :param mpi_is_built: boolean returned by mpi_built
    :return: mocked gloo_built and mpi_built methods
    """
    with mock.patch("horovod.run.runner.gloo_built", return_value=gloo_is_built) as g:
        with mock.patch("horovod.run.runner.mpi_built", return_value=mpi_is_built) as m:
            yield g, m


@contextlib.contextmanager
def mpi_implementation_flags(flags=["--mock-mpi-impl-flags"],
                             binding_args=["--mock-mpi-binding-args"]):
    """
    Patches the _get_mpi_implementation_flags method used by horovod.run.mpi_run to retrieve
    MPI implementation specific command line flags. Patching this method allows to test mpi_run
    without an MPI implementation to be installed.

    :param flags: mock flags
    :return: the mocked method
    """
    with mock.patch("horovod.run.mpi_run._get_mpi_implementation_flags", return_value=(flags, binding_args)) as m:
        yield m


@contextlib.contextmanager
def lsf_and_jsrun(lsf_exists, jsrun_installed):
    """
    Patches the lsf.LSFUtils.using_lsf and is_jsrun_installed methods called from
    horovod.run.run.run_controller to return the given booleans.
    :param lsf_exists: boolean returned by lsf.LSFUtils.using_lsf
    :param jsrun_installed: boolean returned by is_jsrun_installed
    :return: mocked methods
    """
    with mock.patch("horovod.run.runner.lsf.LSFUtils.using_lsf", return_value=lsf_exists) as u:
        with mock.patch("horovod.run.runner.is_jsrun_installed", return_value=jsrun_installed) as i:
            yield u, i
