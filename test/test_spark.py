# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import pytest
import re
import subprocess
import time
import torch
import unittest
import warnings

from horovod.run.mpi_run import _get_mpi_implementation_flags
import horovod.spark
import horovod.torch as hvd

from mock import MagicMock


@contextlib.contextmanager
def spark(app, cores=2, *args):
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf().setAppName(app).setMaster('local[{}]'.format(cores))
    session = SparkSession \
        .builder \
        .config(conf=conf) \
        .getOrCreate()

    try:
        yield session
    finally:
        session.stop()


@contextlib.contextmanager
def os_environ(env):
    old = os.environ
    try:
        os.environ = env
        yield
    finally:
        os.environ = old


class SparkTests(unittest.TestCase):
    """
    Tests for horovod.spark.run().
    """

    def __init__(self, *args, **kwargs):
        super(SparkTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def run(self, result=None):
        if os.environ.get('OMPI_COMM_WORLD_RANK', '0') != '0':
            # Running in MPI as a rank > 0, ignore.
            return

        if 'Open MPI' not in str(subprocess.check_output('mpirun --version', shell=True)):
            return

        super(SparkTests, self).run(result)

    def test_happy_run(self):
        def fn():
            hvd.init()
            res = hvd.allgather(torch.tensor([hvd.rank()])).tolist()
            return res, hvd.rank()

        with spark('test_happy_run'):
            res = horovod.spark.run(fn, env={'PATH': os.environ.get('PATH')}, verbose=0)
            self.assertListEqual([([0, 1], 0), ([0, 1], 1)], res)

    def test_timeout(self):
        with spark('test_timeout'):
            with pytest.raises(Exception, match='^Timed out waiting for Spark tasks to start.'):
                horovod.spark.run(None, num_proc=4, start_timeout=5,
                                  env={'PATH': os.environ.get('PATH')},
                                  verbose=0)

    def test_mpirun_not_found(self):
        start = time.time()
        with spark('test_mpirun_not_found'):
            with pytest.raises(Exception, match='^mpirun failed with exit code 127$'):
                horovod.spark.run(None, env={'PATH': '/nonexistent'}, verbose=0)
        self.assertLessEqual(time.time() - start, 10, 'Failure propagation took too long')

    """
    Test that horovod.spark.run invokes mpi_run properly.
    """
    def test_spark_run_func(self):
        env = {'env1': 'val1', 'env2': 'val2'}
        expected_env = '-x env1 -x env2'
        extra_mpi_args = '<extra args go here>'
        self.do_test_spark_run_func(num_proc=2, extra_mpi_args=extra_mpi_args,
                                    env=env, stdout='<stdout>', stderr='<stderr>',
                                    cores=4, expected_np=2, expected_env=expected_env)

    """
    Test that horovod.spark.run defaults num_proc to spark parallelism.
    """
    def test_spark_run_func_defaults_num_proc_to_spark_cores(self):
        self.do_test_spark_run_func(num_proc=None, cores=2, expected_np=2)

    """
    Test that horovod.spark.run defaults env to the full system env.
    """
    def test_spark_run_func_defaults_env_to_os_env(self):
        env = {'env1': 'val1', 'env2': 'val2'}
        expected_env = '-x env1 -x env2'

        with os_environ(env):
            self.do_test_spark_run_func(env=None, expected_env=expected_env)

    """
    Test that horovod.spark.run raises and exception on non-zero exit code of mpi_run.
    """
    def test_spark_run_func_with_non_zero_exit(self):
        run_func = MagicMock(return_value=1)

        def fn():
            return 1

        with spark('test_spark_run_func', cores=4):
            with pytest.raises(Exception, match='^mpirun failed with exit code 1$') as e:
                horovod.spark.run(fn, verbose=0, run_func=run_func)

    """
    Performs the actual horovod.spark.run test.
    """
    def do_test_spark_run_func(self, args=(), kwargs={}, num_proc=1, extra_mpi_args=None, env={},
                               stdout=None, stderr=None, verbose=0,
                               cores=2, expected_np=1, expected_env=''):
        def fn():
            return 1

        run_func = MagicMock(return_value=0)

        with spark('test_spark_run_func', cores=cores):
            with pytest.raises(Exception) as e:
                # we need to timeout horovod because our mocked run_func will block spark otherwise
                # this raises above exception, but allows us to catch run_func arguments
                horovod.spark.run(fn, args=args, kwargs=kwargs,
                                  num_proc=num_proc, start_timeout=1,
                                  extra_mpi_args=extra_mpi_args, env=env,
                                  stdout=stdout, stderr=stderr, verbose=verbose,
                                  run_func=run_func)

        self.assertFalse(str(e.value).startswith('Timed out waiting for Spark tasks to start.'),
                         'Spark timed out before mpi_run was called, test setup is broken.')
        self.assertEqual(str(e.value), 'Spark job has failed, see the error above.')

        mpi_flags = _get_mpi_implementation_flags()
        self.assertIsNotNone(mpi_flags)
        expected_command = ('mpirun '
                            '--allow-run-as-root --tag-output '
                            '-np {expected_np} -H [^ ]+ '
                            '-bind-to none -map-by slot '
                            '{mpi_flags}  '
                            '-mca btl_tcp_if_include [^ ]+ -x NCCL_SOCKET_IFNAME=[^ ]+  '
                            '-x _HOROVOD_SECRET_KEY {expected_env}'
                            '{extra_mpi_args} '
                            '-x NCCL_DEBUG=INFO '
                            r'-mca plm_rsh_agent "[^"]+python[\d]* -m horovod.spark.driver.mpirun_rsh [^ ]+ [^ ]+" '
                            r'[^"]+python[\d]* -m horovod.spark.task.mpirun_exec_fn [^ ]+ [^ ]+'.format(
                                expected_np=expected_np,
                                expected_env=expected_env + ' ' if expected_env else '',
                                mpi_flags=' '.join(mpi_flags),
                                extra_mpi_args=extra_mpi_args if extra_mpi_args else ''))

        run_func.assert_called_once()
        run_func_args, run_func_kwargs = run_func.call_args
        actual_command = run_func_kwargs.get('command')
        actual_env = run_func_kwargs.get('env')
        actual_stdout = run_func_kwargs.get('stdout')
        actual_stderr = run_func_kwargs.get('stderr')
        actual_secret = actual_env.pop('_HOROVOD_SECRET_KEY', None)

        # for better comparison replace sections in actual_command that change across runs / hosts
        for replacement in ('-H [^ ]+', '-mca btl_tcp_if_include [^ ]+', '-x NCCL_SOCKET_IFNAME=[^ ]+',
                            r'"[^"]+python[\d]*', r' [^"]+python[\d]*',
                            '-m horovod.spark.driver.mpirun_rsh [^ ]+ [^ ]+"',
                            '-m horovod.spark.task.mpirun_exec_fn [^ ]+ [^ ]+'):
            actual_command = re.sub(replacement, replacement, actual_command, 1)

        self.assertEqual(run_func_args, ())
        self.assertEqual(actual_command, expected_command)
        if env:
            self.assertEqual(actual_env, env)
        else:
            self.assertIsNotNone(actual_env)
        self.assertIsNotNone(actual_secret)
        self.assertTrue(len(actual_secret) > 0)
        self.assertEqual(actual_stdout, stdout)
        self.assertEqual(actual_stderr, stderr)
