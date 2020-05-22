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

import copy
import itertools
import os
import platform
import pytest
import re
import sys
import threading
import time
import unittest
import warnings

from distutils.version import LooseVersion

import mock
import torch

import pyspark

from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, FloatType, IntegerType, \
    NullType, StructField, StructType

import horovod.spark
import horovod.torch as hvd

from horovod.common.util import gloo_built, mpi_built
from horovod.run.common.util import codec, secret, safe_shell_exec
from horovod.run.common.util import settings as hvd_settings
from horovod.run.mpi_run import is_open_mpi
from horovod.spark.common import constants, util
from horovod.spark.common.store import HDFSStore
from horovod.spark.driver.rsh import rsh
from horovod.spark.task import get_available_devices, gloo_exec_fn, mpirun_exec_fn
from horovod.spark.driver.driver_service import SparkDriverService, SparkDriverClient
from horovod.spark.task.task_service import SparkTaskService, SparkTaskClient

from spark_common import spark_session, create_test_data_from_schema, create_xor_data, local_store

from common import is_built, mpi_implementation_flags, tempdir, override_env, undo, delay


# Spark will fail to initialize correctly locally on Mac OS without this
if platform.system() == 'Darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'


class SparkTests(unittest.TestCase):
    """
    Tests for horovod.spark.run().
    """

    def __init__(self, *args, **kwargs):
        super(SparkTests, self).__init__(*args, **kwargs)
        self.maxDiff = None
        warnings.simplefilter('module')

    def run(self, result=None):
        # These unit tests should not be run with horovodrun as some tests
        # setup their own Horovod cluster, where both will then interfere.
        if 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ:
            self.skipTest("These tests should not be executed via horovodrun, just pytest")

        super(SparkTests, self).run(result)

    @pytest.mark.skipif(sys.version_info >= (3, 0), reason='Skipped on Python 3')
    def test_run_throws_on_python2(self):
        if not gloo_built():
            self.skipTest("Gloo is not available")

        with pytest.raises(Exception, match='^Horovod on Spark over Gloo only supported on Python3$'):
            self.do_test_happy_run(use_mpi=False, use_gloo=True)

    """
    Test that horovod.spark.run works properly in a simple setup using MPI.
    """
    def test_happy_run_with_mpi(self):
        if not (mpi_built() and is_open_mpi()):
            self.skipTest("Open MPI is not available")

        self.do_test_happy_run(use_mpi=True, use_gloo=False)

    """
    Test that horovod.spark.run works properly in a simple setup using Gloo.
    """
    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason='Horovod on Spark over Gloo only supported on Python3')
    def test_happy_run_with_gloo(self):
        if not gloo_built():
            self.skipTest("Gloo is not available")

        self.do_test_happy_run(use_mpi=False, use_gloo=True)

    """
    Actually tests that horovod.spark.run works properly in a simple setup.
    """
    def do_test_happy_run(self, use_mpi, use_gloo):
        def fn():
            hvd.init()
            res = hvd.allgather(torch.tensor([hvd.rank()])).tolist()
            return res, hvd.rank()

        with spark_session('test_happy_run'):
            with is_built(gloo_is_built=use_gloo, mpi_is_built=use_mpi):
                res = horovod.spark.run(fn, start_timeout=10,
                                        use_mpi=use_mpi, use_gloo=use_gloo,
                                        verbose=2)
                self.assertListEqual([([0, 1], 0), ([0, 1], 1)], res)

    """
    Test that horovod.spark.run times out when it does not start up fast enough using MPI.
    """
    def test_timeout_with_mpi(self):
        if not (mpi_built() and is_open_mpi()):
            self.skipTest("Open MPI is not available")

        self.do_test_timeout(use_mpi=True, use_gloo=False)

    """
    Test that horovod.spark.run times out when it does not start up fast enough using Gloo.
    """
    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason='Horovod on Spark over Gloo only supported on Python3')
    def test_timeout_with_gloo(self):
        if not gloo_built():
            self.skipTest("Gloo is not available")

        self.do_test_timeout(use_mpi=False, use_gloo=True)

    """
    Actually tests that horovod.spark.run times out when it does not start up fast enough.
    """
    def do_test_timeout(self, use_mpi, use_gloo):
        # with 2 cores and 4 num_proc this spark run will never start up completely and time out
        with spark_session('test_timeout', cores=2):
            with is_built(gloo_is_built=use_gloo, mpi_is_built=use_mpi):
                with pytest.raises(Exception, match='^Timed out waiting for Spark tasks to start.'):
                    horovod.spark.run(None, num_proc=4, start_timeout=5,
                                      use_mpi=use_mpi, use_gloo=use_gloo,
                                      verbose=0)

    """
    Test that horovod.spark.run fails with meaningful exception when mpirun cannot be found.
    This test does not require MPI to be installed.
    """
    def test_mpirun_not_found(self):
        start = time.time()
        with spark_session('test_mpirun_not_found'):
            with is_built(gloo_is_built=False, mpi_is_built=True):
                with mpi_implementation_flags():
                    with pytest.raises(Exception, match='^mpirun failed with exit code 127$'):
                        horovod.spark.run(None, start_timeout=20, env={'PATH': '/nonexistent'}, verbose=0)
        self.assertLessEqual(time.time() - start, 10, 'Failure propagation took too long')

    """
    Test that horovod.spark.run uses MPI properly.
    """
    def test_spark_run_with_mpi(self):
        with mpi_implementation_flags():
            self.do_test_spark_run(use_mpi=True, use_gloo=False)

    """
    Test that horovod.spark.run uses Gloo properly.
    """
    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason='Horovod on Spark over Gloo only supported on Python3')
    def test_spark_run_with_gloo(self):
        self.do_test_spark_run(use_mpi=False, use_gloo=True)

    """
    Actually tests that horovod.spark.run invokes mpi_run properly.
    """
    def do_test_spark_run(self, use_mpi, use_gloo):
        env = {'env1': 'val1', 'env2': 'val2'}
        expected_env = '-x env1 -x env2'
        extra_mpi_args = '<extra args go here>'
        with is_built(gloo_is_built=use_gloo, mpi_is_built=use_mpi):
            self._do_test_spark_run(num_proc=2, use_mpi=use_mpi, use_gloo=use_gloo,
                                    extra_mpi_args=extra_mpi_args,
                                    env=env, stdout='<stdout>', stderr='<stderr>',
                                    cores=2, expected_np=2, expected_env=expected_env)

    """
    Test that horovod.spark.run does not default to spark parallelism given num_proc using MPI.
    """
    def test_spark_run_num_proc_precedes_spark_cores_with_mpi(self):
        with mpi_implementation_flags():
            self.do_test_spark_run_num_proc_precedes_spark_cores(use_mpi=True, use_gloo=False)

    """
    Test that horovod.spark.run does not default to spark parallelism given num_proc using Gloo.
    """
    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason='Horovod on Spark over Gloo only supported on Python3')
    def test_spark_run_num_proc_precedes_spark_cores_with_gloo(self):
        self.do_test_spark_run_num_proc_precedes_spark_cores(use_mpi=False, use_gloo=True)

    """
    Actually tests that horovod.spark.run does not default to spark parallelism given num_proc.
    """
    def do_test_spark_run_num_proc_precedes_spark_cores(self, use_mpi, use_gloo):
        self._do_test_spark_run(num_proc=1, cores=2, expected_np=1,
                                use_mpi=use_mpi, use_gloo=use_gloo)

    """
    Tests that horovod.spark.run invokes mpi_run with a given PATH properly.
    """
    def test_spark_run_with_path_with_mpi(self):
        env = {'env1': 'val1', 'env2': 'val2', 'PATH': 'path'}
        expected_env = '-x PATH -x env1 -x env2'
        extra_mpi_args = '<extra args go here>'
        with is_built(gloo_is_built=False, mpi_is_built=True):
            self._do_test_spark_run(num_proc=2, use_mpi=True, use_gloo=False,
                                    extra_mpi_args=extra_mpi_args,
                                    env=env, stdout='<stdout>', stderr='<stderr>',
                                    cores=4, expected_np=2, expected_env=expected_env)

    """
    Test that horovod.spark.run defaults num_proc to spark parallelism using MPI.
    """
    def test_spark_run_defaults_num_proc_to_spark_cores_with_mpi(self):
        with mpi_implementation_flags():
            self.do_test_spark_run_defaults_num_proc_to_spark_cores(use_mpi=True, use_gloo=False)

    """
    Test that horovod.spark.run defaults num_proc to spark parallelism using Gloo.
    """
    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason='Horovod on Spark over Gloo only supported on Python3')
    def test_spark_run_defaults_num_proc_to_spark_cores_with_gloo(self):
        self.do_test_spark_run_defaults_num_proc_to_spark_cores(use_mpi=False, use_gloo=True)

    """
    Actually tests that horovod.spark.run defaults num_proc to spark parallelism.
    """
    def do_test_spark_run_defaults_num_proc_to_spark_cores(self, use_mpi, use_gloo):
        self._do_test_spark_run(num_proc=None, cores=2, expected_np=2,
                                use_mpi=use_mpi, use_gloo=use_gloo)

    """
    Test that horovod.spark.run defaults env to the full system env using MPI.
    """
    def test_spark_run_does_not_default_env_to_os_env_with_mpi(self):
        with mpi_implementation_flags():
            self.do_test_spark_run_does_not_default_env_to_os_env(use_mpi=True, use_gloo=False)

    """
    Test that horovod.spark.run defaults env to the full system env using Gloo.
    """
    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason='Horovod on Spark over Gloo only supported on Python3')
    def test_spark_run_does_not_default_env_to_os_env_with_gloo(self):
        self.do_test_spark_run_does_not_default_env_to_os_env(use_mpi=False, use_gloo=True)

    """
    Actually tests that horovod.spark.run defaults env to the full system env.
    """
    def do_test_spark_run_does_not_default_env_to_os_env(self, use_mpi, use_gloo):
        env = {'env1': 'val1', 'env2': 'val2'}
        expected_env = ''

        with override_env(env):
            self._do_test_spark_run(env=None, use_mpi=use_mpi, use_gloo=use_gloo,
                                    expected_env=expected_env)

    """
    Test that horovod.spark.run raises an exception on non-zero exit code of mpi_run using MPI.
    """
    def test_spark_run_with_non_zero_exit_with_mpi(self):
        expected = '^mpirun failed with exit code 1$'
        with mpi_implementation_flags():
            self.do_test_spark_run_with_non_zero_exit(use_mpi=True, use_gloo=False,
                                                      expected=expected)

    """
    Test that horovod.spark.run raises an exception on non-zero exit code of mpi_run using Gloo.
    """
    @pytest.mark.skipif(sys.version_info < (3, 0),
                        reason='Horovod on Spark over Gloo only supported on Python3')
    def test_spark_run_with_non_zero_exit_with_gloo(self):
        expected = '^Gloo job detected that one or more processes exited with non-zero ' \
                   'status, thus causing the job to be terminated. The first process ' \
                   'to do so was:\nProcess name: 0\nExit code: 1$'
        self.do_test_spark_run_with_non_zero_exit(use_mpi=False, use_gloo=True,
                                                  expected=expected)

    """
    Actually tests that horovod.spark.run raises an exception on non-zero exit code of mpi_run.
    """
    def do_test_spark_run_with_non_zero_exit(self, use_mpi, use_gloo, expected):
        def fn():
            return 0

        def mpi_impl_flags(tcp):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"]

        def gloo_exec_command_fn(driver_addresses, key, settings, env):
            def _exec_command(command, alloc_info, event):
                return 1, alloc_info.rank
            return _exec_command

        with mock.patch("horovod.run.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.run.mpi_run.safe_shell_exec.execute", return_value=1):
                with mock.patch("horovod.spark.gloo_run._exec_command_fn", side_effect=gloo_exec_command_fn):
                    with spark_session('test_spark_run'):
                        with is_built(gloo_is_built=use_gloo, mpi_is_built=use_mpi):
                            with pytest.raises(Exception, match=expected):
                                horovod.spark.run(fn, start_timeout=10, use_mpi=use_mpi, use_gloo=use_gloo, verbose=2)

    """
    Performs an actual horovod.spark.run test using MPI or Gloo.
    """
    def _do_test_spark_run(self, args=(), kwargs={}, num_proc=1, extra_mpi_args=None,
                           env=None, use_mpi=None, use_gloo=None,
                           stdout=None, stderr=None, verbose=2,
                           cores=2, expected_np=1, expected_env=''):
        if use_mpi:
            self._do_test_spark_run_with_mpi(args, kwargs, num_proc, extra_mpi_args, env,
                                             stdout, stderr, verbose, cores,
                                             expected_np, expected_env)
        if use_gloo:
            self._do_test_spark_run_with_gloo(args, kwargs, num_proc, extra_mpi_args, env,
                                              stdout, stderr, verbose, cores,
                                              expected_np)

    """
    Performs an actual horovod.spark.run test using MPI.
    """
    def _do_test_spark_run_with_mpi(self, args=(), kwargs={}, num_proc=1, extra_mpi_args=None,
                                    env=None, stdout=None, stderr=None, verbose=2,
                                    cores=2, expected_np=1, expected_env=''):
        if env is None:
            env = {}

        def fn():
            return 1

        def mpi_impl_flags(tcp):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"]

        def exception(*args, **argv):
            raise Exception('Test Exception')

        with mock.patch("horovod.run.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.run.mpi_run.safe_shell_exec.execute", side_effect=exception) as execute:
                with spark_session('test_spark_run', cores=cores):
                    with is_built(gloo_is_built=False, mpi_is_built=True):
                        # we make the run fail just after we caught our mocked method calls
                        with pytest.raises(Exception) as e:
                            horovod.spark.run(fn, args=args, kwargs=kwargs,
                                              num_proc=num_proc, start_timeout=10,
                                              use_mpi=True, use_gloo=False,
                                              extra_mpi_args=extra_mpi_args, env=env,
                                              stdout=stdout, stderr=stderr, verbose=verbose)

                self.assertFalse(str(e.value).startswith('Timed out waiting for Spark tasks to start.'),
                                 'Spark timed out before mpi_run was called, test setup is broken.')
                self.assertEqual(str(e.value), 'Test Exception')

                # call the mocked _get_mpi_implementation_flags method
                mpi_flags, binding_args = horovod.run.mpi_run._get_mpi_implementation_flags(False)
                self.assertIsNotNone(mpi_flags)
                expected_command = ('mpirun '
                                    '--allow-run-as-root --tag-output '
                                    '-np {expected_np} -H [^ ]+ '
                                    '{binding_args} '
                                    '{mpi_flags}  '
                                    '-mca btl_tcp_if_include [^ ]+ -x NCCL_SOCKET_IFNAME=[^ ]+  '
                                    '{expected_env} '
                                    '{extra_mpi_args} '
                                    '-x NCCL_DEBUG=INFO '
                                    r'-mca plm_rsh_agent "[^"]+python[0-9.]* -m horovod.spark.driver.mpirun_rsh [^ ]+ [^ ]+" '
                                    r'[^"]+python[0-9.]* -m horovod.spark.task.mpirun_exec_fn [^ ]+ [^ ]+'.format(
                    expected_np=expected_np,
                    binding_args=' '.join(binding_args),
                    expected_env=expected_env if expected_env else '',
                    mpi_flags=' '.join(mpi_flags),
                    extra_mpi_args=extra_mpi_args if extra_mpi_args else ''))

                execute.assert_called_once()
                execute_args, execute_kwargs = execute.call_args

        self.assertIsNotNone(execute_args)
        actual_command = execute_args[0]
        actual_env = execute_kwargs.get('env')
        actual_stdout = execute_kwargs.get('stdout')
        actual_stderr = execute_kwargs.get('stderr')

        # the settings should not contain the key
        serialized_settings = actual_command.split(' ')[-1]
        actual_settings = codec.loads_base64(serialized_settings)
        self.assertIsNone(actual_settings.key)

        # the settings for the rsh agent should not contain the key
        actual_rsh_command_match = re.match('.* -mca plm_rsh_agent "([^"]+)" .*', actual_command)
        self.assertTrue(actual_rsh_command_match, 'could not extract rsh agent from mpirun command')
        actual_rsh_command = actual_rsh_command_match.group(1)
        serialized_rsh_settings = actual_rsh_command.split(' ')[-1]
        actual_rsh_settings = codec.loads_base64(serialized_rsh_settings)
        self.assertIsNone(actual_rsh_settings.key)

        # for better comparison replace sections in actual_command that change across runs / hosts
        for replacement in ['-H [^ ]+', '-mca btl_tcp_if_include [^ ]+', '-x NCCL_SOCKET_IFNAME=[^ ]+',
                            r'"[^"]+python[0-9.]*', r' [^"]+python[0-9.]*',
                            '-m horovod.spark.driver.mpirun_rsh [^ ]+ [^ ]+"',
                            '-m horovod.spark.task.mpirun_exec_fn [^ ]+ [^ ]+']:
            actual_command = re.sub(replacement, replacement, actual_command, 1)

        actual_secret = actual_env.pop(secret.HOROVOD_SECRET_KEY, None)
        self.assertEqual(expected_command, actual_command)
        if env:
            if 'PATH' not in env and 'PATH' in os.environ:
                env = copy.copy(env)
                env['PATH'] = os.environ['PATH']
            self.assertEqual(env, actual_env)
        else:
            self.assertIsNotNone(actual_env)
        self.assertIsNotNone(actual_secret)
        self.assertTrue(len(actual_secret) > 0)
        self.assertEqual(stdout, actual_stdout)
        self.assertEqual(stderr, actual_stderr)

    """
    Performs an actual horovod.spark.run test using Gloo.
    """
    def _do_test_spark_run_with_gloo(self, args=(), kwargs={}, num_proc=1, extra_mpi_args=None,
                                     env=None, stdout=None, stderr=None, verbose=2,
                                     cores=2, expected_np=1):
        if env is None:
            env = {}

        def fn():
            return 1

        def _exec_command(command, alloc_info, event):
            return 1, alloc_info.rank

        exec_command = mock.MagicMock(side_effect=_exec_command)
        gloo_exec_command_fn = mock.MagicMock(return_value=exec_command)

        with mock.patch("horovod.spark.gloo_run._exec_command_fn", side_effect=gloo_exec_command_fn):
            with spark_session('test_spark_run', cores=cores):
                with is_built(gloo_is_built=True, mpi_is_built=False):
                    # we make the run fail just after we caught our mocked method calls
                    with pytest.raises(Exception) as e:
                        # we need to timeout horovod because our mocked methods will block Spark
                        # this raises above exception, but allows us to catch execute's arguments
                        horovod.spark.run(fn, args=args, kwargs=kwargs,
                                          num_proc=num_proc, start_timeout=10,
                                          use_mpi=False, use_gloo=True,
                                          extra_mpi_args=extra_mpi_args, env=env,
                                          stdout=stdout, stderr=stderr, verbose=verbose)

        self.assertFalse(str(e.value).startswith('Timed out waiting for Spark tasks to start.'),
                         'Spark timed out before mpi_run was called, test setup is broken.')
        self.assertRegexpMatches(str(e.value),
                                 '^Gloo job detected that one or more processes exited with non-zero status, '
                                 'thus causing the job to be terminated. The first process to do so was:\n'
                                 'Process name: [0-9]\nExit code: 1+\n$')

        num_proc = cores if num_proc is None else num_proc
        self.assertEqual(expected_np, num_proc)
        self.assertEqual(1, gloo_exec_command_fn.call_count)
        _, _, _, call_env = gloo_exec_command_fn.call_args[0]
        self.assertEqual(env or {}, call_env)
        self.assertEqual({}, gloo_exec_command_fn.call_args[1])
        self.assertEqual(num_proc, exec_command.call_count)
        self.assertEqual(num_proc, len(exec_command.call_args_list))

        # expect all ranks exist
        # exec_command.call_args_list is [(args, kwargs)] with args = (command, alloc_info, event)
        actual_ranks = sorted([call_args[0][1].rank for call_args in exec_command.call_args_list])
        self.assertEqual(list(range(0, num_proc)), actual_ranks)

        first_event = exec_command.call_args_list[0][0][2]
        first_host = exec_command.call_args_list[0][0][1].hostname
        for call_args in exec_command.call_args_list:
            # all events are the same instance
            self.assertEqual(first_event, call_args[0][2])
            # all kwargs are empty
            self.assertEqual({}, call_args[1])

            # all alloc_info refer to the same host
            alloc_info = call_args[0][1]
            self.assertEqual(first_host, alloc_info.hostname)
            self.assertEqual(num_proc, alloc_info.size)
            self.assertEqual(num_proc, alloc_info.local_size)
            self.assertEqual(alloc_info.local_rank, alloc_info.rank)

            # command fully derived from alloc_info
            expected_command = ('HOROVOD_RANK={rank} '
                                'HOROVOD_SIZE={size} '
                                'HOROVOD_LOCAL_RANK={local_rank} '
                                'HOROVOD_LOCAL_SIZE={local_size} '
                                'HOROVOD_CROSS_RANK=0 '
                                'HOROVOD_CROSS_SIZE=1  '
                                'PYTHONUNBUFFERED=1 '
                                'HOROVOD_GLOO_RENDEZVOUS_ADDR=[^ ]+ '
                                'HOROVOD_GLOO_RENDEZVOUS_PORT=[0-9]+ '
                                'HOROVOD_CONTROLLER=gloo '
                                'HOROVOD_CPU_OPERATIONS=gloo '
                                'HOROVOD_GLOO_IFACE=[^ ]+ '
                                'NCCL_SOCKET_IFNAME=[^ ]+ '
                                '[^ ]+python[0-9.]* -m horovod.spark.task.gloo_exec_fn '
                                '[^ ]+ [^ ]+$'.format(rank=alloc_info.rank,
                                                      size=alloc_info.size,
                                                      local_rank=alloc_info.local_rank,
                                                      local_size=alloc_info.local_size,
                                                      np=num_proc))

            actual_command = call_args[0][0]

            # the settings should not contain the key
            serialized_settings = actual_command.split(' ')[-1]
            actual_settings = codec.loads_base64(serialized_settings)
            self.assertIsNone(actual_settings.key)

            # for better comparison replace sections in actual_command that change across runs / hosts
            for replacement in ['_HOROVOD_SECRET_KEY=[^ ]+',
                                'HOROVOD_GLOO_RENDEZVOUS_ADDR=[^ ]+',
                                'HOROVOD_GLOO_RENDEZVOUS_PORT=[0-9]+',
                                'HOROVOD_GLOO_IFACE=[^ ]+',
                                'NCCL_SOCKET_IFNAME=[^ ]+',
                                '[^ ]+python[0-9.]*',
                                '[^ ]+ [^ ]+$']:
                actual_command = re.sub(replacement, replacement, actual_command, 1)

            self.assertEqual(expected_command, actual_command)

    def test_rsh_with_zero_exit_code(self):
        self.do_test_rsh('true', 0)

    def test_rsh_with_non_zero_exit_code(self):
        self.do_test_rsh('false', 1)

    def test_rsh_event(self):
        sleep = 10
        command = 'sleep {}'.format(sleep)
        event = threading.Event()
        delay(lambda: event.set(), 1.0)

        start = time.time()
        self.do_test_rsh(command, 143, event=event)
        duration = time.time() - start

        self.assertGreaterEqual(duration, 1.0)
        self.assertLess(duration, 2.00 + safe_shell_exec.GRACEFUL_TERMINATION_TIME_S,
                        'sleep should not finish')
        self.assertGreater(sleep, 2.00 + safe_shell_exec.GRACEFUL_TERMINATION_TIME_S,
                           'sleep should be large enough')

    def do_test_rsh(self, command, expected_result, event=None):
        def fn():
            return 0

        # setup infrastructure so we can call rsh
        key = secret.make_secret_key()
        host_hash = 'test-host'
        driver = SparkDriverService(1, fn, (), {}, key, None)
        client = SparkDriverClient(driver.addresses(), key, 2)
        task = SparkTaskService(0, key, None, 2)
        client.register_task(0, task.addresses(), host_hash)
        settings = hvd_settings.Settings(verbose=2, key=key)
        env = {}

        res = rsh(driver.addresses(), key, host_hash, command, env, 0, settings.verbose, False, event=event)
        self.assertEqual(expected_result, res)

    def test_mpirun_exec_fn(self):
        bool_values = [False, True]
        for work_dir_env_set, python_path_is_set, hvd_python_path_is_set in \
            itertools.product(bool_values, bool_values, bool_values):
            with tempdir() as tmp_path:
                driver = mock.MagicMock()
                settings = mock.MagicMock()
                settings.verbose = 2

                test_env = {}
                test_dir = os.getcwd()
                test_sys_path = copy.copy(sys.path)

                def reset():
                    os.chdir(test_dir)
                    sys.path = test_sys_path

                if work_dir_env_set:
                    # ask mpirun_exec_fn to change cwd to test_dir
                    test_env['HOROVOD_SPARK_WORK_DIR'] = test_dir
                if python_path_is_set:
                    test_python_path = ['python/path', 'python/path2']
                    test_env['PYTHONPATH'] = os.pathsep.join(test_python_path)
                if hvd_python_path_is_set:
                    # ingest tmp_path into workers PYTHONPATH
                    test_horovod_python_path = ['horovod', 'horovod/python']
                    test_env['HOROVOD_SPARK_PYTHONPATH'] = os.pathsep.join(test_horovod_python_path)

                with override_env(test_env):
                    with undo(reset):  # restores current working dir and sys.path after test
                        with mock.patch('horovod.spark.task.mpirun_exec_fn.task_exec') as task_exec:
                            msg = 'work_dir_env_set={} python_path_is_set={} hvd_python_path_is_set={}'\
                                .format(work_dir_env_set, python_path_is_set, hvd_python_path_is_set)
                            print('testing with {}'.format(msg))

                            # change cwd to tmp_path and test mpirun_exec_fn
                            os.chdir(tmp_path)
                            mpirun_exec_fn.main(driver, settings)

                            # work dir changed if HOROVOD_SPARK_WORK_DIR set
                            if work_dir_env_set:
                                self.assertEqual(test_dir, os.getcwd(), msg)
                            else:
                                self.assertEqual(tmp_path, os.getcwd(), msg)

                            # PYTHONPATH prepended with HOROVOD_SPARK_PYTHONPATH
                            expected_python_path = []
                            if hvd_python_path_is_set:
                                expected_python_path = test_horovod_python_path
                            if python_path_is_set:
                                expected_python_path = expected_python_path + test_python_path
                            if 'PYTHONPATH' in os.environ:
                                actual_python_path = os.environ['PYTHONPATH']
                            else:
                                actual_python_path = ""
                            self.assertEqual(os.pathsep.join(expected_python_path), actual_python_path, msg)

                            # HOROVOD_SPARK_PYTHONPATH injected at sys.path[1]
                            expected_sys_path = copy.copy(test_sys_path)
                            if hvd_python_path_is_set:
                                expected_sys_path = expected_sys_path[0:1] + \
                                                    test_horovod_python_path + \
                                                    expected_sys_path[1:]
                            self.assertEqual(expected_sys_path, sys.path, msg)

                            task_exec.assert_called_once()
                            task_exec_args, task_exec_kwargs = task_exec.call_args
                            expected_task_exec_args = (driver, settings, 'OMPI_COMM_WORLD_RANK')
                            expected_task_exec_kwargs = {}
                            self.assertEqual(expected_task_exec_args, task_exec_args, msg)
                            self.assertEqual(expected_task_exec_kwargs, task_exec_kwargs, msg)

    def test_gloo_exec_fn(self):
        driver = mock.MagicMock()
        settings = mock.MagicMock()
        settings.verbose = 2

        with mock.patch('horovod.spark.task.gloo_exec_fn.task_exec') as task_exec:
            gloo_exec_fn.main(driver, settings)

            task_exec.assert_called_once()
            task_exec_args, task_exec_kwargs = task_exec.call_args
            expected_task_exec_args = (driver, settings, 'HOROVOD_RANK')
            expected_task_exec_kwargs = {}
            self.assertEqual(expected_task_exec_args, task_exec_args)
            self.assertEqual(expected_task_exec_kwargs, task_exec_kwargs)

    def test_df_cache(self):
        # Clean the cache before starting the test
        util.clear_training_cache()
        util._training_cache.get_dataset = mock.Mock(side_effect=util._training_cache.get_dataset)

        with spark_session('test_df_cache') as spark:
            with local_store() as store:
                df = create_xor_data(spark)
                df2 = create_xor_data(spark)
                df3 = create_xor_data(spark)

                key = util._training_cache.create_key(df, store, None)
                key2 = util._training_cache.create_key(df2, store, None)
                key3 = util._training_cache.create_key(df3, store, None)

                # All keys are distinct
                assert key != key2
                assert key != key3
                assert key2 != key3

                # The cache should be empty to start
                assert not util._training_cache.is_cached(key, store)
                assert not util._training_cache.is_cached(key2, store)
                assert not util._training_cache.is_cached(key3, store)

                # First insertion into the cache
                with util.prepare_data(num_processes=2,
                                       store=store,
                                       df=df,
                                       feature_columns=['features'],
                                       label_columns=['y']) as dataset_idx:
                    train_rows, val_rows, metadata, avg_row_size = util.get_dataset_properties(dataset_idx)
                    util._training_cache.get_dataset.assert_not_called()
                    assert len(util._training_cache._key_to_dataset) == 1
                    assert util._training_cache.is_cached(key, store)
                    assert dataset_idx == 0

                    # The first dataset is still in use, so we assign the next integer in sequence to this
                    # dataset
                    assert not util._training_cache.is_cached(key2, store)
                    with util.prepare_data(num_processes=2,
                                           store=store,
                                           df=df2,
                                           feature_columns=['features'],
                                           label_columns=['y']) as dataset_idx2:
                        util._training_cache.get_dataset.assert_not_called()
                        assert len(util._training_cache._key_to_dataset) == 2
                        assert util._training_cache.is_cached(key2, store)
                        assert dataset_idx2 == 1

                # Even though the first dataset is no longer in use, it is still cached
                with util.prepare_data(num_processes=2,
                                       store=store,
                                       df=df,
                                       feature_columns=['features'],
                                       label_columns=['y']) as dataset_idx1:
                    train_rows1, val_rows1, metadata1, avg_row_size1 = util.get_dataset_properties(dataset_idx1)
                    util._training_cache.get_dataset.assert_called()
                    assert train_rows == train_rows1
                    assert val_rows == val_rows1
                    assert metadata == metadata1
                    assert avg_row_size == avg_row_size1
                    assert dataset_idx1 == 0

                # The first dataset is no longer in use, so we can reclaim its dataset index
                assert not util._training_cache.is_cached(key3, store)
                with util.prepare_data(num_processes=2,
                                       store=store,
                                       df=df3,
                                       feature_columns=['features'],
                                       label_columns=['y']) as dataset_idx3:
                    train_rows3, val_rows3, metadata3, avg_row_size3 = util.get_dataset_properties(dataset_idx3)
                    assert train_rows == train_rows3
                    assert val_rows == val_rows3
                    assert metadata == metadata3
                    assert avg_row_size == avg_row_size3
                    assert dataset_idx3 == 0

                # Same dataframe, different validation
                bad_key = util._training_cache.create_key(df, store, 0.1)
                assert not util._training_cache.is_cached(bad_key, store)

    def test_get_col_info(self):
        with spark_session('test_get_col_info') as spark:
            data = [[
                0,
                0.0,
                None,
                [1, 1],
                DenseVector([1.0, 1.0]),
                SparseVector(2, {1: 1.0}),
                DenseVector([1.0, 1.0])
            ], [
                1,
                None,
                None,
                [1, 1],
                DenseVector([1.0, 1.0]),
                SparseVector(2, {1: 1.0}),
                SparseVector(2, {1: 1.0})
            ]]

            schema = StructType([
                StructField('int', IntegerType()),
                StructField('float', FloatType()),
                StructField('null', NullType()),
                StructField('array', ArrayType(IntegerType())),
                StructField('dense', VectorUDT()),
                StructField('sparse', VectorUDT()),
                StructField('mixed', VectorUDT())
            ])

            df = create_test_data_from_schema(spark, data, schema)
            all_col_types, col_shapes, col_max_sizes = util._get_col_info(df)

            expected = [
                ('int', {int}, 1, 1),
                ('float', {float, NullType}, 1, 1),
                ('null', {NullType}, 1, 1),
                ('array', {list}, 2, 2),
                ('dense', {DenseVector}, 2, 2),
                ('sparse', {SparseVector}, 2, 1),
                ('mixed', {DenseVector, SparseVector}, 2, 2)
            ]

            for expected_col_info in expected:
                col_name, col_types, col_shape, col_size = expected_col_info
                assert all_col_types[col_name] == col_types, col_name
                assert col_shapes[col_name] == col_shape, col_name
                assert col_max_sizes[col_name] == col_size, col_name

    def test_get_col_info_error_bad_shape(self):
        with spark_session('test_get_col_info_error_bad_shape') as spark:
            data_bad_shape = [
                [SparseVector(2, {0: 1.0})],
                [SparseVector(1, {0: 1.0})]
            ]
            schema = StructType([StructField('data', VectorUDT())])
            df = create_test_data_from_schema(spark, data_bad_shape, schema)

            with pytest.raises(ValueError):
                util._get_col_info(df)

    def test_get_col_info_error_bad_size(self):
        with spark_session('test_get_col_info_error_bad_size') as spark:
            data_bad_size = [
                [DenseVector([1.0, 1.0])],
                [DenseVector([1.0])]
            ]
            schema = StructType([StructField('data', VectorUDT())])
            df = create_test_data_from_schema(spark, data_bad_size, schema)

            with pytest.raises(ValueError):
                util._get_col_info(df)

    def test_train_val_split_ratio(self):
        with spark_session('test_train_val_split_ratio') as spark:
            data = [
                [1.0], [1.0], [1.0], [1.0], [1.0]
            ]
            schema = StructType([StructField('data', FloatType())])
            df = create_test_data_from_schema(spark, data, schema)

            validation = 0.2
            train_df, val_df, validation_ratio = util._train_val_split(df, validation)

            # Only check validation ratio, as we can't rely on random splitting to produce an exact
            # result of 4 training and 1 validation samples.
            assert validation_ratio == validation

    def test_train_val_split_col_integer(self):
        with spark_session('test_train_val_split_col_integer') as spark:
            data = [
                [1.0, 0], [1.0, 0], [1.0, 0], [1.0, 0], [1.0, 1]
            ]
            schema = StructType([StructField('data', FloatType()), StructField('val', IntegerType())])
            df = create_test_data_from_schema(spark, data, schema)

            validation = 'val'
            train_df, val_df, validation_ratio = util._train_val_split(df, validation)

            # Only check counts as validation ratio cannot be guaranteed due to approx calculation
            assert train_df.count() == 4
            assert val_df.count() == 1

    def test_train_val_split_col_boolean(self):
        with spark_session('test_train_val_split_col_boolean') as spark:
            data = [
                [1.0, False], [1.0, False], [1.0, False], [1.0, False], [1.0, True]
            ]
            schema = StructType([StructField('data', FloatType()), StructField('val', BooleanType())])
            df = create_test_data_from_schema(spark, data, schema)

            validation = 'val'
            train_df, val_df, validation_ratio = util._train_val_split(df, validation)

            # Only check counts as validation ratio cannot be guaranteed due to approx calculation
            assert train_df.count() == 4
            assert val_df.count() == 1

    def test_get_metadata(self):
        expected_metadata = \
            {
                'float': {
                    'spark_data_type': FloatType,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': 1,
                    'shape': 1
                },
                'dense': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.ARRAY,
                    'max_size': 2,
                    'shape': 2
                },
                'sparse': {
                    'spark_data_type': SparseVector,
                    'is_sparse_vector_only': True,
                    'intermediate_format': constants.CUSTOM_SPARSE,
                    'max_size': 1,
                    'shape': 2
                },
                'mixed': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.ARRAY,
                    'max_size': 2,
                    'shape': 2
                },
            }

        with spark_session('test_get_metadata') as spark:
            data = [
                [1.0, DenseVector([1.0, 1.0]), SparseVector(2, {0: 1.0}), DenseVector([1.0, 1.0])],
                [1.0, DenseVector([1.0, 1.0]), SparseVector(2, {1: 1.0}), SparseVector(2, {1: 1.0})]
            ]
            schema = StructType([
                StructField('float', FloatType()),
                StructField('dense', VectorUDT()),
                StructField('sparse', VectorUDT()),
                StructField('mixed', VectorUDT())
            ])
            df = create_test_data_from_schema(spark, data, schema)

            metadata = util._get_metadata(df)
            self.assertDictEqual(metadata, expected_metadata)

    def test_prepare_data_no_compression(self):
        util.clear_training_cache()

        expected_metadata = \
            {
                'float': {
                    'spark_data_type': DoubleType,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': None,
                    'shape': None
                },
                'dense': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': None,
                    'shape': None
                },
                'sparse': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': None,
                    'shape': None
                },
                'mixed': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': None,
                    'shape': None
                },
            }

        with mock.patch('horovod.spark.common.util._get_metadata',
                        side_effect=util._get_metadata) as mock_get_metadata:
            with spark_session('test_prepare_data') as spark:
                data = [[
                    0.0,
                    DenseVector([1.0, 1.0]),
                    SparseVector(2, {1: 1.0}),
                    DenseVector([1.0, 1.0])
                ], [
                    1.0,
                    DenseVector([1.0, 1.0]),
                    SparseVector(2, {1: 1.0}),
                    SparseVector(2, {1: 1.0})
                ]]

                schema = StructType([
                    StructField('float', FloatType()),
                    StructField('dense', VectorUDT()),
                    StructField('sparse', VectorUDT()),
                    StructField('mixed', VectorUDT())
                ])

                df = create_test_data_from_schema(spark, data, schema)

                with local_store() as store:
                    with util.prepare_data(num_processes=2,
                                           store=store,
                                           df=df,
                                           feature_columns=['dense', 'sparse', 'mixed'],
                                           label_columns=['float']) as dataset_idx:
                        mock_get_metadata.assert_not_called()
                        assert dataset_idx == 0

                        train_rows, val_rows, metadata, avg_row_size = util.get_dataset_properties(dataset_idx)
                        self.assertDictEqual(metadata, expected_metadata)

    def test_prepare_data_compress_sparse(self):
        util.clear_training_cache()

        expected_metadata = \
            {
                'float': {
                    'spark_data_type': FloatType,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': 1,
                    'shape': 1
                },
                'dense': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.ARRAY,
                    'max_size': 2,
                    'shape': 2
                },
                'sparse': {
                    'spark_data_type': SparseVector,
                    'is_sparse_vector_only': True,
                    'intermediate_format': constants.CUSTOM_SPARSE,
                    'max_size': 1,
                    'shape': 2
                },
                'mixed': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.ARRAY,
                    'max_size': 2,
                    'shape': 2
                },
            }

        with mock.patch('horovod.spark.common.util._get_metadata',
                        side_effect=util._get_metadata) as mock_get_metadata:
            with spark_session('test_prepare_data') as spark:
                data = [[
                    0.0,
                    DenseVector([1.0, 1.0]),
                    SparseVector(2, {1: 1.0}),
                    DenseVector([1.0, 1.0])
                ], [
                    1.0,
                    DenseVector([1.0, 1.0]),
                    SparseVector(2, {1: 1.0}),
                    SparseVector(2, {1: 1.0})
                ]]

                schema = StructType([
                    StructField('float', FloatType()),
                    StructField('dense', VectorUDT()),
                    StructField('sparse', VectorUDT()),
                    StructField('mixed', VectorUDT())
                ])

                df = create_test_data_from_schema(spark, data, schema)

                with local_store() as store:
                    with util.prepare_data(num_processes=2,
                                           store=store,
                                           df=df,
                                           feature_columns=['dense', 'sparse', 'mixed'],
                                           label_columns=['float'],
                                           compress_sparse=True) as dataset_idx:
                        mock_get_metadata.assert_called()
                        assert dataset_idx == 0

                        train_rows, val_rows, metadata, avg_row_size = util.get_dataset_properties(dataset_idx)
                        self.assertDictEqual(metadata, expected_metadata)

    def test_check_shape_compatibility(self):
        feature_columns = ['x1', 'x2', 'features']
        label_columns = ['y1', 'y_embedding']

        schema = StructType([StructField('x1', DoubleType()),
                             StructField('x2', IntegerType()),
                             StructField('features', VectorUDT()),
                             StructField('y1', FloatType()),
                             StructField('y_embedding', VectorUDT())])
        data = [[1.0, 1, DenseVector([1.0] * 12), 1.0, DenseVector([1.0] * 12)]] * 10

        with spark_session('test_df_cache') as spark:
                df = create_test_data_from_schema(spark, data, schema)
                metadata = util._get_metadata(df)

                input_shapes = [[1], [1], [-1, 3, 4]]
                output_shapes = [[1], [-1, 3, 4]]
                util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                               input_shapes, output_shapes)

                input_shapes = [[1], [1], [3, 2, 2]]
                output_shapes = [[1, 1], [-1, 2, 3, 2]]
                util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                               input_shapes, output_shapes)

                bad_input_shapes = [[1], [1], [-1, 3, 5]]
                with pytest.raises(ValueError):
                    util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                                   bad_input_shapes, output_shapes)

                bad_input_shapes = [[2], [1], [-1, 3, 4]]
                with pytest.raises(ValueError):
                    util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                                   bad_input_shapes, output_shapes)

                bad_output_shapes = [[7], [-1, 3, 4]]
                with pytest.raises(ValueError):
                    util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                                   input_shapes, bad_output_shapes)

    @mock.patch('horovod.spark.common.store.HDFSStore._get_filesystem_fn')
    def test_sync_hdfs_store(self, mock_get_fs_fn):
        mock_fs = mock.Mock()
        mock_get_fs_fn.return_value = lambda: mock_fs

        hdfs_root = '/user/test/output'
        store = HDFSStore(hdfs_root)

        run_id = 'run_001'
        get_local_output_dir = store.get_local_output_dir_fn(run_id)
        sync_to_store = store.sync_fn(run_id)
        run_root = store.get_run_path(run_id)

        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        with get_local_output_dir() as local_dir:
            touch(os.path.join(local_dir, 'a.txt'), (1330712280, 1330712280))
            sync_to_store(local_dir)
            mock_fs.upload.assert_called_with(os.path.join(run_root, 'a.txt'), mock.ANY)

            touch(os.path.join(local_dir, 'b.txt'), (1330712280, 1330712280))
            sync_to_store(local_dir)
            mock_fs.upload.assert_called_with(os.path.join(run_root, 'b.txt'), mock.ANY)

            subdir = os.path.join(local_dir, 'subdir')
            os.mkdir(subdir)
            touch(os.path.join(subdir, 'c.txt'), (1330712280, 1330712280))
            sync_to_store(local_dir)
            mock_fs.upload.assert_called_with(os.path.join(run_root, 'subdir/c.txt'), mock.ANY)

            touch(os.path.join(local_dir, 'a.txt'), (1330712292, 1330712292))
            touch(os.path.join(local_dir, 'b.txt'), (1330712292, 1330712292))
            assert mock_fs.upload.call_count == 3

            sync_to_store(local_dir)
            assert mock_fs.upload.call_count == 5

    @mock.patch('horovod.spark.common.store.HDFSStore._get_filesystem_fn')
    def test_hdfs_store_parse_url(self, mock_get_filesystem_fn):
        # Case 1: full path
        hdfs_root = 'hdfs://namenode01:8020/user/test/output'
        store = HDFSStore(hdfs_root)
        assert store.path_prefix() == 'hdfs://namenode01:8020', hdfs_root
        assert store.get_full_path('/user/test/output') == 'hdfs://namenode01:8020/user/test/output', hdfs_root
        assert store.get_localized_path('hdfs://namenode01:8020/user/test/output') == '/user/test/output', hdfs_root
        assert store._hdfs_kwargs['host'] == 'namenode01', hdfs_root
        assert store._hdfs_kwargs['port'] == 8020, hdfs_root

        # Case 2: no host and port
        hdfs_root = 'hdfs:///user/test/output'
        store = HDFSStore(hdfs_root)
        assert store.path_prefix() == 'hdfs://', hdfs_root
        assert store.get_full_path('/user/test/output') == 'hdfs:///user/test/output', hdfs_root
        assert store.get_localized_path('hdfs:///user/test/output') == '/user/test/output', hdfs_root
        assert store._hdfs_kwargs['host'] == 'default', hdfs_root
        assert store._hdfs_kwargs['port'] == 0, hdfs_root

        # Case 3: no prefix
        hdfs_root = '/user/test/output'
        store = HDFSStore(hdfs_root)
        assert store.path_prefix() == 'hdfs://', hdfs_root
        assert store.get_full_path('/user/test/output') == 'hdfs:///user/test/output', hdfs_root
        assert store.get_localized_path('hdfs:///user/test/output') == '/user/test/output', hdfs_root
        assert store._hdfs_kwargs['host'] == 'default', hdfs_root
        assert store._hdfs_kwargs['port'] == 0, hdfs_root

        # Case 4: no namespace
        hdfs_root = 'hdfs://namenode01:8020/user/test/output'
        store = HDFSStore(hdfs_root)
        assert store.path_prefix() == 'hdfs://namenode01:8020', hdfs_root
        assert store.get_full_path('/user/test/output') == 'hdfs://namenode01:8020/user/test/output', hdfs_root
        assert store.get_localized_path('hdfs://namenode01:8020/user/test/output') == '/user/test/output', hdfs_root
        assert store._hdfs_kwargs['host'] == 'namenode01', hdfs_root
        assert store._hdfs_kwargs['port'] == 8020, hdfs_root

        # Case 5: bad prefix
        with pytest.raises(ValueError):
            hdfs_root = 'file:///user/test/output'
            HDFSStore(hdfs_root)

        # Case 6: override paths, no prefix
        hdfs_root = '/user/prefix'
        store = HDFSStore(hdfs_root,
                          train_path='/user/train_path',
                          val_path='/user/val_path',
                          test_path='/user/test_path')
        assert store.get_train_data_path() == 'hdfs:///user/train_path', hdfs_root
        assert store.get_val_data_path() == 'hdfs:///user/val_path', hdfs_root
        assert store.get_test_data_path() == 'hdfs:///user/test_path', hdfs_root

        # Case 7: override paths, prefix
        hdfs_root = 'hdfs:///user/prefix'
        store = HDFSStore(hdfs_root,
                          train_path='hdfs:///user/train_path',
                          val_path='hdfs:///user/val_path',
                          test_path='hdfs:///user/test_path')
        assert store.get_train_data_path() == 'hdfs:///user/train_path', hdfs_root
        assert store.get_val_data_path() == 'hdfs:///user/val_path', hdfs_root
        assert store.get_test_data_path() == 'hdfs:///user/test_path', hdfs_root

    def test_spark_task_service_env(self):
        key = secret.make_secret_key()
        service_env = {
            'HADOOP_TOKEN_FILE_LOCATION': 'path',
            'PYTHONPATH': 'pypath',
            'other': 'values'
        }
        with override_env(service_env):
            service = SparkTaskService(1, key, None)
            client = SparkTaskClient(1, service.addresses(), key, 3)

            with tempdir() as d:
                file = '{}/env'.format(d)
                command = "env | grep -v '^PWD='> {}".format(file)
                command_env = {"test": "value"}

                try:
                    client.run_command(command, command_env)
                    client.wait_for_command_termination()
                finally:
                    service.shutdown()

                with open(file) as f:
                    env = sorted([line.strip() for line in f.readlines()])
                    expected = [
                        'HADOOP_TOKEN_FILE_LOCATION=path',
                        'HOROVOD_SPARK_WORK_DIR={cwd}'.format(cwd=os.getcwd()),
                        'PYTHONPATH=pypath',
                        '{}={}'.format(secret.HOROVOD_SECRET_KEY, codec.dumps_base64(key)),
                        'other=values',
                        'test=value'
                    ]
                    self.assertEqual(expected, env)

    @pytest.mark.skipif(LooseVersion(pyspark.__version__) < LooseVersion('3.0.0'),
                        reason='get_available_devices only supported in Spark 3.0 and above')
    def test_get_available_devices(self):
        def fn():
            hvd.init()
            devices = get_available_devices()
            return devices, hvd.local_rank()

        with spark_session('test_get_available_devices', gpus=2):
            res = horovod.spark.run(fn, env={'PATH': os.environ.get('PATH')}, verbose=0)
            self.assertListEqual([(['0'], 0), (['1'], 1)], res)

    def test_to_list(self):
        none_output = util.to_list(None, 1)
        assert none_output is none_output

        out1 = util.to_list('one_item', 1)
        assert out1 == ['one_item']

        out2 = util.to_list('one_item', 2)
        assert out2 == ['one_item', 'one_item']

        out3 = util.to_list(['one_item'], 1)
        assert out3 == ['one_item']

        out4 = util.to_list(['item1', 'item2'], 2)
        assert out4 == ['item1', 'item2']

        with pytest.raises(ValueError):
            util.to_list(['item1', 'item2'], 4)
