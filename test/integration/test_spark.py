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

import contextlib
import copy
import io
import itertools
import logging
import os
import platform
import re
import sys
import threading
import time
import unittest
import warnings
from distutils.version import LooseVersion

import mock
import psutil
import pyspark
import pytest
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, FloatType, IntegerType, \
    NullType, StructField, StructType

import horovod.spark
from horovod.common.util import gloo_built, mpi_built
from horovod.runner.common.util import codec, secret, safe_shell_exec, timeout
from horovod.runner.common.util import settings as hvd_settings
from horovod.runner.mpi_run import is_open_mpi
from horovod.runner.util.threads import in_thread
from horovod.spark.common import constants, util
from horovod.spark.common.store import DBFSLocalStore, HDFSStore, LocalStore, Store
from horovod.spark.driver.host_discovery import SparkDriverHostDiscovery
from horovod.spark.driver.rsh import rsh
from horovod.spark.runner import _task_fn
from horovod.spark.task import get_available_devices
from horovod.spark.task import gloo_exec_fn, mpirun_exec_fn
from horovod.spark.task.task_service import SparkTaskClient

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from spark_common import spark_driver_service, spark_session, spark_task_service, \
    create_test_data_from_schema, create_xor_data, local_store

from common import is_built, mpi_implementation_flags, tempdir, override_env, undo, delay, spawn


# Spark will fail to initialize correctly locally on Mac OS without this
if platform.system() == 'Darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'


def fn(result=0):
    return result


@spawn
def run_get_available_devices():
    # Run this test in an isolated "spawned" environment because creating a local-cluster
    # leads to errors that cause downstream tests to stall:
    # https://issues.apache.org/jira/browse/SPARK-31922
    def fn():
        import horovod.torch as hvd

        hvd.init()
        devices = get_available_devices()
        return devices, hvd.local_rank()

    with spark_session('test_get_available_devices', gpus=2):
        return horovod.spark.run(fn, env={'PATH': os.environ.get('PATH')}, verbose=0)


class SparkTests(unittest.TestCase):
    """
    Tests for horovod.spark.run().
    """

    def __init__(self, *args, **kwargs):
        super(SparkTests, self).__init__(*args, **kwargs)
        self.maxDiff = None
        logging.getLogger('py4j.java_gateway').setLevel(logging.INFO)
        warnings.simplefilter('module')

    def run(self, result=None):
        # These unit tests should not be run with horovodrun as some tests
        # setup their own Horovod cluster, where both will then interfere.
        if 'OMPI_COMM_WORLD_RANK' in os.environ or 'HOROVOD_RANK' in os.environ:
            self.skipTest("These tests should not be executed via horovodrun, just pytest")

        super(SparkTests, self).run(result)

    def test_host_hash(self):
        hash = util.host_hash()

        # host hash should be consistent
        self.assertEqual(util.host_hash(), hash)

        # host_hash should consider CONTAINER_ID environment variable
        with override_env({'CONTAINER_ID': 'a container id'}):
            containered_hash = util.host_hash()
            self.assertNotEqual(containered_hash, hash)

            # given an extra salt, host hash must differ
            salted_containered_hash = util.host_hash('salt')
            self.assertNotEqual(salted_containered_hash, hash)
            self.assertNotEqual(salted_containered_hash, containered_hash)

            # host hash should be consistent
            self.assertEqual(util.host_hash(), containered_hash)
            self.assertEqual(util.host_hash('salt'), salted_containered_hash)

        # host hash should still be consistent
        self.assertEqual(util.host_hash(), hash)

        # given an extra salt, host hash must differ
        salted_hash = util.host_hash('salt')
        self.assertNotEqual(salted_hash, hash)
        self.assertNotEqual(salted_hash, salted_containered_hash)

    def test_driver_common_interfaces(self):
        with spark_driver_service(num_proc=2) as (driver, client, _):
            client.register_task_to_task_addresses(0, {'lo': [('127.0.0.1', 31321)], 'eth0': [('192.168.0.1', 31321)]})
            client.register_task_to_task_addresses(1, {'eth1': [('10.0.0.1', 31322)], 'eth0': [('192.168.0.2', 31322)]})

            nics = driver.get_common_interfaces()
            self.assertEqual({'eth0'}, nics)

    def test_driver_common_interfaces_from_settings(self):
        nics = list(psutil.net_if_addrs().keys())
        if not nics:
            self.skipTest('this machine has no network interfaces')

        nic = nics[0]
        with spark_driver_service(num_proc=2, nics={nic}) as (driver, client, _):
            client.register_task_to_task_addresses(0, {'eth0': [('192.168.0.1', 31321)]})
            client.register_task_to_task_addresses(1, {'eth1': [('10.0.0.1', 31322)]})

            nics = driver.get_common_interfaces()
            self.assertEqual({nic}, nics)

    def test_driver_common_interfaces_fails(self):
        with spark_driver_service(num_proc=2) as (driver, client, _):
            client.register_task_to_task_addresses(0, {'eth0': [('192.168.0.1', 31321)]})
            client.register_task_to_task_addresses(1, {'eth1': [('10.0.0.1', 31322)]})

            with pytest.raises(Exception, match=r"^Unable to find a set of common task-to-task "
                                                r"communication interfaces: \["
                                                r"\(0, \{'eth0': \[\('192.168.0.1', 31321\)\]\}\), "
                                                r"\(1, \{'eth1': \[\('10.0.0.1', 31322\)\]\}\)"
                                                r"\]$"):
                driver.get_common_interfaces()

    def test_driver_set_local_rank_to_index(self):
        with spark_driver_service(num_proc=3) as (driver, client, _):
            self.assertEqual({}, driver.get_ranks_to_indices())

            client.register_task(0, {'lo': [('127.0.0.1', 31320)]}, 'host-1')
            client.register_task(2, {'lo': [('127.0.0.1', 31322)]}, 'host-1')
            client.register_task(1, {'lo': [('127.0.0.1', 31321)]}, 'host-2')

            # host-1, local-rank 1: rank 0 -> index 2
            index = client.set_local_rank_to_rank('host-1', 1, 0)
            self.assertEqual({0: 2}, driver.get_ranks_to_indices())
            self.assertEqual(2, index)

            # host-2, local-rank 0: rank 1 -> index 1
            index = client.set_local_rank_to_rank('host-2', 0, 1)
            self.assertEqual({0: 2, 1: 1}, driver.get_ranks_to_indices())
            self.assertEqual(1, index)

            # host-1, local-rank 0: rank 2 -> index 0
            index = client.set_local_rank_to_rank('host-1', 0, 2)
            self.assertEqual({0: 2, 1: 1, 2: 0}, driver.get_ranks_to_indices())
            self.assertEqual(0, index)

    def test_task_service_wait_for_command_start_without_timeout(self):
        with spark_task_service(0) as (task, client, _):
            start = time.time()
            delay(lambda: client.run_command('true', {}), 1.0)
            task.wait_for_command_start(None)
            duration = time.time() - start
            self.assertGreaterEqual(duration, 1.0)

    def test_task_service_wait_for_command_start_with_timeout(self):
        with spark_task_service(0) as (task, client, _):
            tmout = timeout.Timeout(1.0, 'Timed out waiting for {activity}.')
            start = time.time()
            d = delay(lambda: client.run_command('true', {}), 0.5)
            task.wait_for_command_start(tmout)
            duration = time.time() - start
            self.assertGreaterEqual(duration, 0.5)
            self.assertLess(duration, 0.75)
            d.join()

        with spark_task_service(0) as (task, client, _):
            tmout = timeout.Timeout(1.0, 'Timed out waiting for {activity}.')
            start = time.time()
            d = delay(lambda: client.run_command('true', {}), 1.5)
            with pytest.raises(Exception, match='^Timed out waiting for command to run. Timeout after 1.0 seconds.$'):
                task.wait_for_command_start(tmout)
            duration = time.time() - start
            self.assertGreaterEqual(duration, 1.0)
            self.assertLess(duration, 1.25)
            d.join()

    def test_task_service_check_for_command_start(self):
        for tmout in [1.0, 1]:
            with spark_task_service(0) as (task, client, _):
                start = time.time()
                delay(lambda: client.run_command('true', {}), 0.5)
                res = task.check_for_command_start(tmout)
                duration = time.time() - start
                self.assertGreaterEqual(duration, 0.5)
                self.assertLess(duration, 0.75)
                self.assertTrue(res)

            with spark_task_service(0) as (task, client, _):
                start = time.time()
                d = delay(lambda: client.run_command('true', {}), 1.5)
                res = task.check_for_command_start(tmout)
                duration = time.time() - start
                self.assertGreaterEqual(duration, 1.0)
                self.assertLess(duration, 1.25)
                self.assertFalse(res)
                d.join()

    @contextlib.contextmanager
    def spark_tasks(self, tasks, start_timeout, results):
        with spark_driver_service(num_proc=tasks, fn=fn, args=(123,)) \
                as (driver, driver_client, key):

            use_gloo = True
            is_elastic = True
            driver_addresses = driver.addresses()
            tmout = timeout.Timeout(start_timeout, message='Timed out waiting for {activity}.')
            settings = hvd_settings.BaseSettings(num_proc=tasks, verbose=2, key=key, start_timeout=tmout)

            def run_task(index):
                result = _task_fn(index, driver_addresses, key, settings, use_gloo, is_elastic)
                results[index] = result

            # start tasks
            threads = list([in_thread(run_task, args=(index,)) for index in range(tasks)])
            driver.wait_for_initial_registration(tmout)
            task_clients = list([SparkTaskClient(index, driver.task_addresses_for_driver(index), key, 2)
                                 for index in range(tasks)])

            yield (driver, driver_client, task_clients, settings)

            # wait a bit and expect all threads to still run
            self.assertGreater(horovod.spark.runner.MINIMUM_COMMAND_LIFETIME_S, 2.0)
            time.sleep(1.0)
            for index in range(tasks):
                self.assertTrue(threads[index].is_alive(),
                                'task thread {} should still be alive'.format(index))

            # tasks should terminate within MINIMUM_COMMAND_LIFETIME_S
            time.sleep(horovod.spark.runner.MINIMUM_COMMAND_LIFETIME_S)
            for index in range(tasks):
                self.assertFalse(threads[index].is_alive(),
                                 'task thread {} should have terminated by now'.format(index))

    def test_task_fn_run_commands(self):
        if not gloo_built():
            self.skipTest("Gloo is not available")

        tasks = 3
        start_timeout = 5.0
        self.assertGreater(tasks, 1, 'test should not be trivial')
        results = {}

        with self.spark_tasks(tasks, start_timeout, results) \
                as (driver, driver_client, task_clients, settings):
            with tempdir() as d:
                # command template
                file_tmpl = os.path.sep.join([d, 'task_{}_executed_command_{}'])
                cmd_tmpl = 'touch {}'.format(file_tmpl)

                # before we execute the first command
                for index in range(tasks):
                    terminated, res = task_clients[index].command_result()
                    self.assertEqual(False, terminated)
                    self.assertEqual(None, res)

                # all tasks execute the command
                for index in range(tasks):
                    self.assertFalse(os.path.exists(file_tmpl.format(index, '1')))
                    task_clients[index].run_command(cmd_tmpl.format(index, '1'), {})
                    task_clients[index].wait_for_command_termination(delay=0.1)

                    terminated, res = task_clients[index].command_result()
                    self.assertEqual(True, terminated)
                    self.assertEqual(0, res)
                    self.assertTrue(os.path.exists(file_tmpl.format(index, '1')))

    def test_task_fn_run_gloo_exec(self):
        if not gloo_built():
            self.skipTest("Gloo is not available")

        tasks = 3
        start_timeout = 5.0
        self.assertGreater(tasks, 1, 'test should not be trivial')
        results = {}

        with self.spark_tasks(tasks, start_timeout, results) \
                as (driver, driver_client, task_clients, settings):
            # all tasks execute gloo_exec_fn to get a result back into the task service
            cmd = 'python -m horovod.spark.task.gloo_exec_fn {} {}'.format(
                codec.dumps_base64(driver.addresses()),
                codec.dumps_base64(settings)
            )
            for index in range(tasks):
                env = {'HOROVOD_RANK': str(index),
                       'HOROVOD_LOCAL_RANK': '0',
                       'HOROVOD_HOSTNAME': driver.task_index_host_hash(index),
                       secret.HOROVOD_SECRET_KEY: codec.dumps_base64(settings.key)}
                in_thread(lambda: task_clients[index].run_command(cmd, env))

            for index in range(tasks):
                task_clients[index].wait_for_command_termination(delay=0.1)
                terminated, res = task_clients[index].command_result()
                self.assertEqual(True, terminated)
                self.assertEqual(0, res)

        self.assertEqual(dict([(index, 123) for index in range(tasks)]), results)

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
    def test_happy_run_with_gloo(self):
        if not gloo_built():
            self.skipTest("Gloo is not available")

        self.do_test_happy_run(use_mpi=False, use_gloo=True)

    """
    Actually tests that horovod.spark.run works properly in a simple setup.
    """
    def do_test_happy_run(self, use_mpi, use_gloo):
        def fn():
            import horovod.torch as hvd
            import torch

            hvd.init()
            print(f'running fn {hvd.size()}')
            print(f'error line', file=sys.stderr)
            res = hvd.allgather(torch.tensor([hvd.rank()])).tolist()
            return res, hvd.rank()

        stdout = io.StringIO()
        stderr = io.StringIO()
        with spark_session('test_happy_run'):
            with is_built(gloo_is_built=use_gloo, mpi_is_built=use_mpi):
                res = horovod.spark.run(fn, start_timeout=10,
                                        use_mpi=use_mpi, use_gloo=use_gloo,
                                        env={'HOROVOD_LOG_LEVEL': 'WARNING'},
                                        stdout=stdout if use_gloo else None,
                                        stderr=stderr if use_gloo else None,
                                        verbose=2)
                self.assertListEqual([([0, 1], 0), ([0, 1], 1)], res)

                if use_gloo:
                    self.assertRegex(stdout.getvalue(),
                                     r'\[[01]\]<stdout>:running fn 2\n'
                                     r'\[[01]\]<stdout>:running fn 2\n')
                    self.assertRegex(stderr.getvalue(),
                                     r'\[[01]\]<stderr>:error line\n'
                                     r'\[[01]\]<stderr>:error line\n')

    """
    Test that horovod.spark.run_elastic works properly in a simple setup.
    """
    def test_happy_run_elastic(self):
        if not gloo_built():
            self.skipTest("Gloo is not available")

        def fn():
            # training function does not use ObjectState and @hvd.elastic.run
            # only testing distribution of state-less training function here
            # see test_spark_torch.py for testing that
            import horovod.torch as hvd
            import torch

            hvd.init()
            print(f'running fn {hvd.size()}')
            print(f'error line', file=sys.stderr)
            res = hvd.allgather(torch.tensor([hvd.rank()])).tolist()
            return res, hvd.rank()

        stdout = io.StringIO()
        stderr = io.StringIO()
        with spark_session('test_happy_run_elastic'):
            res = horovod.spark.run_elastic(fn, num_proc=2, min_np=2, max_np=2,
                                            env={'HOROVOD_LOG_LEVEL': 'WARNING'},
                                            stdout=stdout, stderr=stderr,
                                            start_timeout=10, verbose=2)
            self.assertListEqual([([0, 1], 0), ([0, 1], 1)], res)
            self.assertRegex(stdout.getvalue(),
                             r'\[[01]\]<stdout>:running fn 2\n'
                             r'\[[01]\]<stdout>:running fn 2\n')
            self.assertRegex(stderr.getvalue(),
                             r'\[[01]\]<stderr>:error line\n'
                             r'\[[01]\]<stderr>:error line\n')

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
    Test that horovod.spark.run fails with os.environ as env with mpi.

    TODO: figure out why this test sometimes causes hangs in CI
          https://github.com/horovod/horovod/issues/2217
    """
    @pytest.mark.skip
    def test_spark_run_with_os_environ_with_mpi(self):
        with is_built(gloo_is_built=False, mpi_is_built=True):
            with spark_session('test_spark_run', cores=2):
                with pytest.raises(Exception, match="^env argument must be a dict, not <class 'os._Environ'>: "):
                    horovod.spark.run(fn, num_proc=2, use_mpi=True, use_gloo=False,
                                      env=os.environ, verbose=2)

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
    def test_spark_run_with_non_zero_exit_with_gloo(self):
        expected = '^Horovod detected that one or more processes exited with non-zero ' \
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

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"], None

        def gloo_exec_command_fn(driver, key, settings, env, stdout, stderr, prefix_output_with_timestamp):
            def _exec_command(command, alloc_info, event):
                return 1, alloc_info.rank
            return _exec_command

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", return_value=1):
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

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"], None

        def exception(*args, **argv):
            raise Exception('Test Exception')

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", side_effect=exception) as execute:
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
                mpi_flags, binding_args, _ = horovod.runner.mpi_run._get_mpi_implementation_flags(False)
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

        # we are not asserting on the actual PYTHONPATH in actual_env, this is done in test_run.py
        self.assertEqual(expected_command, actual_command)
        if 'PYTHONPATH' in actual_env:
            actual_env.pop('PYTHONPATH')
        # we compare this secret below, not by comparing actual_env with env
        actual_secret = actual_env.pop(secret.HOROVOD_SECRET_KEY, None)

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
        self.assertEqual('Horovod detected that one or more processes exited with non-zero status, '
                         'thus causing the job to be terminated. The first process to do so was:\n'
                         'Process name: 0\n'
                         'Exit code: 1\n', str(e.value))

        num_proc = cores if num_proc is None else num_proc
        self.assertEqual(expected_np, num_proc)
        self.assertEqual(1, gloo_exec_command_fn.call_count)
        _, _, _, call_env, _, _, _ = gloo_exec_command_fn.call_args[0]
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
            expected_command = ('HOROVOD_HOSTNAME=[^ ]+ '
                                'HOROVOD_RANK={rank} '
                                'HOROVOD_SIZE={size} '
                                'HOROVOD_LOCAL_RANK={local_rank} '
                                'HOROVOD_LOCAL_SIZE={local_size} '
                                'HOROVOD_CROSS_RANK=0 '
                                'HOROVOD_CROSS_SIZE=1 '
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
                                'HOROVOD_HOSTNAME=[^ ]+',
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

    @pytest.mark.skip(reason='https://github.com/horovod/horovod/issues/1993')
    def test_rsh_event(self):
        self.do_test_rsh_events(1)

    @pytest.mark.skip(reason='https://github.com/horovod/horovod/issues/1993')
    def test_rsh_events(self):
        self.do_test_rsh_events(3)

    def do_test_rsh_events(self, test_events):
        self.assertGreater(test_events, 0, 'test should not be trivial')

        event_delay = 1.0
        wait_for_exit_code_delay = 1.0
        sleep = event_delay + safe_shell_exec.GRACEFUL_TERMINATION_TIME_S + \
            wait_for_exit_code_delay + 2.0
        command = 'sleep {}'.format(sleep)
        for triggered_event in range(test_events):
            events = [threading.Event() for _ in range(test_events)]
            delay(lambda: events[triggered_event].set(), 1.0)

            start = time.time()
            self.do_test_rsh(command, 143, events=events)
            duration = time.time() - start

            self.assertGreaterEqual(duration, event_delay)
            self.assertLess(duration, sleep - 1.0, 'sleep should not finish')

    def do_test_rsh(self, command, expected_result, events=None):
        # setup infrastructure so we can call rsh
        host_hash = 'test-host'
        with spark_driver_service(num_proc=1) as (driver, client, key):
            with spark_task_service(index=0, key=key) as (task, _, _):
                client.register_task(0, task.addresses(), host_hash)
                settings = hvd_settings.Settings(verbose=2, key=key)
                env = {}

                res = rsh(driver.addresses(), key, host_hash, command, env, 0, settings.verbose,
                          stdout=None, stderr=None, prefix_output_with_timestamp=False,
                          background=False, events=events)
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
                            expected_task_exec_args = (driver, settings, 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK')
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
            expected_task_exec_args = (driver, settings, 'HOROVOD_RANK', 'HOROVOD_LOCAL_RANK')
            expected_task_exec_kwargs = {}
            self.assertEqual(expected_task_exec_args, task_exec_args)
            self.assertEqual(expected_task_exec_kwargs, task_exec_kwargs)

    def test_mpi_exec_fn_provides_driver_with_local_rank(self):
        self.do_test_exec_fn_provides_driver_with_local_rank(
            mpirun_exec_fn, 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK'
        )

    def test_gloo_exec_fn_provides_driver_with_local_rank(self):
        self.do_test_exec_fn_provides_driver_with_local_rank(
            gloo_exec_fn, 'HOROVOD_RANK', 'HOROVOD_LOCAL_RANK'
        )

    def do_test_exec_fn_provides_driver_with_local_rank(self, exec_fn, rank_env, local_rank_env):
        with mock.patch("horovod.spark.task.task_service.SparkTaskService._get_resources", return_value={}):
            with spark_driver_service(num_proc=3) as (driver, client, key), \
                    spark_task_service(index=0, key=key) as (task0, _, _), \
                    spark_task_service(index=1, key=key) as (task1, _, _), \
                    spark_task_service(index=2, key=key) as (task2, _, _):
                self.assertIsNone(task0.fn_result())
                self.assertIsNone(task1.fn_result())
                self.assertIsNone(task2.fn_result())

                client.register_task(0, task0.addresses(), 'host-1')
                client.register_task(1, task1.addresses(), 'host-2')
                client.register_task(2, task2.addresses(), 'host-1')
                self.assertEqual({}, driver.get_ranks_to_indices())

                settings = mock.MagicMock(verbose=2)

                with override_env({rank_env: 0,
                                   local_rank_env: 1,
                                   'HOROVOD_HOSTNAME': 'host-1',
                                   secret.HOROVOD_SECRET_KEY: codec.dumps_base64(key)}):
                    exec_fn.main(driver.addresses(), settings)
                    self.assertEqual(None, task0.fn_result())
                    self.assertEqual(None, task1.fn_result())
                    self.assertEqual(0, task2.fn_result())
                    self.assertEqual({0: 2}, driver.get_ranks_to_indices())

                with override_env({rank_env: 2,
                                   local_rank_env: 0,
                                   'HOROVOD_HOSTNAME': 'host-2',
                                   secret.HOROVOD_SECRET_KEY: codec.dumps_base64(key)}):
                    exec_fn.main(driver.addresses(), settings)
                    self.assertEqual(None, task0.fn_result())
                    self.assertEqual(0, task1.fn_result())
                    self.assertEqual(0, task2.fn_result())
                    self.assertEqual({0: 2, 2: 1}, driver.get_ranks_to_indices())

                with override_env({rank_env: 1,
                                   local_rank_env: 0,
                                   'HOROVOD_HOSTNAME': 'host-1',
                                   secret.HOROVOD_SECRET_KEY: codec.dumps_base64(key)}):
                    exec_fn.main(driver.addresses(), settings)
                    self.assertEqual(0, task0.fn_result())
                    self.assertEqual(0, task1.fn_result())
                    self.assertEqual(0, task2.fn_result())
                    self.assertEqual({0: 2, 2: 1, 1: 0}, driver.get_ranks_to_indices())

    def test_spark_driver_host_discovery(self):
        with spark_driver_service(num_proc=4) as (driver, client, _):
            discovery = SparkDriverHostDiscovery(driver)

            slots = discovery.find_available_hosts_and_slots()
            self.assertEqual({}, slots)

            client.register_task(0, driver.addresses(), 'host-hash-1')
            slots = discovery.find_available_hosts_and_slots()
            self.assertEqual({'host-hash-1': 1}, slots)

            client.register_task(1, driver.addresses(), 'host-hash-2')
            slots = discovery.find_available_hosts_and_slots()
            self.assertEqual({'host-hash-1': 1, 'host-hash-2': 1}, slots)

            client.register_task(2, driver.addresses(), 'host-hash-2')
            slots = discovery.find_available_hosts_and_slots()
            self.assertEqual({'host-hash-1': 1, 'host-hash-2': 2}, slots)

            client.register_task(3, driver.addresses(), 'host-hash-1')
            slots = discovery.find_available_hosts_and_slots()
            self.assertEqual({'host-hash-1': 2, 'host-hash-2': 2}, slots)

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
                ('float', {float, type(None)}, 1, 1),
                ('null', {type(None)}, 1, 1),
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

                input_shapes = [[1], [1], [-1, 3, 4]]
                label_shapes = [[1], [-1, 3, 4]]
                util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                               input_shapes, output_shapes, label_shapes)

                # The case where label_shapes is different from output_shapes
                input_shapes = [[1], [1], [-1, 3, 4]]
                output_shapes = [[1, 1], [-1, 2, 3, 2]]
                label_shapes = [[1], [-1, 3, 4]]
                util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                               input_shapes, output_shapes, label_shapes)

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

                input_shapes = [[1], [1], [-1, 3, 4]]
                bad_label_shapes = [[-1, 3, 4]]
                with pytest.raises(ValueError):
                    util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                                   input_shapes, output_shapes, bad_label_shapes)

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
        service_env = {
            'HADOOP_TOKEN_FILE_LOCATION': 'path',
            'PYTHONPATH': 'pypath',
            'other': 'values'
        }
        with override_env(service_env):
            with spark_task_service(index=1) as (service, client, key):
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

    def do_test_spark_task_service_executes_command(self, client, file):
        self.assertFalse(os.path.exists(file))
        client.run_command('touch {}'.format(file), {})
        client.wait_for_command_termination(delay=0.1)
        terminated, exit_code = client.command_result()
        self.assertEqual(True, terminated)
        self.assertEqual(0, exit_code)
        self.assertTrue(os.path.exists(file))

    def test_spark_task_service_execute_command(self):
        with spark_task_service(index=0) as (service, client, _):
            with tempdir() as d:
                file = os.path.sep.join([d, 'command_executed'])
                self.do_test_spark_task_service_executes_command(client, file)

    @mock.patch('horovod.runner.common.util.safe_shell_exec.GRACEFUL_TERMINATION_TIME_S', 0.5)
    def test_spark_task_service_abort_command(self):
        with spark_task_service(index=0) as (service, client, _):
            with tempdir() as d:
                file = os.path.sep.join([d, 'command_executed'])
                sleep = safe_shell_exec.GRACEFUL_TERMINATION_TIME_S * 2 + 2.0

                start = time.time()
                client.run_command('set -x; sleep {} && touch {}'.format(sleep, file), {})
                client.abort_command()
                client.wait_for_command_termination(delay=0.1)
                duration = time.time() - start

                self.assertLess(duration, safe_shell_exec.GRACEFUL_TERMINATION_TIME_S + 1.0)
                self.assertFalse(os.path.exists(file))

    def test_spark_task_service_abort_no_command(self):
        with spark_task_service(index=0) as (service, client, _):
            with tempdir() as d:
                file = os.path.sep.join([d, 'command_executed'])
                client.abort_command()
                self.do_test_spark_task_service_executes_command(client, file)

    @pytest.mark.skipif(LooseVersion(pyspark.__version__) < LooseVersion('3.0.0'),
                        reason='get_available_devices only supported in Spark 3.0 and above')
    def test_get_available_devices(self):
        res = run_get_available_devices()
        # we expect res being list of (devices, rank)
        # to be either [(['0'], 0), (['1'], 1)] or [(['1'], 0), (['0'], 1)]
        self.assertEqual([0, 1], [rank for (_, rank) in res])
        self.assertListEqual([['0'], ['1']], sorted([devices for (devices, _) in res]))

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

    def test_dbfs_local_store(self):
        import h5py
        import io
        import os

        import tensorflow
        from tensorflow import keras

        # test Store.create will not create DBFSLocalStore on non-databricks environment
        local_store = Store.create("/dbfs/tmp/test_local_dir")
        assert isinstance(local_store, LocalStore)

        # test Store.create will create DBFSLocalStore on databricks environment
        try:
            os.environ["DATABRICKS_RUNTIME_VERSION"] = "7.4"
            dbfs_local_store = Store.create("/dbfs/tmp/test_local_dir1")
            assert isinstance(dbfs_local_store, DBFSLocalStore)
            dbfs_local_store = Store.create("dbfs:/tmp/test_local_dir2")
            assert isinstance(dbfs_local_store, DBFSLocalStore)
            dbfs_local_store = Store.create("file:///dbfs/tmp/test_local_dir3")
            assert isinstance(dbfs_local_store, DBFSLocalStore)
        finally:
            if "DATABRICKS_RUNTIME_VERSION" in os.environ:
                del os.environ["DATABRICKS_RUNTIME_VERSION"]

        # test get_checkpoint_filename suffix
        # Use a tmp path for testing.
        dbfs_store = DBFSLocalStore("/tmp/test_dbfs_dir")
        dbfs_ckpt_name = dbfs_store.get_checkpoint_filename()
        assert dbfs_ckpt_name.endswith(".tf")

        # prepare for testing read_serialized_keras_model
        model = keras.Sequential([keras.Input([32]), keras.layers.Dense(1)])

        def deserialize_keras_model(serialized_model):
            model_bytes = codec.loads_base64(serialized_model)
            bio = io.BytesIO(model_bytes)
            with h5py.File(bio, 'r') as f:
                return keras.models.load_model(f)

        # test dbfs_store.read_serialized_keras_model
        get_dbfs_output_dir = dbfs_store.get_local_output_dir_fn("0")
        with get_dbfs_output_dir() as run_output_dir:
            dbfs_ckpt_path = run_output_dir + "/" + local_store.get_checkpoint_filename()
            if LooseVersion(tensorflow.__version__) < LooseVersion("2.0.0"):
                model.save_weights(dbfs_ckpt_path)
            else:
                model.save(dbfs_ckpt_path)
            serialized_model_dbfs = dbfs_store.read_serialized_keras_model(dbfs_ckpt_path, model,
                                                                           custom_objects={})
            reconstructed_model_dbfs = deserialize_keras_model(serialized_model_dbfs)
            if LooseVersion(tensorflow.__version__) >= LooseVersion("2.3.0"):
                assert reconstructed_model_dbfs.get_config() == model.get_config()

        # test local_store.read_serialized_keras_model
        with tempdir() as tmp:
            local_store = Store.create(tmp)
            get_local_output_dir = local_store.get_local_output_dir_fn("0")
            with get_local_output_dir() as run_output_dir:
                local_ckpt_path = run_output_dir + "/" + local_store.get_checkpoint_filename()
                model.save(local_ckpt_path, save_format='h5')

                serialized_model_local = \
                    local_store.read_serialized_keras_model(local_ckpt_path, model,
                                                            custom_objects={})

                reconstructed_model_local = deserialize_keras_model(serialized_model_local)
                if LooseVersion(tensorflow.__version__) >= LooseVersion("2.3.0"):
                    assert reconstructed_model_local.get_config() == model.get_config()


    def test_output_df_schema(self):
        label_cols = ['y1', 'y_embedding']
        output_cols = [col + '_pred' for col in label_cols]

        schema = StructType([StructField('x1', DoubleType()),
                             StructField('x2', IntegerType()),
                             StructField('features', VectorUDT()),
                             StructField('y1', FloatType()),
                             StructField('y_embedding', VectorUDT())])
        data = [[1.0, 1, DenseVector([1.0] * 12), 1.0, DenseVector([1.0] * 12)]] * 10

        with spark_session('test_df_cache') as spark:
            df = create_test_data_from_schema(spark, data, schema)
            metadata = util._get_metadata(df)

            output_schema = util.get_spark_df_output_schema(df.schema, label_cols, output_cols, metadata)

            # check output schema size
            assert len(output_schema.fields) == len(df.schema.fields) + len(output_cols)

            # check input col type
            output_field_dict = {f.name: f for f in output_schema.fields}
            for input_feild in df.schema.fields:
                assert type(output_field_dict[input_feild.name].dataType) == type(input_feild.dataType)
                assert output_field_dict[input_feild.name].nullable == input_feild.nullable

            # check output col type
            for label, output in zip(label_cols, output_cols):
                assert type(output_field_dict[label].dataType) == type(output_field_dict[output].dataType)
                assert output_field_dict[label].nullable == output_field_dict[output].nullable
