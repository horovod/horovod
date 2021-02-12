# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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
import io
import itertools
import os
import sys
import warnings

import mock
from parameterized import parameterized
import pytest
import unittest

from horovod.common.util import gloo_built, mpi_built
from horovod.runner.common.util import safe_shell_exec
from horovod.runner import _HorovodArgs
from horovod.runner.launch import _check_all_hosts_ssh_successful, _run
from horovod.runner.mpi_run import mpi_available, is_mpich, is_intel_mpi

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import capture


def values_name_func(testcase_func, param_num, param):
    return '_'.join([testcase_func.__name__] + list(param.args))


def fn(fail_rank=None):
    from horovod.runner.common.util.env import get_env_rank_and_size
    rank = get_env_rank_and_size()
    if rank and rank[0] == fail_rank:
        raise RuntimeError('failing as expected')
    return rank


class StaticRunTests(unittest.TestCase):
    """
    Integration tests for horovod.runner.
    """

    def __init__(self, *args, **kwargs):
        super(StaticRunTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    @contextlib.contextmanager
    def horovod_args(self, mode, controller, run_func=None, command=None):
        if mode == 'local':
            local_hosts = ['localhost', '127.0.0.1']
            remote_hosts = []
        elif mode == 'remote':
            local_hosts = []
            remote_hosts = ['localhost', '127.0.0.1']
        else:
            local_hosts = ['localhost']
            remote_hosts = ['127.0.0.1']
        hosts = local_hosts.copy()
        hosts.extend(remote_hosts)

        if remote_hosts:
            ssh_works = _check_all_hosts_ssh_successful(remote_hosts, fn_cache=None)
            if not ssh_works:
                self.skipTest('password-less ssh to {} is required for this test'
                              .format(' and '.join(remote_hosts)))

        hargs = _HorovodArgs()
        hargs.np = 4
        hargs.hosts = ','.join(['{}:2'.format(host) for host in hosts])
        hargs.use_gloo = controller == 'gloo'
        hargs.use_mpi = controller == 'mpi'
        hargs.run_func = run_func
        hargs.command = [command] if command else None
        hargs.nics = []
        hargs.verbose = 2
        hargs.disable_cache = True

        stdout = io.StringIO()
        try:
            with capture(stdout=stdout):
                with mock.patch('horovod.runner.launch.network.filter_local_addresses',
                                side_effect=lambda hosts: [host for host in hosts if host not in local_hosts]), \
                     mock.patch('horovod.runner.gloo_run.network.get_local_host_addresses',
                                return_value=local_hosts), \
                     mock.patch('horovod.runner.gloo_run.network.resolve_host_address',
                                side_effect=lambda host: host), \
                     mock.patch('horovod.runner.mpi_run.os.execve') as exec:
                    yield hargs, exec
        finally:
            stdout = stdout.readlines()
            print(''.join(stdout), file=sys.stdout)

        if mode == 'local':
            self.assertIn('Remote host found: \n', stdout)
            self.assertIn('All hosts are local, finding the interfaces with address 127.0.0.1\n', stdout)
            self.assertEqual(1, len([line for line in stdout if line.startswith('Local interface found ')]))
        elif mode == 'mixed':
            self.assertIn('Remote host found: 127.0.0.1\n', stdout)
        elif mode == 'remote':
            self.assertIn('Remote host found: localhost 127.0.0.1\n', stdout)
        else:
            raise RuntimeError('unknown mode {}'.format(mode))

        if mode in ['mixed', 'remote']:
            self.assertIn('Checking ssh on all remote hosts.\n', stdout)
            self.assertIn('SSH was successful into all the remote hosts.\n', stdout)
            self.assertIn('Testing interfaces on all the hosts.\n', stdout)
            self.assertIn('Launched horovod server.\n', stdout)
            self.assertEqual(2, len([line for line in stdout if line.startswith('Launching horovod task function: ')]))
            self.assertEqual(1, len([line for line in stdout if line.startswith('Launching horovod task function: ssh -o PasswordAuthentication=no -o StrictHostKeyChecking=no 127.0.0.1 ')]), stdout)
            if mode == 'remote':
                self.assertEqual(1, len([line for line in stdout if line.startswith('Launching horovod task function: ssh -o PasswordAuthentication=no -o StrictHostKeyChecking=no localhost ')]))
            else:
                self.assertEqual(1, len([line for line in stdout if line.startswith('Launching horovod task function: ') and not line.startswith('Launching horovod task function: ssh')]))
            self.assertIn('Attempted to launch horovod task servers.\n', stdout)
            self.assertIn('Waiting for the hosts to acknowledge.\n', stdout)
            self.assertIn('Notified all the hosts that the registration is complete.\n', stdout)
            self.assertIn('Waiting for hosts to perform host-to-host interface checking.\n', stdout)
            self.assertIn('Host-to-host interface checking successful.\n', stdout)
            self.assertIn('Interfaces on all the hosts were successfully checked.\n', stdout)
            self.assertEqual(1, len([line for line in stdout if line.startswith('Common interface found: ')]))

    @parameterized.expand(itertools.product(['mpi', 'gloo'], ['local', 'mixed', 'remote'], ['func', 'cmd']),
                          name_func=values_name_func)
    def test_run_success(self, controller, mode, run):
        if controller == 'gloo' and not gloo_built():
            self.skipTest("Gloo is not available")
        if controller == 'mpi':
            if not (mpi_built() and mpi_available()):
                self.skipTest("MPI is not available")
            if is_mpich():
                self.skipTest("MPICH is not testable")
            if is_intel_mpi():
                self.skipTest("Intel(R) MPI is not testable because it is based on MPICH")

        self.do_test_run_with_controller_success(controller, mode, run)

    @parameterized.expand(itertools.product(['mpi', 'gloo'], ['local', 'mixed', 'remote'], ['func', 'cmd']),
                          name_func=values_name_func)
    def test_run_failure(self, controller, mode, run):
        if controller == 'gloo' and not gloo_built():
            self.skipTest("Gloo is not available")
        if controller == 'mpi':
            if not (mpi_built() and mpi_available()):
                self.skipTest("MPI is not available")
            if is_mpich():
                self.skipTest("MPICH is not testable")
            if is_intel_mpi():
                self.skipTest("Intel(R) MPI is not testable because it is based on MPICH")

        self.do_test_run_with_controller_failure(controller, mode, run)

    def do_test_run_with_controller_success(self, controller, mode, run):
        if run == 'func':
            command = None
            run_func = fn
        elif run == 'cmd':
            command = 'true'
            run_func = None
        else:
            self.fail('unknown run argument {}'.format(run))

        with self.horovod_args(mode, controller, run_func=run_func, command=command) as (hargs, exec):
            if controller == 'mpi' and run == 'cmd':
                self.assertIsNone(_run(hargs))
                exec.assert_called_once()
                args, kwargs = exec.call_args
                executable, args, env = args
                self.assertEqual('/bin/sh', executable)
                self.assertEqual(3, len(args))
                self.assertEqual('/bin/sh', args[0])
                self.assertEqual('-c', args[1])
                exit_code = safe_shell_exec.execute(args[2], env)
                self.assertEqual(0, exit_code)
            else:
                actual = _run(hargs)
                expected = list([(rank, hargs.np) for rank in range(hargs.np)]) if run == 'func' else None
                self.assertEqual(expected, actual)

    def do_test_run_with_controller_failure(self, controller, mode, run):
        if run == 'func':
            command = None
            run_func = lambda: fn(0)
        elif run == 'cmd':
            command = 'false'
            run_func = None
        else:
            self.fail('unknown run argument {}'.format(run))

        if controller == 'mpi':
            exception = 'mpirun failed with exit code 1'
        else:
            exception = 'Horovod detected that one or more processes exited with non-zero status'

        with self.horovod_args(mode, controller=controller, run_func=run_func, command=command) as (hargs, exec):
            if controller == 'mpi' and run == 'cmd':
                self.assertIsNone(_run(hargs))
                exec.assert_called_once()
                args, kwargs = exec.call_args
                executable, args, env = args
                self.assertEqual('/bin/sh', executable)
                self.assertEqual(3, len(args))
                self.assertEqual('/bin/sh', args[0])
                self.assertEqual('-c', args[1])
                exit_code = safe_shell_exec.execute(args[2], env)
                self.assertEqual(1, exit_code)
            else:
                with pytest.raises(RuntimeError, match=exception):
                    _run(hargs)
