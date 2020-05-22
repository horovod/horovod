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
import itertools
import unittest
import warnings

from parameterized import parameterized
import pytest
import mock

from horovod.common.util import gloo_built, mpi_built
from horovod.run.runner import HorovodArgs, _check_all_hosts_ssh_successful, _run


def values_name_func(testcase_func, param_num, param):
    return '_'.join([testcase_func.__name__] + list(param.args))


def fn(fail_rank=None):
    from horovod.run.common.util.env import get_env_rank_and_size
    rank = get_env_rank_and_size()
    if rank and rank[0] == fail_rank:
        raise RuntimeError('failing as expected')
    return rank


class RunIntegrationTests(unittest.TestCase):
    """
    Integration tests for horovod.run.
    """

    def __init__(self, *args, **kwargs):
        super(RunIntegrationTests, self).__init__(*args, **kwargs)
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

        hargs = HorovodArgs()
        hargs.np = 4
        hargs.hosts = ','.join(['{}:2'.format(host) for host in hosts])
        hargs.use_gloo = controller == 'gloo'
        hargs.use_mpi = controller == 'mpi'
        hargs.run_func = run_func
        hargs.command = [command] if command else None
        hargs.nics = []
        hargs.verbose = 2
        hargs.disable_cache = True

        with mock.patch('horovod.run.runner.network.filter_local_addresses',
                        side_effect=lambda hosts: [host for host in hosts if host not in local_hosts]), \
             mock.patch('horovod.run.gloo_run.network.get_local_host_addresses',
                        return_value=local_hosts), \
             mock.patch('horovod.run.gloo_run.network.resolve_host_address',
                        side_effect=lambda host: host):
            yield hargs

    @parameterized.expand(itertools.product(['mpi', 'gloo'], ['local', 'mixed', 'remote'], ['func', 'cmd']),
                          name_func=values_name_func)
    def test_run_success(self, controller, mode, run):
        if controller == 'gloo' and not gloo_built():
            self.skipTest("Gloo is not available")
        if controller == 'mpi' and not mpi_built():
            self.skipTest("MPI is not available")

        if controller == 'mpi' and run == 'cmd':
            self.skipTest('MPI with command cannot be tested as it exits the process')

        self.do_test_run_with_controller_success(controller, mode, run)

    @parameterized.expand(itertools.product(['mpi', 'gloo'], ['local', 'mixed', 'remote'], ['func', 'cmd']),
                          name_func=values_name_func)
    def test_run_failure(self, controller, mode, run):
        if controller == 'gloo' and not gloo_built():
            self.skipTest("Gloo is not available")
        if controller == 'mpi' and not mpi_built():
            self.skipTest("MPI is not available")

        if controller == 'mpi' and run == 'cmd':
            self.skipTest('MPI with command cannot be tested as it exits the process')

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

        with self.horovod_args(mode, controller, run_func=run_func, command=command) as hargs:
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

        with self.horovod_args(mode, controller=controller, run_func=run_func, command=command) as hargs:
            # an unsuccessful command run on the Horovod cluster will raise an exception
            with pytest.raises(RuntimeError, match=exception):
                _run(hargs)
