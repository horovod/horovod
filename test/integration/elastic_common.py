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

from __future__ import absolute_import

import contextlib
import json
import os
import sys

import mock
import pytest

from horovod.run.common.util import config_parser
from horovod.run.runner import parse_args, _run_elastic

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from common import override_args, temppath


DISCOVERY_SCRIPT_TEMPLATE = """#!/bin/bash
epoch=0
if [ -f {logfile} ]; then
    epoch=$(< {logfile} wc -l | tr -d '[:space:]')
fi
"""


def _get_discovery_lines(schedule_step, start, end):
    epoch, hosts = schedule_step
    hosts_str = os.linesep.join(['echo "{}"'.format(host) for host in hosts])
    if start and end:
        return hosts_str + os.linesep
    if start:
        return 'if [ "$epoch" == "{}" ]; then'.format(epoch) + os.linesep + hosts_str + os.linesep
    elif not start and not end:
        return 'elif [ "$epoch" == "{}" ]; then'.format(epoch) + os.linesep + hosts_str + os.linesep
    else:
        return 'else' + os.linesep + hosts_str + os.linesep + 'fi' + os.linesep


@contextlib.contextmanager
def _temp_discovery_script(logfile, discovery_schedule):
    with temppath() as discovery_script:
        with open(discovery_script, 'w') as f:
            f.write(DISCOVERY_SCRIPT_TEMPLATE.format(logfile=logfile) + os.linesep)
            for i, schedule_step in enumerate(discovery_schedule):
                f.write(_get_discovery_lines(schedule_step,
                                             start=i == 0,
                                             end=i == len(discovery_schedule) - 1))
        os.chmod(discovery_script, 0o755)
        yield discovery_script


class BaseElasticTests(object):
    def __init__(self, training_script, *args, **kwargs):
        self._training_script = training_script
        super(BaseElasticTests, self).__init__(*args, **kwargs)

    def _run(self, discovery_schedule, exit_schedule=None, np=2, min_np=2, max_np=4, hosts=None):
        with temppath() as logfile:
            with _temp_discovery_script(logfile, discovery_schedule) as discovery_script:
                command_args = ['horovodrun',
                                '-np', str(np),
                                '--min-np', str(min_np),
                                '--log-level', 'DEBUG']
                if hosts is not None:
                    command_args += ['-H', hosts]
                else:
                    command_args += ['--host-discovery-script', discovery_script,
                                     '--max-np', str(max_np)]
                command_args += ['python', self._training_script,
                                 '--logfile', logfile,
                                 '--discovery-schedule', json.dumps(discovery_schedule),
                                 '--exit-schedule', json.dumps(exit_schedule or {})]
                print(' '.join(command_args))

                with override_args(*command_args):
                    args = parse_args()
                    env = {}
                    config_parser.set_env_from_args(env, args)
                    _run_elastic(args)

                    with open(logfile, 'r') as f:
                        lines = f.readlines()

                    print('logfile:')
                    for line in lines:
                        print(line)

                    return [json.loads(line) for line in lines]

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_hosts_added_and_removed(self):
        discovery_schedule = [
            (0, ['localhost:2']),
            (1, ['localhost:2', '127.0.0.1:2']),
            (None, ['127.0.0.1:2']),
        ]

        results = self._run(discovery_schedule)
        for result in results:
            print(result)

        assert len(results) == 3

        assert results[0]['start_rank'] == 0
        assert results[0]['size'] == 2
        assert results[0]['hostname'] == 'localhost'

        assert results[1]['start_rank'] == 0
        assert results[1]['size'] == 4
        assert results[1]['hostname'] == 'localhost'

        assert results[2]['start_rank'] == 2
        assert results[2]['size'] == 2
        assert results[2]['hostname'] == '127.0.0.1'
        assert results[2]['rendezvous'] == 3

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_single_rank_failure(self):
        discovery_schedule = [
            (None, ['localhost:2', '127.0.0.1:2']),
        ]

        exit_schedule = {
            str((1, 0)): [0],
        }

        results = self._run(discovery_schedule, exit_schedule=exit_schedule)

        assert len(results) == 3

        assert results[0]['start_rank'] == 0
        assert results[0]['size'] == 4
        assert results[0]['rendezvous'] == 1

        assert results[1]['start_rank'] == 2
        assert results[1]['size'] == 2
        assert results[1]['rendezvous'] == 2

        assert results[2]['start_rank'] == 2
        assert results[2]['size'] == 2
        assert results[2]['rendezvous'] == 2

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_without_scaling(self):
        discovery_schedule = [
            (None, ['localhost:2', '127.0.0.1:2']),
        ]

        hosts = 'localhost:2,127.0.0.1:2'

        exit_schedule = {
            str((1, 0)): [0],
        }

        results = self._run(discovery_schedule, hosts=hosts, exit_schedule=exit_schedule)

        assert len(results) == 3

        assert results[0]['start_rank'] == 0
        assert results[0]['size'] == 4
        assert results[0]['rendezvous'] == 1

        assert results[1]['start_rank'] == 2
        assert results[1]['size'] == 2
        assert results[1]['rendezvous'] == 2

        assert results[2]['start_rank'] == 2
        assert results[2]['size'] == 2
        assert results[2]['rendezvous'] == 2

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_all_ranks_failure(self):
        discovery_schedule = [
            (None, ['localhost:2', '127.0.0.1:2']),
        ]

        exit_schedule = {
            str((1, 0)): [0, 1, 2, 3],
        }

        message = 'Horovod detected that one or more processes exited with non-zero status'
        with pytest.raises(RuntimeError, match=message):
            self._run(discovery_schedule, exit_schedule=exit_schedule)

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_all_hosts_blacklisted(self):
        discovery_schedule = [
            (None, ['localhost:2', '127.0.0.1:2']),
        ]

        exit_schedule = {
            str((1, 0)): [0, 2],
        }

        message = 'Horovod detected that one or more processes exited with non-zero status'
        with pytest.raises(RuntimeError, match=message):
            self._run(discovery_schedule, exit_schedule=exit_schedule)

    @mock.patch('horovod.run.elastic.driver.START_TIMEOUT_SECS', 1)
    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_min_hosts_timeout(self):
        discovery_schedule = [
            (None, ['localhost:2', '127.0.0.1:2']),
        ]

        exit_schedule = {
            str((1, 0)): [0],
        }

        message = 'Horovod detected that one or more processes exited with non-zero status'
        with pytest.raises(RuntimeError, match=message):
            self._run(discovery_schedule, exit_schedule=exit_schedule, np=4, min_np=4)
