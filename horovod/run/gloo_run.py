# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

import errno
import math
import os
import signal
import sys
import threading
import time

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from horovod.run.common.util import env as env_util, safe_shell_exec
from horovod.run.common.util.hosts import get_host_assignments, parse_hosts
from horovod.run.elastic.driver import ElasticDriver
from horovod.run.elastic.rendezvous import create_rendezvous_handler
from horovod.run.http.http_server import RendezvousServer
from horovod.run.util import network, threads


def _pad_rank(rank, size):
    width = int(math.log10(size - 1)) + 1
    return str(rank).zfill(width)


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class MultiFile(object):
    def __init__(self, files):
        self._files = files

    def write(self, text):
        for f in self._files:
            f.write(text)

    def flush(self):
        for f in self._files:
            f.flush()


def _create_exec_command(settings, env, local_host_names, run_command):
    """
    executes the jobs defined by run command on hosts.
    :param hosts_alloc: list of dict indicating the allocating info.
    For example,
        [{'Hostname':'worker-0', 'Rank': 0, 'Local_rank': 0, 'Cross_rank':0,
            'Size':2, 'Local_size':1, 'Cross_size':2},
        {'Hostname':'worker-1', 'Rank': 1, 'Local_rank': 0, 'Cross_rank':1,
            'Size':2, 'Local_size':1, 'Cross_size':2}
        ]
    :type hosts_alloc: list(dict)
    :param remote_host_names: names that are resolved to one of the addresses
    of remote hosts interfaces.
    :type remote_host_names: set
    :param run_command: command to execute
    :type run_command: string
    :return:
    :rtype:
    """
    ssh_port_arg = '-p {ssh_port}'.format(ssh_port=settings.ssh_port) if settings.ssh_port else ''

    # Create a event for communication between threads
    event = threading.Event()

    def set_event_on_signal(signum, frame):
        event.set()

    signal.signal(signal.SIGINT, set_event_on_signal)
    signal.signal(signal.SIGTERM, set_event_on_signal)

    def get_command(slot_info):
        # generate env for rendezvous
        host_name = slot_info.hostname
        horovod_rendez_env = (
            'HOROVOD_HOSTNAME={hostname} '
            'HOROVOD_RANK={rank} '
            'HOROVOD_SIZE={size} '
            'HOROVOD_LOCAL_RANK={local_rank} '
            'HOROVOD_LOCAL_SIZE={local_size} '
            'HOROVOD_CROSS_RANK={cross_rank} '
            'HOROVOD_CROSS_SIZE={cross_size} '
            .format(hostname=host_name,
                    rank=slot_info.rank,
                    size=slot_info.size,
                    local_rank=slot_info.local_rank,
                    local_size=slot_info.local_size,
                    cross_rank=slot_info.cross_rank,
                    cross_size=slot_info.cross_size))

        # TODO: Workaround for over-buffered outputs. Investigate how mpirun avoids this problem.
        env['PYTHONUNBUFFERED'] = '1'
        local_command = '{horovod_env} {env} {run_command}' .format(
            horovod_env=horovod_rendez_env,
            env=' '.join(['%s=%s' % (key, quote(value)) for key, value in env.items()
                          if env_util.is_exportable(key)]),
            run_command=run_command)

        if host_name in local_host_names:
            command = local_command
        else:
            command = 'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} ' \
                      '{local_command}'\
                .format(host=host_name,
                        ssh_port_arg=ssh_port_arg,
                        local_command=quote('cd {pwd} > /dev/null 2>&1 ; {local_command}'
                                            .format(pwd=os.getcwd(), local_command=local_command)))
        return command

    def exec_command(slot_info, events=None):
        command = get_command(slot_info)
        if settings.verbose:
            print(command)

        # Redirect output if requested
        stdout = stderr = None
        stdout_file = stderr_file = None
        if settings.output_filename:
            padded_rank = _pad_rank(slot_info.rank, settings.num_proc)
            output_dir_rank = os.path.join(settings.output_filename, 'rank.{rank}'.format(rank=padded_rank))
            if not os.path.exists(output_dir_rank):
                os.mkdir(output_dir_rank)

            stdout_file = open(os.path.join(output_dir_rank, 'stdout'), 'w')
            stderr_file = open(os.path.join(output_dir_rank, 'stderr'), 'w')

            stdout = MultiFile([sys.stdout, stdout_file])
            stderr = MultiFile([sys.stderr, stderr_file])

        try:
            index = slot_info.rank
            events = [event] + (events or [])
            exit_code = safe_shell_exec.execute(command, index=index, stdout=stdout, stderr=stderr, events=events)
            if exit_code != 0:
                print('Process {idx} exit with status code {ec}.'.format(idx=index, ec=exit_code))
        except Exception as e:
            print('Exception happened during safe_shell_exec, exception '
                  'message: {message}'.format(message=e))
            exit_code = 1
        finally:
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()
        return exit_code, time.time()

    return exec_command


def get_run_command(command, common_intfs, port, elastic=False):
    server_ip = network.get_driver_ip(common_intfs)
    iface = list(common_intfs)[0]
    run_command = (
        'HOROVOD_GLOO_RENDEZVOUS_ADDR={addr} '
        'HOROVOD_GLOO_RENDEZVOUS_PORT={port} '
        'HOROVOD_CONTROLLER=gloo '
        'HOROVOD_CPU_OPERATIONS=gloo '
        'HOROVOD_GLOO_IFACE={iface} '
        'NCCL_SOCKET_IFNAME={common_intfs} '
        '{elastic} '
        '{command}'  # expect a lot of environment variables
        .format(addr=server_ip,
                port=port,
                iface=iface,  # TODO: add multiple ifaces in future
                common_intfs=','.join(common_intfs),
                elastic='HOROVOD_ELASTIC=1' if elastic else '',
                command=' '.join(quote(par) for par in command)))
    return run_command


def gloo_run_elastic(settings, env, command, get_common_intfs):
    # Make the output directory if it does not exist
    if settings.output_filename:
        _mkdir_p(settings.output_filename)

    rendezvous = RendezvousServer(settings.verbose)
    driver = ElasticDriver(rendezvous, settings.discovery_script,
                           settings.min_np, settings.max_np, settings.slots,
                           verbose=settings.verbose)

    handler = create_rendezvous_handler(driver)
    global_rendezv_port = rendezvous.start_server(handler)
    driver.wait_for_available_hosts(settings.num_proc)

    common_intfs, local_host_names = get_common_intfs(driver.get_available_hosts(), settings)
    run_command = get_run_command(command, common_intfs, global_rendezv_port, elastic=True)
    exec_command = _create_exec_command(settings, env, local_host_names, run_command)

    driver.start(settings.num_proc, exec_command)
    res = driver.get_results()

    for name, value in sorted(res.items(), key=lambda item: item[1][1]):
        exit_code, timestamp = value
        if exit_code != 0:
            raise RuntimeError('Horovod detected that one or more processes exited with non-zero '
                               'status, thus causing the job to be terminated. The first process '
                               'to do so was:\nProcess name: {name}\nExit code: {code}\n'
                               .format(name=name, code=exit_code))


def gloo_run(settings, local_host_names, common_intfs, env, command):
    # Make the output directory if it does not exist
    if settings.output_filename:
        _mkdir_p(settings.output_filename)

    # start global rendezvous server and get port that it is listening on
    rendezvous = RendezvousServer(settings.verbose)

    # allocate processes into slots
    hosts = parse_hosts(settings.hosts)
    host_alloc_plan = get_host_assignments(hosts, settings.num_proc)

    # start global rendezvous server and get port that it is listening on
    global_rendezv_port = rendezvous.start_server()
    rendezvous.httpd.init(host_alloc_plan)
    run_command = get_run_command(command, common_intfs, global_rendezv_port)
    exec_command = _create_exec_command(settings, env, local_host_names, run_command)

    # Each thread will use ssh command to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session. In case, the main thread receives
    # a SIGINT, the event will be set and the spawned threads will kill their
    # corresponding middleman processes and thus the jobs will be killed as
    # well.
    args_list = [[slot_info] for slot_info in host_alloc_plan]
    res = threads.execute_function_multithreaded(exec_command,
                                                 args_list,
                                                 block_until_all_done=True)

    for name, value in sorted(res.items(), key=lambda item: item[1][1]):
        exit_code, timestamp = value
        if exit_code != 0:
            raise RuntimeError('Horovod detected that one or more processes exited with non-zero '
                               'status, thus causing the job to be terminated. The first process '
                               'to do so was:\nProcess name: {name}\nExit code: {code}\n'
                               .format(name=name, code=exit_code))
