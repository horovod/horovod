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

import collections
import copy
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
from horovod.run.http.http_server import RendezvousServer
from horovod.run.util import threads


class HostInfo:
    def __init__(self, host_item):
        hostname, slots = host_item.strip().split(':')
        self.hostname = hostname
        self.slots = int(slots)


class SlotInfo:
    def __init__(self, hostname, rank, local_rank, cross_rank, size):
        self.hostname = hostname
        self.rank = rank
        self.size = size
        self.local_rank = local_rank
        self.local_size = None
        self.cross_rank = cross_rank
        self.cross_size = None


def _allocate(hosts, np):
    """
    Find the allocation of processes on hosts, this function will try to
    allocate as many as possible processes on the same host to leverage
    local network.
    :param hosts: list of addresses and number of processes on each host.
    For example,
        'worker-0:2,worker-1:2'
        '10.11.11.11:4,10.11.11.12,4'
    :type hosts: string
    :param np: total number of processes to be allocated
    :type np: int
    :return: a list of the allocation of process on hosts in a AllocInfo object.
            Members in the object include: hostname, rank, local_rank, cross_rank,
            total_size, local_size, cross_size
    :rtype: list[dict()]
    """

    host_list = []
    # split the host string to host list
    for host_item in hosts.split(','):
        host_list.append(HostInfo(host_item))

    rank = 0
    alloc_list = []

    # key: local_rank; value: cross_size for this local_rank
    local_sizes = collections.defaultdict(int)
    # key: cross_rank; value: local_size for this cross_rank
    cross_sizes = collections.defaultdict(int)

    # allocate processes into slots
    for host_idx, host_info in enumerate(host_list):
        for local_rank in range(host_info.slots):
            if rank == np:
                break
            cross_rank = host_idx
            alloc_list.append(
                SlotInfo(
                    host_info.hostname,
                    rank,
                    local_rank,
                    cross_rank,
                    np))
            cross_sizes[local_rank] += 1
            local_sizes[cross_rank] += 1
            rank += 1

    if rank < np:
        raise ValueError("Process number should not be larger than "
                         "total available slots.")

    # Fill in the local_size and cross_size because we can only know these number after
    # allocation is done.
    for alloc_item in alloc_list:
        alloc_item.local_size = local_sizes[alloc_item.cross_rank]
        alloc_item.cross_size = cross_sizes[alloc_item.local_rank]

    return alloc_list


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


def _alloc_info_to_command_fn(run_command, env):
    def alloc_info_to_command(alloc_info):
        """
        Given an alloc_info, creates a command used by gloo to launch a single job.

        :param alloc_info: host and slot to execute the run command on
        :return:
        """
        # generate env for rendezvous
        horovod_rendez_env = 'HOROVOD_RANK={rank} HOROVOD_SIZE={size} ' \
                             'HOROVOD_LOCAL_RANK={local_rank} HOROVOD_LOCAL_SIZE={local_size} ' \
                             'HOROVOD_CROSS_RANK={cross_rank} HOROVOD_CROSS_SIZE={cross_size} ' \
            .format(rank=alloc_info.rank, size=alloc_info.size,
                    local_rank=alloc_info.local_rank, local_size=alloc_info.local_size,
                    cross_rank=alloc_info.cross_rank, cross_size=alloc_info.cross_size)

        return '{horovod_env} {env} {run_command}' .format(
            horovod_env=horovod_rendez_env,
            env=' '.join(['%s=%s' % (key, quote(value)) for key, value in env.items()
                          if env_util.is_exportable(key)]),
            run_command=run_command)

    return alloc_info_to_command


def _exec_command_fn(settings, remote_host_names):
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
    :param _run_command: command to execute
    :type _run_command: string
    :return:
    :rtype:
    """
    ssh_port_arg = '-p {ssh_port}'.format(ssh_port=settings.ssh_port) if settings.ssh_port else ''

    def _exec_command(command, alloc_info, event):
        index = alloc_info.rank
        host_name = alloc_info.hostname

        if host_name in remote_host_names:
            command = 'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} ' \
                      '{local_command}'\
                .format(host=host_name,
                        ssh_port_arg=ssh_port_arg,
                        local_command=quote('cd {pwd} > /dev/null 2>&1 ; {local_command}'
                                            .format(pwd=os.getcwd(), local_command=command)))

        if settings.verbose:
            print(command)

        # Redirect output if requested
        stdout = stderr = None
        stdout_file = stderr_file = None
        if settings.output_filename:
            padded_rank = _pad_rank(index, settings.num_proc)
            output_dir_rank = os.path.join(settings.output_filename, 'rank.{rank}'.format(rank=padded_rank))
            if not os.path.exists(output_dir_rank):
                os.mkdir(output_dir_rank)

            stdout_file = open(os.path.join(output_dir_rank, 'stdout'), 'w')
            stderr_file = open(os.path.join(output_dir_rank, 'stderr'), 'w')

            stdout = MultiFile([sys.stdout, stdout_file])
            stderr = MultiFile([sys.stderr, stderr_file])

        try:
            exit_code = safe_shell_exec.execute(command, index=index, events=[event], stdout=stdout, stderr=stderr)
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

    return _exec_command


def launch_gloo(command, exec_command, settings, nics, env, server_ip):
    """
    Launches the given command multiple times using gloo.
    Each command is launched via exec_command.

    :param command: command to launch
    :param exec_command: means to execute a single command
    :param settings: settings for the distribution
    :param nics: common interfaces
    :param env: environment to use
    :param server_ip: ip to use for rendezvous server
    """
    # allocate processes into slots
    host_alloc_plan = _allocate(settings.hosts, settings.num_proc)

    # create global rendezvous server
    global_rendezv = RendezvousServer(settings.verbose)
    # Start rendezvous server and get port that it is listening
    global_rendezv_port = global_rendezv.start_server(host_alloc_plan)

    run_command = (
        'HOROVOD_GLOO_RENDEZVOUS_ADDR={addr} '
        'HOROVOD_GLOO_RENDEZVOUS_PORT={port} '
        'HOROVOD_CONTROLLER=gloo '
        'HOROVOD_CPU_OPERATIONS=gloo '
        'HOROVOD_GLOO_IFACE={iface} '
        'NCCL_SOCKET_IFNAME={nics} '
        '{command}'  # expect a lot of environment variables
            .format(addr=server_ip,
                    port=global_rendezv_port,
                    iface=list(nics)[0],  # TODO: add multiple ifaces in future
                    nics=','.join(nics),
                    command=' '.join(quote(par) for par in command)))

    # Create a event for communication between threads
    event = threading.Event()

    def set_event_on_sigterm(signum, frame):
        event.set()

    signal.signal(signal.SIGINT, set_event_on_sigterm)
    signal.signal(signal.SIGTERM, set_event_on_sigterm)

    # TODO: Workaround for over-buffered outputs. Investigate how mpirun avoids this problem.
    env = copy.copy(env)  # copy env so we do not leak env modifications
    env['PYTHONUNBUFFERED'] = '1'

    # In case, the main thread receives a SIGINT, the event will be set so the spawned threads can
    # kill their corresponding middleman processes so the jobs can be killed as well.
    alloc_info_to_command = _alloc_info_to_command_fn(run_command, env)
    args_list = [[alloc_info_to_command(alloc_info), alloc_info, event]
                 for alloc_info in host_alloc_plan]

    # Make the output directory if it does not exist
    if settings.output_filename:
        _mkdir_p(settings.output_filename)

    # If an error occurs in one thread, entire process will be terminated.
    # Otherwise, threads will keep running.
    res = threads.execute_function_multithreaded(exec_command, args_list, block_until_all_done=True)

    for name, value in sorted(res.items(), key=lambda item: item[1][1]):
        exit_code, timestamp = value
        if exit_code != 0:
            raise RuntimeError('Gloo job detected that one or more processes exited with non-zero '
                               'status, thus causing the job to be terminated. The first process '
                               'to do so was:\nProcess name: {name}\nExit code: {code}\n'
                               .format(name=name, code=exit_code))


def gloo_run(settings, remote_host_names, nics, env, server_ip, command):
    # Each thread will use ssh command to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session.
    exec_command = _exec_command_fn(settings, remote_host_names)
    launch_gloo(command, exec_command, settings, nics, env, server_ip)
