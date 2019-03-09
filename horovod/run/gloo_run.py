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

import os
import netifaces as ni
from horovod.run.rendezvous.http_server import RendezvousServer
from horovod.run.common.util import env as env_util, safe_shell_exec
from horovod.run.util import threads
import time

try:
    from shlex import quote
except ImportError:
    from pipes import quote


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
    :return: a list of the allocation of process on hosts in a dictionary.
            Keys in the dict includes: hostname, rank, local_rank, cross_rank,
            total_size, local_size, cross_size
    :rtype: list[dict()]
    """
    res = []
    hosts_split = []

    # split the host string to host list
    for item in hosts.split(','):
        tmp = item.split(':')
        hosts_split.append([tmp[0], int(tmp[1])])

    idx = 0
    local_rank = 0
    rank = 0
    local_sizes = []
    cross_size = 0

    # place one process at each iteration
    while rank < np and idx < len(hosts_split):

        # current host is full, go to the next one
        if local_rank >= hosts_split[idx][1]:
            local_sizes.append(local_rank)
            idx += 1
            local_rank = 0
            continue

        alloc_item = {'Hostname': hosts_split[idx][0],
                      'Rank': rank,
                      'Local_rank': local_rank,
                      'Size': np}

        # If this is the first process on the host, increase host count
        if alloc_item['Local_rank'] == 0:
            cross_size += 1
        alloc_item['Cross_rank'] = idx

        res.append(alloc_item)
        local_rank += 1
        rank += 1

    if rank < np:
        raise Exception("Process number should not be larger than "
                        "total available slots.")

    local_sizes.append(local_rank)

    for item in res:
        item['Local_size'] = local_sizes[item['Cross_rank']]
        item['Cross_size'] = cross_size

    return res


def _launch_job(args, hosts_alloc, remote_host_names, _run_command):
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

    def _exec_command(_command, _index):
        if args.verbose >= 3:
            print(_command)
        try:
            exit_code = safe_shell_exec.execute(_command, index=_index)
            if exit_code != 0:
                os._exit(exit_code)
        except Exception as e:
            print('Exception happened during safe_shell_exec, exception '
                  'message: {message}'.format(message=e))
        return 0

    if args.ssh_port:
        ssh_port_arg = "-p {ssh_port}".format(ssh_port=args.ssh_port)
    else:
        ssh_port_arg = ""

    args_list = []
    for index in range(len(hosts_alloc)):
        alloc_item = hosts_alloc[index]

        # generate env for rendezvous
        horovod_rendez_env = 'HOROVOD_RANK={rank} HOROVOD_SIZE={size} ' \
                             'HOROVOD_LOCAL_RANK={local_rank} HOROVOD_LOCAL_SIZE={local_size} ' \
                             'HOROVOD_CROSS_RANK={cross_rank} HOROVOD_CROSS_SIZE={cross_size} ' \
            .format(rank=alloc_item['Rank'], size=alloc_item['Size'],
                    local_rank=alloc_item['Local_rank'], local_size=alloc_item['Local_size'],
                    cross_rank=alloc_item['Cross_rank'], cross_size=alloc_item['Cross_size'], )

        host_name = alloc_item['Hostname']
        if host_name not in remote_host_names:
            command = \
                '{horovod_env} ' \
                '{run_command}'.format(
                    horovod_env=horovod_rendez_env,
                    run_command=_run_command
                )
        else:
            command = \
                'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} ' \
                '\'{horovod_env} ' \
                '{run_command}\''.format(
                    host=host_name,
                    ssh_port_arg=ssh_port_arg,
                    horovod_env=horovod_rendez_env,
                    run_command=_run_command
                )
        args_list.append([command, index])
    # Each thread will use ssh command to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session -- and the the task server --
    # will be bound to the thread. In case, the horovodrun process dies, all
    # the ssh sessions and all the task servers will die as well.
    threads.execute_function_multithreaded(_exec_command,
                                           args_list,
                                           block_until_all_done=True)


def gloo_run(args, remote_host_names, common_intfs):
    # allocate processes into slots
    host_alloc = _allocate(args.host, args.np)

    # create global rendezvous server
    global_rendezv = RendezvousServer(args.np, args.verbose)
    # Start rendezvous server and get port that it is listening
    global_rendezv_port = global_rendezv.rendezvous()

    # get the server address
    if remote_host_names:
        # if remote hosts exist, need to use the address of common interface
        iface = list(common_intfs)[0]
        server_ip = ni.ifaddresses(iface)[ni.AF_INET][0]['addr']
    else:
        # if all hosts are local, use 'lo' as common interface
        iface = 'lo'
        server_ip = ni.ifaddresses(iface)[ni.AF_INET][0]['addr']

    # env for horovod logging level
    log_level_arg = 'HOROVOD_LOG_LEVEL=' + env_util.LOG_LEVEL_STR[args.verbose]

    env = os.environ.copy()
    run_command = (
        'HOROVOD_RENDEZVOUS_ADDR={addr} '
        'HOROVOD_RENDEZVOUS_PORT={port} '
        'HOROVOD_CONTROLLER=gloo '
        'HOROVOD_CPU_OPERATIONS=gloo '
        'HOROVOD_IFACE={iface} '
        '{log_level_arg} '
        'NCCL_SOCKET_IFNAME={common_intfs} '
        '{env} {command}'  # expect a lot of environment variables
            .format(addr=server_ip,
                    port=global_rendezv_port,
                    iface=iface,  # TODO: add multiple ifaces in future
                    log_level_arg=log_level_arg,
                    common_intfs=','.join(common_intfs),
                    env=' '.join('%s=%s' % (key, value) for key, value in env.items()
                                 if env_util.is_exportable(key) and env_util.is_exportable(value)),
                    command=' '.join(quote(par) for par in args.command))
    )

    _launch_job(args, host_alloc, remote_host_names, run_command)

    # Finalize
    # global_rendezv.finalize()
    # cross_rendezv.finalize()
    return
