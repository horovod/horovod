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

import io
import os
import sys

from socket import AF_INET
from psutil import net_if_addrs

from horovod.runner.common.service import driver_service
from horovod.runner.common.util import codec, safe_shell_exec
from horovod.runner.task import task_service
from horovod.runner.util import cache, lsf, network, threads
from horovod.runner.util.remote import get_remote_command


class HorovodRunDriverService(driver_service.BasicDriverService):
    NAME = 'horovod driver service'

    def __init__(self, num_hosts, key, nics):
        super(HorovodRunDriverService, self).__init__(num_hosts,
                                                      HorovodRunDriverService.NAME,
                                                      key, nics)


class HorovodRunDriverClient(driver_service.BasicDriverClient):
    def __init__(self, driver_addresses, key, verbose, match_intf=False):
        super(HorovodRunDriverClient, self).__init__(
            HorovodRunDriverService.NAME,
            driver_addresses,
            key,
            verbose,
            match_intf=match_intf)


def _launch_task_servers(all_host_names, local_host_names, driver_addresses,
                         settings):
    """
    Executes the task server and service client task for registration on the
    hosts.
    :param all_host_names: list of addresses. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type all_host_names: list(string)
    :param local_host_names: names that are resolved to one of the addresses
    of local hosts interfaces. For example,
        set(['localhost', '127.0.0.1'])
    :type local_host_names: set
    :param driver_addresses: map of interfaces and their address and port for
    the service. For example:
        {
            'lo': [('127.0.0.1', 34588)],
            'docker0': [('172.122.10.1', 34588)],
            'eth0': [('11.111.33.73', 34588)]
        }
    :type driver_addresses: map
    :param settings: the object that contains the setting for running horovod
    :type settings: horovod.runner.common.util.settings.Settings
    :return:
    :rtype:
    """

    def _exec_command(command):
        host_output = io.StringIO()
        try:
            exit_code = safe_shell_exec.execute(command,
                                                stdout=host_output,
                                                stderr=host_output)
            if exit_code != 0:
                print(
                    'Launching horovod task function was not '
                    'successful:\n{host_output}'
                    .format(host_output=host_output.getvalue()))
                os._exit(exit_code)
        finally:
            host_output.close()
        return exit_code

    args_list = []
    num_hosts = len(all_host_names)
    for index in range(num_hosts):
        host_name = all_host_names[index]
        command = \
            '{python} -m horovod.runner.task_fn {index} {num_hosts} ' \
            '{driver_addresses} {settings}' \
            .format(python=sys.executable,
                    index=codec.dumps_base64(index),
                    num_hosts=codec.dumps_base64(num_hosts),
                    driver_addresses=codec.dumps_base64(driver_addresses),
                    settings=codec.dumps_base64(settings))
        if host_name not in local_host_names:
            command = get_remote_command(command,
                                         host=host_name,
                                         port=settings.ssh_port,
                                         identity_file=settings.ssh_identity_file)

        if settings.verbose >= 2:
            print('Launching horovod task function: {}'.format(command))
        args_list.append([command])
    # Each thread will use ssh command to launch the server on one task. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session -- and the the task server --
    # will be bound to the thread. In case, the horovod process dies, all
    # the ssh sessions and all the task servers will die as well.
    threads.execute_function_multithreaded(_exec_command,
                                           args_list,
                                           block_until_all_done=False)

def _run_probe(driver, settings, num_hosts):
       # wait for all the hosts to register with the service service.
    if settings.verbose >= 2:
        print('Waiting for the hosts to acknowledge.')
    driver.wait_for_initial_registration(settings.start_timeout)
    tasks = [
        task_service.HorovodRunTaskClient(
            index,
            driver.task_addresses_for_driver(index),
            settings.key,
            settings.verbose) for index in range(
            num_hosts)]
    # Notify all the drivers that the initial registration is complete.
    for task in tasks:
        task.notify_initial_registration_complete()
    if settings.verbose >= 2:
        print('Notified all the hosts that the registration is complete.')
    # Each worker should probe the interfaces of the next worker in a ring
    # manner and filter only the routed ones -- it should filter out
    # interfaces that are not really connected to any external networks
    # such as lo0 with address 127.0.0.1.
    if settings.verbose >= 2:
        print('Waiting for hosts to perform host-to-host interface checking.')
    driver.wait_for_task_to_task_address_updates(settings.start_timeout)
    if settings.verbose >= 2:
        print('Host-to-host interface checking successful.')
    # Determine a set of common interfaces for task-to-task communication.
    nics = set(driver.task_addresses_for_tasks(0).keys())
    for index in range(1, num_hosts):
        nics.intersection_update(
            driver.task_addresses_for_tasks(index).keys())
    if not nics:
        raise Exception(
            'Unable to find a set of common task-to-task communication interfaces: %s'
            % [(index, driver.task_addresses_for_tasks(index))
               for index in range(num_hosts)])
    return nics


@cache.use_cache()
def _driver_fn(all_host_names, local_host_names, settings):
    """
    launches the service service, launches the task service on each worker and
    have them register with the service service. Each worker probes all the
    interfaces of the worker index + 1 (in a ring manner) and only keeps the
    routed interfaces. Function returns the intersection of the set of all the
    routed interfaces on all the workers.
    :param all_host_names: list of addresses. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type all_host_names: list(string)
    :param local_host_names: host names that resolve into a local addresses.
    :type local_host_names: set
    :param settings: the object that contains the setting for running horovod
    :type settings: horovod.runner.common.util.settings.Settings
    :return: example: ['eth0', 'eth1']
    :rtype: list[string]
    """
    # Launch a TCP server called service service on the host running horovod
    num_hosts = len(all_host_names)
    driver = HorovodRunDriverService(num_hosts, settings.key, settings.nics)
    if settings.verbose >= 2:
        print('Launched horovod server.')
    # Have all the workers register themselves with the service service.
    _launch_task_servers(all_host_names, local_host_names,
                         driver.addresses(), settings)
    if settings.verbose >= 2:
        print('Attempted to launch horovod task servers.')
    try:
        return _run_probe(driver, settings, num_hosts)
    finally:
        driver.shutdown()

def get_local_interfaces(settings):
    if settings.verbose >= 2:
        print('All hosts are local, finding the interfaces '
              'with address 127.0.0.1')
    # If all the given hosts are local, find the interfaces with address
    # 127.0.0.1
    nics = set()
    for iface, addrs in net_if_addrs().items():
        if settings.nics and iface not in settings.nics:
            continue
        for addr in addrs:
            if addr.family == AF_INET and addr.address == '127.0.0.1':
                nics.add(iface)
                break

    if len(nics) == 0:
        raise ValueError('No interface is found for address 127.0.0.1.')

    if settings.verbose >= 2:
        print('Local interface found ' + ' '.join(nics))
    return nics


def get_common_interfaces(settings, all_host_names, remote_host_names=None, fn_cache=None):
    '''
    Find the set of common and routed interfaces on all the hosts.
    :param settings: the object that contains the setting for running horovod
    :type settings: horovod.runner.common.util.settings.Settings
    :param all_host_names: list of the host names
    :type all_host_names: list(string)
    :param remote_host_names: list of the remote host names.
    :type remote_host_names: list(string)
    :param fn_cache: Cache storing the results of checks performed by horovod
    :type fn_cache: horovod.runner.util.cache.Cache
    :return: List of common interfaces
    '''
    # Skipping interface discovery for LSF cluster as it slows down considerably the job start
    if lsf.LSFUtils.using_lsf():
        return None

    if remote_host_names is None:
        remote_host_names = network.filter_local_addresses(all_host_names)

    if len(remote_host_names) > 0:
        if settings.nics:
            # If args.nics is provided, we will use those interfaces. All the workers
            # must have at least one of those interfaces available.
            nics = settings.nics
        else:
            # Find the set of common, routed interfaces on all the hosts (remote
            # and local) and specify it in the args to be used by NCCL. It is
            # expected that the following function will find at least one interface
            # otherwise, it will raise an exception.
            if settings.verbose >= 2:
                print('Testing interfaces on all the hosts.')

            local_host_names = set(all_host_names) - set(remote_host_names)
            nics = _driver_fn(all_host_names, local_host_names, settings, fn_cache=fn_cache)

            if settings.verbose >= 2:
                print('Interfaces on all the hosts were successfully checked.')
                print('Common interface found: ' + ' '.join(nics))

    else:
        nics = get_local_interfaces(settings)
    return nics
