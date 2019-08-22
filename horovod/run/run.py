# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import argparse
import hashlib
import os
import sys
import six
import re
import textwrap
try:
    from shlex import quote
except ImportError:
    from pipes import quote
import horovod

from horovod.common.util import (extension_available,
                                 gloo_built, mpi_built,
                                 nccl_built, ddl_built, mlsl_built)
from horovod.run.common.util import codec, safe_shell_exec, timeout, secret
from horovod.run.common.util import settings as hvd_settings
from horovod.run.driver import driver_service
from horovod.run.task import task_service
from horovod.run.util import cache, threads, network
from horovod.run.gloo_run import gloo_run
from horovod.run.mpi_run import mpi_run
from socket import AF_INET
from psutil import net_if_addrs

# Cached information of horovodrun functions be stored in this directory
CACHE_FOLDER = os.path.join(os.path.expanduser('~'), '.horovod')

# Cache entries will be stale if they are older than this number of minutes
CACHE_STALENESS_THRESHOLD_MINUTES = 60

# Number of retries for sshing into the hosts
SSH_RETRIES = 5


@cache.use_cache()
def _check_all_hosts_ssh_successful(host_addresses, ssh_port=None):
    """
    checks if ssh can successfully be performed to all the hosts.
    :param host_addresses: list of addresses to ssh into. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type host_addresses: list(strings)
    :return: Returns True if all ssh was successful into all the addresses.
    """

    def exec_command(command):
        exit_code = 1
        output_msg = ''

        # Try ssh 5 times
        for i in range(SSH_RETRIES):
            output = six.StringIO()
            try:
                exit_code = safe_shell_exec.execute(command,
                                                    stdout=output,
                                                    stderr=output)
                if exit_code == 0:
                    break
                else:
                    output_msg = output.getvalue()
            finally:
                output.close()
        return exit_code, output_msg

    ssh_port_arg = '-p {ssh_port}'.format(
        ssh_port=ssh_port) if ssh_port else ''

    ssh_command_format = 'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} date'

    args_list = [[ssh_command_format.format(host=host_address,
                                            ssh_port_arg=ssh_port_arg)]
                 for host_address in host_addresses]
    ssh_exit_codes = \
        threads.execute_function_multithreaded(exec_command,
                                               args_list)

    ssh_successful_to_all_hosts = True
    for index, ssh_status in six.iteritems(ssh_exit_codes):
        exit_code, output_msg = ssh_status[0], ssh_status[1]
        if exit_code != 0:
            print('ssh not successful for host {host}:\n{msg_output}'
                  .format(host=host_addresses[index],
                          msg_output=output_msg))

            ssh_successful_to_all_hosts = False
    if not ssh_successful_to_all_hosts:
        exit(1)
    return True


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
    :type settings: Horovod.run.common.util.settings.Settings
    :return:
    :rtype:
    """

    def _exec_command(command):
        host_output = six.StringIO()
        try:
            exit_code = safe_shell_exec.execute(command,
                                                stdout=host_output,
                                                stderr=host_output)
            if exit_code != 0:
                print(
                    'Launching horovodrun task function was not '
                    'successful:\n{host_output}'
                    .format(host_output=host_output.getvalue()))
                os._exit(exit_code)
        finally:
            host_output.close()
        return exit_code

    if settings.ssh_port:
        ssh_port_arg = '-p {ssh_port}'.format(ssh_port=settings.ssh_port)
    else:
        ssh_port_arg = ''
    args_list = []
    for index in range(len(all_host_names)):
        host_name = all_host_names[index]
        if host_name in local_host_names:
            command = \
                '{python} -m horovod.run.task_fn {index} ' \
                '{driver_addresses} {settings}'\
                .format(python=sys.executable,
                        index=codec.dumps_base64(index),
                        driver_addresses=codec.dumps_base64(driver_addresses),
                        settings=codec.dumps_base64(settings))
        else:
            command = \
                'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} ' \
                '\'{python} -m horovod.run.task_fn {index} {driver_addresses}' \
                ' {settings}\''\
                .format(host=host_name,
                        ssh_port_arg=ssh_port_arg,
                        python=sys.executable,
                        index=codec.dumps_base64(index),
                        driver_addresses=codec.dumps_base64(driver_addresses),
                        settings=codec.dumps_base64(settings))
        args_list.append([command])
    # Each thread will use ssh command to launch the server on one task. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session -- and the the task server --
    # will be bound to the thread. In case, the horovodrun process dies, all
    # the ssh sessions and all the task servers will die as well.
    threads.execute_function_multithreaded(_exec_command,
                                           args_list,
                                           block_until_all_done=False)


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
    :type settings: Horovod.run.common.util.settings.Settings
    :return: example: ['eth0', 'eth1']
    :rtype: list[string]
    """
    # Launch a TCP server called service service on the host running
    # horovodrun.
    driver = driver_service.HorovodRunDriverService(
        settings.num_hosts, settings.key)
    if settings.verbose >= 2:
        print('Launched horovodrun server.')
    # Have all the workers register themselves with the service service.
    _launch_task_servers(all_host_names, local_host_names,
                         driver.addresses(), settings)
    if settings.verbose >= 2:
        print('Attempted to launch horovod task servers.')
    try:
        # wait for all the hosts to register with the service service.
        if settings.verbose >= 2:
            print('Waiting for the hosts to acknowledge.')
        driver.wait_for_initial_registration(settings.timeout)
        tasks = [
            task_service.HorovodRunTaskClient(
                index,
                driver.task_addresses_for_driver(index),
                settings.key,
                settings.verbose) for index in range(
                settings.num_hosts)]
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
            print('Waiting for hosts to perform host-to-host '
                  'interface checking.')
        driver.wait_for_task_to_task_address_updates(settings.timeout)
        if settings.verbose >= 2:
            print('Host-to-host interface checking successful.')
        # Determine a set of common interfaces for task-to-task communication.
        common_intfs = set(driver.task_addresses_for_tasks(0).keys())
        for index in range(1, settings.num_hosts):
            common_intfs.intersection_update(
                driver.task_addresses_for_tasks(index).keys())
        if not common_intfs:
            raise Exception(
                'Unable to find a set of common task-to-task communication '
                'interfaces: %s'
                % [(index, driver.task_addresses_for_tasks(index))
                   for index in range(settings.num_hosts)])
        return common_intfs
    finally:
        driver.shutdown()

def check_build(verbose):
    def get_check(value):
        return 'X' if value else ' '

    output = '''{verbose_newline}\
    Horovod v{version}:

    Available Frameworks:
        [{tensorflow}] TensorFlow
        [{torch}] PyTorch
        [{mxnet}] MXNet

    Available Controllers:
        [{mpi}] MPI
        [{gloo}] Gloo

    Available Tensor Operations:
        [{nccl_ops}] NCCL
        [{ddl_ops}] DDL
        [{mlsl_ops}] MLSL
        [{mpi_ops}] MPI
        [{gloo_ops}] Gloo\
    '''.format(verbose_newline='\n' if verbose else '',
               version=horovod.__version__,
               tensorflow=get_check(extension_available('tensorflow', verbose=verbose)),
               torch=get_check(extension_available('torch', verbose=verbose)),
               mxnet = get_check(extension_available('mxnet', verbose=verbose)),
               mpi=get_check(mpi_built(verbose=verbose)),
               gloo=get_check(gloo_built(verbose=verbose)),
               nccl_ops=get_check(nccl_built(verbose=verbose)),
               ddl_ops=get_check(ddl_built(verbose=verbose)),
               mpi_ops=get_check(mpi_built(verbose=verbose)),
               mlsl_ops=get_check(mlsl_built(verbose=verbose)),
               gloo_ops=get_check(gloo_built(verbose=verbose)))
    print(textwrap.dedent(output))
    os._exit(0)


def make_check_build_action(np_arg):
    class CheckBuildAction(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            # If -cb is specified, make -np optional
            np_arg.required = False
            args.check_build = True

    return CheckBuildAction


def parse_args():
    parser = argparse.ArgumentParser(description='Horovod Runner')

    parser.add_argument('-v', '--version', action='version', version=horovod.__version__,
                        help='Shows Horovod version.')

    np_arg = parser.add_argument('-np', '--num-proc', action='store', dest='np',
                                 type=int, required=True,
                                 help='Total number of training processes.')

    parser.add_argument('-cb', '--check-build', action=make_check_build_action(np_arg), nargs=0,
                        help='Shows which frameworks and libraries have been built into Horovod.')

    parser.add_argument('-p', '--ssh-port', action='store', dest='ssh_port',
                        type=int, help='SSH port on all the hosts.')

    parser.add_argument('--disable-cache', action='store_true',
                        dest='disable_cache',
                        help='If the flag is not set, horovodrun will perform '
                             'the initialization checks only once every 60 '
                             'minutes -- if the checks successfully pass. '
                             'Otherwise, all the checks will run every time '
                             'horovodrun is called.')

    parser.add_argument('--start-timeout', action='store',
                        dest='start_timeout', type=int,
                        help='Horovodrun has to perform all the checks and '
                             'start the processes before the specified '
                             'timeout. The default value is 30 seconds. '
                             'Alternatively, The environment variable '
                             'HOROVOD_START_TIMEOUT can also be used to '
                             'specify the initialization timeout.')

    parser.add_argument('--verbose', action='store_true',
                        dest='verbose',
                        help='If this flag is set, extra messages will '
                             'printed.')

    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Command to be executed.')

    group_hosts_parent = parser.add_argument_group('host arguments')
    group_hosts = group_hosts_parent.add_mutually_exclusive_group()
    group_hosts.add_argument('-H', '--hosts', action='store', dest='hosts',
                             help='List of host names and the number of available slots '
                                  'for running processes on each, of the form: <hostname>:<slots> '
                                  '(e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run on host1, '
                                  '4 on host2, and 1 on host3). If not specified, defaults to using '
                                  'localhost:<np>')
    group_hosts.add_argument('-hostfile', '--hostfile', action='store', dest='hostfile',
                             help='Path to a host file containing the list of host names and the number of '
                                  'available slots. Each line of the file must be of the form: '
                                  '<hostname> slots=<slots>')

    group_controller_parent = parser.add_argument_group('controller arguments')
    group_controller = group_controller_parent.add_mutually_exclusive_group()
    group_controller.add_argument('--gloo', action='store_true', dest='use_gloo',
                                  help='Run Horovod using the Gloo controller. This will '
                                       'be the default if Horovod was not built with MPI support.')
    group_controller.add_argument('--mpi', action='store_true', dest='use_mpi',
                                  help='Run Horovod using the MPI controller. This will '
                                       'be the default if Horovod was built with MPI support.')

    return parser.parse_args()


def parse_host_files(filename):
    hosts = []
    for line in open(filename):
        hostname = line.split()[0]
        slots = line.split('=')[1]
        hosts.append('{name}:{slots}'.format(name=hostname, slots=slots))

    return ','.join(hosts)


def run():
    args = parse_args()

    if args.check_build:
        check_build(args.verbose)

    # if hosts are not specified, either parse from hostfile, or default as
    # localhost
    if not args.hosts:
        if args.hostfile:
            args.hosts = parse_host_files(args.hostfile)
        else:
            # Set hosts to localhost if not specified
            args.hosts = 'localhost:{np}'.format(np=args.np)

    host_list = args.hosts.split(',')
    all_host_names = []
    pattern = re.compile(r'^[\w.-]+:\d+$')
    for host in host_list:
        if not pattern.match(host.strip()):
            raise ValueError('Invalid host input, please make sure it has '
                             'format as : worker-0:2,worker-1:2.')
        all_host_names.append(host.strip().split(':')[0])

    # horovodrun has to finish all the checks before this timeout runs out.
    if args.start_timeout:
        start_timeout = args.start_timeout
    else:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_START_TIMEOUT', '30'))

    tmout = timeout.Timeout(start_timeout,
                            message='Timed out waiting for {activity}. Please '
                                    'check connectivity between servers. You '
                                    'may need to increase the --start-timeout '
                                    'parameter if you have too many servers.')
    settings = hvd_settings.Settings(verbose=2 if args.verbose else 0,
                                     ssh_port=args.ssh_port,
                                     key=secret.make_secret_key(),
                                     timeout=tmout,
                                     num_hosts=len(all_host_names),
                                     num_proc=args.np,
                                     hosts=args.hosts,
                                     command=args.command)

    # This cache stores the results of checks performed by horovodrun
    # during the initialization step. It can be disabled by setting
    # --disable-cache flag.
    fn_cache = None
    if not args.disable_cache:
        params = ''
        if args.np:
            params += str(args.np) + ' '
        if args.hosts:
            params += str(args.hosts) + ' '
        if args.ssh_port:
            params += str(args.ssh_port)
        parameters_hash = hashlib.md5(params.encode('utf-8')).hexdigest()
        fn_cache = cache.Cache(CACHE_FOLDER, CACHE_STALENESS_THRESHOLD_MINUTES,
                               parameters_hash)

    if settings.verbose >= 2:
        print('Filtering local host names.')
    remote_host_names = network.filter_local_addresses(all_host_names)
    if settings.verbose >= 2:
        print('Remote host found: ' + ' '.join(remote_host_names))

    if len(remote_host_names) > 0:
        if settings.verbose >= 2:
            print('Checking ssh on all remote hosts.')
        # Check if we can ssh into all remote hosts successfully.
        _check_all_hosts_ssh_successful(remote_host_names, args.ssh_port,
                                        fn_cache=fn_cache)
        if settings.verbose >= 2:
            print('SSH was successful into all the remote hosts.')

    if len(remote_host_names) > 0:
        if settings.verbose >= 2:
            print('Testing interfaces on all the hosts.')

        local_host_names = set(all_host_names) - set(remote_host_names)
        # Find the set of common, routed interfaces on all the hosts (remote
        # and local) and specify it in the args to be used by NCCL. It is
        # expected that the following function will find at least one interface
        # otherwise, it will raise an exception.
        common_intfs = _driver_fn(all_host_names, local_host_names,
                                  settings, fn_cache=fn_cache)

        if settings.verbose >= 2:
            print('Interfaces on all the hosts were successfully checked.')
            print('Common interface found: ' + ' '.join(common_intfs))

    else:
        if settings.verbose >= 2:
            print('All hosts are local, finding the interfaces '
                  'with address 127.0.0.1')
        # If all the given hosts are local, find the interfaces with address
        # 127.0.0.1
        common_intfs = set()
        for iface, addrs in net_if_addrs().items():
            for addr in addrs:
                if addr.family == AF_INET and addr.address == '127.0.0.1':
                    common_intfs.add(iface)
                    break

        if len(common_intfs) == 0:
            raise ValueError('No interface is found for address 127.0.0.1.')

        if settings.verbose >= 2:
            print('Local interface found ' + ' '.join(common_intfs))

    if args.use_gloo:
        if not gloo_built(verbose=(settings.verbose >= 2)):
            raise ValueError('Gloo support has not been built.  If this is not expected, ensure CMake is installed '
                             'and reinstall Horovod with HOROVOD_WITH_GLOO=1 to debug the build error.')
        gloo_run(settings, remote_host_names, common_intfs)
    elif args.use_mpi:
        if not mpi_built(verbose=(settings.verbose >= 2)):
            raise ValueError('MPI support has not been built.  If this is not expected, ensure MPI is installed '
                             'and reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error.')
        mpi_run(settings, common_intfs)
    else:
        if mpi_built(verbose=(settings.verbose >= 2)):
            mpi_run(settings, common_intfs)
        elif gloo_built(verbose=(settings.verbose >= 2)):
            gloo_run(settings, remote_host_names, common_intfs)
        else:
            raise ValueError('Neither MPI nor Gloo support has been built. Try reinstalling Horovod ensuring that '
                             'either MPI is installed (MPI) or CMake is installed (Gloo).')


if __name__ == '__main__':
    run()
