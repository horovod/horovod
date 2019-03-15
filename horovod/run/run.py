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

import argparse
import os
import shlex
import sys

import six

import horovod
from horovod.run.common.util import codec, safe_shell_exec, timeout, secret
from horovod.run.driver import driver_service
from horovod.run.task import task_service
from horovod.run.util import cache, threads

# Cached information of horovodrun functions be stored in this directory
CACHE_FOLDER = os.path.join(os.path.expanduser('~'), '.horovod')

# Cache entries will be stale if they are older than this number of minutes
CACHE_STALENESS_THRESHOLD_MINUTES = 60

# Maximum number of concurrent worker threads to prevent from over saturation.
MAX_CONCURRENT_EXECUTIONS = 20

# Number of retries for sshing into the hosts
SSH_RETRIES = 5


def use_cache():
    """
    If decorates a function, and if cache_disabled is set, it will store the
        output of the function if it not None. If a function output is None, the
    execution will not be cached.
    :return:
    """

    def wrap(func):
        def wrap_f(*args, **kwargs):
            fn_cache = kwargs.pop('fn_cache')
            if fn_cache is None:
                results = func(*args, **kwargs)
            else:
                cached_result = fn_cache.get(
                    (func.__name__, tuple(args[0]), frozenset(kwargs.items())))
                if cached_result is not None:
                    return cached_result
                else:
                    results = func(*args, **kwargs)
                    if results is not None:
                        fn_cache.put(
                            (func.__name__, tuple(args[0]),
                             frozenset(kwargs.items())),
                            results)
            return results

        return wrap_f

    return wrap


@use_cache()
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
        msg_stdout = ""
        msg_stderr = ""

        # Try ssh 5 times
        for i in range(SSH_RETRIES):
            stdout_w = six.StringIO()
            stderr_w = six.StringIO()
            try:
                exit_code = safe_shell_exec.execute(command,
                                                    stdout=stdout_w,
                                                    stderr=stderr_w)
                if exit_code == 0:
                    break
                else:
                    msg_stderr = stderr_w.getvalue()
                    msg_stdout = stdout_w.getvalue()
            finally:
                stdout_w.close()
                stderr_w.close()
        return exit_code, msg_stderr, msg_stdout

    if ssh_port:
        ssh_port_arg = "-p {ssh_port}".format(ssh_port=ssh_port)
    else:
        ssh_port_arg = ""

    ssh_command_format = 'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} date'
    args_list = [[ssh_command_format.format(host=host_address,
                                            ssh_port_arg=ssh_port_arg)]
                 for host_address in host_addresses]
    ssh_exit_codes = \
        threads.execute_function_multithreaded(exec_command,
                                               args_list)

    ssh_successful_to_all_hosts = True
    for index, ssh_status in six.iteritems(ssh_exit_codes):
        exit_code, stderr, stdout = ssh_status[0], ssh_status[1], ssh_status[2]
        if exit_code != 0:
            print("ssh not successful for host {host}.".format(
                host=host_addresses[index]))
            print("stderr:\n{stderr}\nstdout:\n{stdout}".format(stdout=stdout,
                                                                stderr=stderr))
            ssh_successful_to_all_hosts = False
    if not ssh_successful_to_all_hosts:
        exit(1)
    return True


def _launch_task_servers(host_addresses, driver_addresses, num_hosts, tmout,
                         key, ssh_port=None):
    """
    executes the task server and service client task for registration on the
    hosts.
    :param host_addresses: list of addresses. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type host_addresses: list(string)
    :param driver_addresses: map of interfaces and their address and port for
    the service. For example:
        {
            'lo': [('127.0.0.1', 34588)],
            'docker0': [('172.122.10.1', 34588)],
            'eth0': [('11.111.33.73', 34588)]
        }
    :type driver_addresses: map
    :param num_hosts:
    :type num_hosts: int
    :param tmout:
    :type tmout: horovod.spark.util.timeout.Timeout
    :return:
    :rtype:
    """

    def _exec_command(command):
        stdout_w = six.StringIO()
        stderr_w = six.StringIO()
        try:
            exit_code = safe_shell_exec.execute(command,
                                                stdout=stdout_w,
                                                stderr=stderr_w)
            if exit_code != 0:
                print(
                    "Launching horovodrun task function was not successful.")
                print("stderr from host:\n {stderr}".format(
                    stderr=stderr_w.getvalue()))
                print("stdout from host:\n {stdout}".format(
                    stdout=stdout_w.getvalue()))
                os._exit(exit_code)
        finally:
            stdout_w.close()
            stderr_w.close()
        return exit_code

    if ssh_port:
        ssh_port_arg = "-p {ssh_port}".format(ssh_port=ssh_port)
    else:
        ssh_port_arg = ""

    command_format = \
        'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} ' \
        '\'{python} -m horovod.run.horovod_task_fn {index} ' \
        '{driver_addresses} {num_hosts} {timeout} {key}\''
    args_list = [
        [command_format.format(
            host=host_addresses[index],
            ssh_port_arg=ssh_port_arg,
            python=sys.executable,
            index=codec.dumps_base64(index),
            driver_addresses=codec.dumps_base64(driver_addresses),
            num_hosts=codec.dumps_base64(num_hosts),
            timeout=codec.dumps_base64(tmout),
            key=codec.dumps_base64(key)
        )]
        for index in range(num_hosts)
    ]
    # Each thread will use ssh command to launch the server on one task. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session -- and the the task server --
    # will be bound to the thread. In case, the horovodrun process dies, all
    # the ssh sessions and all the task servers will die as well.
    threads.execute_function_multithreaded(_exec_command,
                                           args_list,
                                           block_until_all_done=False)


@use_cache()
def _driver_fn(key, host_addresses, tmout, ssh_port=None,
               verbose=False):
    """
    launches the service service, launches the task service on each worker and
    have them register with the service service. Each worker probes all the
    interfaces of the worker index + 1 (in a ring manner) and only keeps the
    routed interfaces. Function returns the intersection of the set of all the
    routed interfaces on all the workers.
    :param key:
    :type key: string
    :param host_addresses: list of addresses. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
    :type host_addresses: list(string)
    :param tmout:
    :type tmout: horovod.spark.util.timeout.Timeout
    :param verbose:
    :type verbose: bool
    :return: example: ['eth0', 'eth1']
    :rtype: list[string]
    """
    num_hosts = len(host_addresses)
    # Launch a TCP server called service service on the host running horovodrun.
    driver = driver_service.HorovodRunDriverService(num_hosts, key)
    if verbose:
        print("Launched horovodrun server.")
    # Have all the workers register themselves with the service service.
    _launch_task_servers(host_addresses, driver.addresses(), num_hosts, tmout,
                         key, ssh_port)
    if verbose:
        print("Attempted to launch horovod task servers.")
    try:
        # wait for all the hosts to register with the service service.
        if verbose:
            print("Waiting for the hosts to acknowledge.")
        driver.wait_for_initial_registration(tmout)
        tasks = [task_service.HorovodRunTaskClient(index,
                                                   driver.task_addresses_for_driver(index),
                                                   key)
                 for index in range(num_hosts)]
        # Notify all the drivers that the initial registration is complete.
        for task in tasks:
            task.notify_initial_registration_complete()
        if verbose:
            print("Notified all the hosts that the registration is complete.")
        # Each worker should probe the interfaces of the next worker in a ring
        # manner and filter only the routed ones -- it should filter out
        # interfaces that are not really connected to any external networks
        # such as lo0 with address 127.0.0.1.
        if verbose:
            print("Waiting for hosts to perform host-to-host "
                  "interface checking.")
        driver.wait_for_task_to_task_address_updates(tmout)
        if verbose:
            print("Host-to-host interface checking successful.")
        # Determine a set of common interfaces for task-to-task communication.
        common_intfs = set(driver.task_addresses_for_tasks(0).keys())
        for index in range(1, num_hosts):
            common_intfs.intersection_update(
                driver.task_addresses_for_tasks(index).keys())
        if not common_intfs:
            raise Exception(
                'Unable to find a set of common task-to-task communication '
                'interfaces: %s'
                % [(index, driver.task_addresses_for_tasks(index))
                   for index in range(num_hosts)])
        return common_intfs
    finally:
        driver.shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description='Horovod Runner')

    parser.add_argument('-v', '--version', action="store_true", dest="version",
                        help="Show Horovod version.")

    parser.add_argument('-np', '--num-proc', action="store", dest="np",
                        type=int,
                        help="Number of processes which should be equal to "
                             "the total number of available GPUs.")

    parser.add_argument('-p', '--ssh-port', action="store", dest="ssh_port",
                        type=int,
                        help="SSH port on all the workers.")

    parser.add_argument('-H', '--host', action="store", dest="host",
                        help="To specify the list of hosts on which "
                             "to invoke processes. Takes a comma-delimited "
                             "list of hosts.")

    parser.add_argument('--disable-cache', action="store_true",
                        dest="disable_cache",
                        help="If flag is set, horovod run will not cache the "
                             "initial checks and execute them every time.")
    parser.add_argument('--horovod-start-timeout', action="store",
                        dest="start_timeout",
                        help="Horovodrun has to perform all the checks and and "
                             "start before specified timeout.")

    parser.add_argument('--verbose', action="store_true",
                        dest="verbose",
                        help="If this flag is set, extra messages will printed.")

    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help="Command to be executed.")

    return parser.parse_args()


def run():
    args = parse_args()

    if args.version:
        print(horovod.__version__)
        exit(0)

    if args.host:
        host_addresses = [x for x in
                          [y.split(':')[0] for y in args.host.split(',')]]
    else:
        host_addresses = []

    # This cache stores the results of checks performed by horovodrun
    # during the initialization step. It can be disabled by setting
    # --disable-cache flag.
    fn_cache = None
    if not args.disable_cache:
        parameters_hash = str(hash(str(args.host) + str(args.np)))
        fn_cache = cache.Cache(CACHE_FOLDER, CACHE_STALENESS_THRESHOLD_MINUTES,
                               parameters_hash)

    # horovodrun has to finish all the checks before this timeout runs out.
    if args.start_timeout:
        start_timeout = args.start_timeout
    else:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_START_TIMEOUT', '600'))
    tmout = timeout.Timeout(start_timeout)

    key = secret.make_secret_key()
    if args.host:
        if args.verbose:
            print("Checking ssh on all hosts.")
        # Check if we can ssh into all hosts successfully. This is required for
        _check_all_hosts_ssh_successful(host_addresses, args.ssh_port,
                                        fn_cache=fn_cache)
        if args.verbose:
            print("SSH was successful into all the hosts.")

        if args.verbose:
            print("Testing interfaces on all the hosts.")
        # Find the set of common, routed interfaces on all the hosts and specify
        # it to be used by NCCL.
        common_intfs = _driver_fn(key, host_addresses, tmout,
                                  args.ssh_port,
                                  args.verbose,
                                  fn_cache=fn_cache)
        if args.verbose:
            print("Interfaces on all the hosts were successfully checked.")

        hosts_arg = "-H {hosts}".format(hosts=args.host)
        tcp_intf_arg = "-mca btl_tcp_if_include {common_intfs}".format(
            common_intfs=','.join(common_intfs))
        nccl_socket_intf_arg = "-x NCCL_SOCKET_IFNAME={common_intfs}".format(
            common_intfs=','.join(common_intfs))
    else:
        # if user does not specify any hosts, mpirun by default uses local host.
        # There is no need to specify localhost.
        hosts_arg = ""
        tcp_intf_arg = ""
        nccl_socket_intf_arg = ""

    # Pass all the env variables to the mpirun command.
    env = os.environ.copy()

    # Pass secret key through the environment variables.
    env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(key)

    if args.ssh_port:
        ssh_port_arg = "-mca plm_rsh_args \"-p {ssh_port}\"".format(
            ssh_port=args.ssh_port)
    else:
        ssh_port_arg = ""

    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '-np {num_proc} {hosts_arg} '
        '-bind-to none -map-by slot '
        '-mca pml ob1 -mca btl ^openib '
        '{ssh_port_arg} '
        '{tcp_intf_arg} '
        '-x NCCL_DEBUG=INFO '
        '{nccl_socket_intf_arg} '
        '{env} {command}'  # expect a lot of environment variables
            .format(num_proc=args.np,
                    hosts_arg=hosts_arg,
                    tcp_intf_arg=tcp_intf_arg,
                    nccl_socket_intf_arg=nccl_socket_intf_arg,
                    ssh_port_arg=ssh_port_arg,
                    env=' '.join('-x %s' % key for key in env.keys()),
                    command=' '.join(shlex.quote(par) for par in args.command))
    )

    if args.verbose:
        print(mpirun_command)
    # Execute the mpirun command.
    exit_code = safe_shell_exec.execute(mpirun_command, env)
    if exit_code != 0:
        raise Exception(
            'mpirun exited with code %d, see the error above.' % exit_code)


if __name__ == "__main__":
    run()
