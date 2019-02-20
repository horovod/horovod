__version__ = '0.16.0'

import argparse
import os
import socket
import six
import sys
import psutil

import horovod.spark.util.safe_shell_exec as safe_shell_exec
import horovod.run.util as util
import horovod.spark.util.codec as codec

# Names of the modules for server, client and executer script
SERVER_MODULE = 'horovod.run.server'
CLIENT_MODULE = 'horovod.run.client'

CACHE_FOLDER = os.path.join(os.path.expanduser('~'), '.horovod')

# Define constant parameters
MAX_CONCURRENT_EXECUTIONS = 1000
CACHE_STALENESS_THRESHOLD_MINUTES = 60

cache = util.Cache(CACHE_FOLDER, CACHE_STALENESS_THRESHOLD_MINUTES)


def use_cache(func):
    def func_wrapper(*args, **kwargs):
        cached_result = cache.get(
            (func.__name__, tuple(args[0]), frozenset(kwargs.items())))
        if cached_result is not None:
            return cached_result
        else:
            results = func(*args, **kwargs)
            if results is not None:
                cache.put(
                    (func.__name__, tuple(args[0]), frozenset(kwargs.items())),
                    results)
        return results

    return func_wrapper


@use_cache
def all_hosts_ssh_successful(host_addresses):
    def exec_command(command):
        with open(os.devnull, 'wrb', 0) as devnull:
            exit_code = safe_shell_exec.execute(command, stdout=devnull)
        return exit_code

    ssh_command_format = 'ssh -o StrictHostKeyChecking=no {host} ls'
    args_list = [[ssh_command_format.format(host=host_address)] for host_address
                 in host_addresses]

    ssh_exit_codes = \
        util.execute_function_multithreaded(exec_command,
                                            args_list,
                                            MAX_CONCURRENT_EXECUTIONS)

    unsuccessful_ssh_hosts = []
    for index, ssh_status in ssh_exit_codes.items():
        if ssh_status != 0:
            unsuccessful_ssh_hosts.append(host_addresses[index])

    if len(unsuccessful_ssh_hosts) > 0:
        print("SSH not successful for: {hosts}".format(
            hosts=', '.join(unsuccessful_ssh_hosts)))
        exit(0)

    return True


@use_cache
def gather_host_interfaces(host_addresses):
    def interfaces():
        result = {}
        for intf, intf_addresses in psutil.net_if_addrs().items():
            for addr in intf_addresses:
                if addr.family == socket.AF_INET:
                    if intf not in result:
                        result[intf] = []
                    result[intf].append(addr.address)
        return result

    args_list = [[[], host_address] for host_address in
                 host_addresses]
    results = util.execute_function_over_ssh_multithreaded(interfaces,
                                                           args_list,
                                                           MAX_CONCURRENT_EXECUTIONS)
    return results


@use_cache
def circular_check_interfaces(host_addresses, host_interfaces):
    def perform_check_on_host(host_address,
                              target_address,
                              target_interfaces):
        hvd_secret_bs64 = codec.dumps_base64('salam')
        ssh_target = util.Ssh(target_address)
        ssh_host = util.Ssh(host_address)
        try:
            ssh_target.openShell()
            target_address_bs64 = codec.dumps_base64(target_address)
            command = '_HOROVOD_SECRET_KEY={hvd_secred} ' \
                      'python -m {server_module} {server_name_bs64}'.format(
                server_module=SERVER_MODULE,
                hvd_secred=hvd_secret_bs64,
                server_name_bs64=target_address_bs64)

            ssh_target.sendShell(command)
            response = ssh_target.recv()

            port = \
                util.check_success_msg_and_extract_result(response,
                                                          'SERVER LAUNCH SUCCESSFUL',
                                                          'PORT IS: (.+?) EOM')
            if not port:
                raise RuntimeError('Could not launch the server on '
                                   'host {host}'.format(host=target_address))

            ssh_host.openShell()
            intfc_addr_to_probe = {}
            for intfc, if_addresses in six.iteritems(target_interfaces):
                for addr in if_addresses:
                    existing_intfc_address = intfc_addr_to_probe.get(intfc, [])
                    existing_intfc_address.append((addr, int(port)))
                    intfc_addr_to_probe[intfc] = existing_intfc_address

            command = '_HOROVOD_SECRET_KEY={hvd_secred} ' \
                      '{python} -m {client_module} {server_name_bs64} ' \
                      '{addresses_bs64}'.format(
                client_module=CLIENT_MODULE, python=sys.executable,
                hvd_secred=hvd_secret_bs64,
                server_name_bs64=target_address_bs64,
                addresses_bs64=codec.dumps_base64(intfc_addr_to_probe))

            ssh_host.sendShell(command)
            response = ssh_host.recv()

            addresses = util.check_success_msg_and_extract_result(response,
                                                                  'CLIENT LAUNCH SUCCESSFUL',
                                                                  'SUCCESSFUL INTERFACE ADDRESSES (.+?) EOM')
            functional_target_interfaces = []
            if addresses:
                functional_target_interfaces = addresses.keys()
            return functional_target_interfaces

        finally:
            ssh_target.closeConnection()
            ssh_host.closeConnection()

    num_hosts = len(host_addresses)
    args_list = [[host_addresses[i],
                  host_addresses[(i + 1) % num_hosts],
                  host_interfaces[(i + 1) % num_hosts]] for i in
                 range(num_hosts)]

    result_queue = util.execute_function_multithreaded(perform_check_on_host,
                                                       args_list,
                                                       MAX_CONCURRENT_EXECUTIONS)

    valid_common_interfaces = set(result_queue[0])
    for i in range(1, num_hosts):
        valid_common_interfaces = valid_common_interfaces.intersection(
            set(result_queue[i]))

    return list(valid_common_interfaces)


def parse_args():
    parser = argparse.ArgumentParser(description='Horovod Runner')

    parser.add_argument("main_script", help="Main script path.")

    parser.add_argument('-np', '--num-proc', action="store", dest="np",
                        type=int,
                        help="Number of processes which should be equal to "
                             "the total number of available GPUs.")

    parser.add_argument('-H', '--host', action="store", dest="host",
                        help="To specify the list of hosts on which "
                             "to invoke processes. Takes a comma-delimited "
                             "list of hosts.")

    parser.add_argument('--disable-cache', action="store_true",
                        dest="disable_cache",
                        help="If flag is set, horovod run will cache the "
                             "resutls and use them.")

    return parser.parse_args()


def main():
    hvd_arg = parse_args()

    host_addresses = [x for x in
                      [y.split(':')[0] for y in hvd_arg.host.split(',')]]

    all_hosts_ssh_successful(host_addresses)
    host_interfaces = gather_host_interfaces(host_addresses)

    common_intfs = set(host_interfaces[0].keys())
    for index in range(1, len(host_addresses)):
        common_intfs.intersection_update(host_interfaces[index].keys())
    if not common_intfs:
        raise Exception(
            'Unable to find a set of common host-to-host communication '
            'interfaces: %s'
            % [(index, interfaces) for index, interfaces in
               host_interfaces.items()])
    print('Finding common set of routerd NICs on hosts.')

    common_intfs_to_use = circular_check_interfaces(host_addresses,
                                                    host_interfaces)

    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '-np {num_proc} -H {hosts} '
        '-bind-to none -map-by slot '
        '-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include {common_intfs} '
        '-x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME={common_intfs} '
        '{env} '  # expect a lot of environment variables
        '{python} {code}'
            .format(num_proc=hvd_arg.np,
                    hosts=hvd_arg.host,
                    common_intfs=','.join(common_intfs_to_use),
                    env=' '.join('-x %s' % key for key in os.environ.keys()),
                    python=sys.executable,
                    code=hvd_arg.main_script)
    )

    safe_shell_exec.execute(mpirun_command)


if __name__ == "__main__":
    main()
>>>>>>> Added Horovodrun
