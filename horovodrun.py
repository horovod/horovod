import argparse
import subprocess
import threading
import Queue
import os
import time
import cloudpickle
import tempfile
import random
import string

import horovod.spark.util.safe_shell_exec as safe_shell_exec

tmp_exec_script_path = tempfile.NamedTemporaryFile(delete=False)
remote_res_file = 'result.bin'

# Define constant parameters
SSH_TIMEOUT = 20


class RemoteOperationError(Exception):
    pass


def host_ssh_check(host, status_queue, timeout):
    with open(os.devnull, 'w+r+b', 0) as devnull:
        child = subprocess.Popen(
            ['ssh',
             '-o', 'StrictHostKeyChecking=yes',
             '%s' % host, 'ls'],
            stdout=devnull, stderr=devnull, stdin=devnull)

        while child.poll() is None and timeout > 0:
            time.sleep(1)
            timeout -= 1

        rc = child.poll()
        if rc is None:
            child.kill()

        if rc == 0:
            ssh_success = True
        else:
            ssh_success = False
        status_queue.put((host, ssh_success))


def multithreaded_run(fn, arg_list):
    threads = []
    for arg in arg_list:
        thread = threading.Thread(target=fn,
                                  args=arg)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()


def all_hosts_ssh_check(host_addresses, timeout=SSH_TIMEOUT):
    ssh_status_queue = Queue.Queue()
    arg_list = [(host, ssh_status_queue, timeout) for host in host_addresses]
    multithreaded_run(host_ssh_check, arg_list)

    unsuccessful_ssh_hosts = []
    while not ssh_status_queue.empty():
        host_address, ssh_status = ssh_status_queue.get()
        if not ssh_status:
            unsuccessful_ssh_hosts.append(host_address)

    if len(unsuccessful_ssh_hosts) > 0:
        print("SSH was not successful for the following hosts: %s" % ', '.join(
            unsuccessful_ssh_hosts))
        exit(1)
    else:
        return True


def scp_local2host(local_path, host, remote_path):
    scp_command = \
        'scp {file} {host}:{path}'.format(file=local_path, host=host,
                                          path=remote_path)
    exit_code = safe_shell_exec.execute(scp_command)
    if exit_code != 0:
        raise RemoteOperationError(
            'scp not successful for local path:{local_path} and remote path:{remote_path}'.format(
                local_path=local_path, remote_path=remote_path))
    return remote_path


def scp_host2local(remote_path, host, local_path):
    scp_command = \
        'scp {host}:{remote_path} {local_path}'.format(local_path=local_path,
                                                       host=host,
                                                       remote_path=remote_path)
    exit_code = safe_shell_exec.execute(scp_command)
    if exit_code != 0:
        raise RemoteOperationError(
            'scp not successful for remote path:{remote_path} and local path:{local_path}'.format(
                remote_path=remote_path, local_path=local_path))
    return local_path


def find_valid_remote_folder_name_and_create(host, base=None):
    while True:
        random_dir_name = ''.join(
            random.choice(string.ascii_letters + string.digits) for _ in
            range(15))
        if base:
            random_dir_name = os.path.join(base, random_dir_name)
        exit_code = safe_shell_exec.execute(
            'ssh {host} mkdir {path}'.format(host=host,
                                             path=random_dir_name))
        if exit_code == 0:
            break
        elif exit_code == 1:
            continue
        else:
            raise RemoteOperationError(
                'Was not able to create a valid folder on remote host: {host}'.format(
                    host=host))

    return random_dir_name


def test_fun(num1, num2):
    return num1+1, num2+1


def run_function_over_ssh(fn, args, host):
    tmp_fn_file = tempfile.TemporaryFile()
    with open(tmp_fn_file.name, 'w') as pickled_fn:
        cloudpickle.dumps(fn, pickled_fn)

    tmp_args_file = tempfile.NamedTemporaryFile(delete=False)
    with open(tmp_args_file, 'w') as picked_args:
        cloudpickle.dumps(args, picked_args)

    remote_folder_path = find_valid_remote_folder_name_and_create(host)
    remote_fn_file_path = scp_local2host(tmp_fn_file, host,
                                         remote_path=remote_folder_path)
    remote_args_file_path = scp_local2host(tmp_args_file, host,
                                           remote_path=remote_folder_path)
    remote_script_file_path = scp_local2host(tmp_exec_script_path,
                                             remote_path=remote_folder_path)

    command = 'ssh {host} python {remote_script_file_path} {remote_fn_file_path} {remote_args_file_path}'.format(
        host=host,
        remote_script_file_path=remote_script_file_path,
        remote_fn_file_path=remote_fn_file_path,
        remote_args_file_path=remote_args_file_path)

    with open(os.devnull, 'w+r+b', 0) as devnull:
        exit_code = safe_shell_exec.execute(command, stdout=devnull,
                                            stderr=devnull)
        if exit_code != 0:
            RemoteOperationError(
                "remote execution was unsuccessful on host: {host} for command: {command}".format(
                    host=host, command=command))
            exit(exit_code)

    local_res_file = scp_host2local(
        remote_path=os.path.join(remote_folder_path, remote_res_file),
        host=host, local_path='.')

    result = cloudpickle.loads(local_res_file)
    return result


def main():
    parser = argparse.ArgumentParser(description='Horovod Runner')

    parser.add_argument("program", help="Program file")

    parser.add_argument('-np', '--num-proc', action="store", dest="np",
                        type=int,
                        help="Number of processes which should be equal to the total number of available GPUs.")

    parser.add_argument('-H', '--host', action="store", dest="host",
                        help="To specify the list of hosts on which to invoke processes. Takes a comma-delimited list of hosts.")

    args = parser.parse_args()

    host_addresses = [x for x in
                      [y.split(':')[0] for y in args.host.split(',')]]

    all_hosts_ssh_check(host_addresses)

    result = run_function_over_ssh(test_fun, (1,2))
    print result


if __name__ == "__main__":
    remote_exec_script_content = """
import cloudpickle
import sys


def execute_function(fn_file_path, args_file_path):
    with open(fn_file_path, 'r') as pickled_fn:
        fn = cloudpickle.loads(pickled_fn)

    with open(args_file_path, 'r') as picked_args:
        args = cloudpickle.loads(picked_args)

    res = fn(args)
    with open('{remote_res_file}') as result_file:
        cloudpickle.dumps(res, result_file)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Usage: %s <pickled function obj path> <pickled args obj path>' %
              sys.argv[0])
        sys.exit(1)

    execute_function(sys.argv[1], sys.argv[2])
""".format(remote_res_file=remote_res_file)

    with open(tmp_exec_script_path, 'w') as f:
        f.writelines(remote_exec_script_content)

    main()

#
# # Determine a set of common interfaces for task-to-task communication.
# common_intfs = set(driver.task_addresses_for_tasks(0).keys())
# for index in range(1, num_proc):
#     common_intfs.intersection_update(driver.task_addresses_for_tasks(index).keys())
# if not common_intfs:
#     raise Exception('Unable to find a set of common task-to-task communication interfaces: %s'
#                     % [(index, driver.task_addresses_for_tasks(index)) for index in range(num_proc)])
#
#
#
# mpirun_command = (
#     'mpirun --allow-run-as-root --tag-output '
#     '-np {num_proc} -H {hosts} '
#     '-bind-to none -map-by slot '
#     '-mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include {common_intfs} '
#     '-x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME={common_intfs} '
#     '{env} '  # expect a lot of environment variables
#     '-mca plm_rsh_agent "{python} -m horovod.spark.driver.mpirun_rsh {encoded_driver_addresses}" '
#     '{python} -m horovod.spark.task.mpirun_exec_fn {encoded_driver_addresses} '
#         .format(num_proc=argparse.np,
#                 hosts=','.join('%s:%d' % (
#                 host_hash, len(driver.task_host_hash_indices()[host_hash]))
#                                for host_hash in host_hashes),
#                 common_intfs=','.join(common_intfs),
#                 env=' '.join('-x %s' % key for key in env.keys()),
#                 python=sys.executable,
#                 encoded_driver_addresses=codec.dumps_base64(
#                     driver.addresses())))
