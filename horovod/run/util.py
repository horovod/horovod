import paramiko
import time
import socket
import os
import Queue
import re
import sys
import datetime
import cloudpickle
import errno
import threading

import horovod.spark.util.codec as codec

REMOTE_EXECUTER_SCRIPT = 'horovod.run.exec_script'


class Ssh:
    shell = None
    client = None
    transport = None

    def __init__(self, hostname):

        self.client = paramiko.client.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
        try:
            self.client.connect(hostname, look_for_keys=False)
        except socket.gaierror:
            config = paramiko.SSHConfig()
            config.parse(
                open(os.path.join(os.path.expanduser('~'), '.ssh/config')))
            options = config.lookup(hostname)
            params = {}
            params['hostname'] = options['hostname']
            params['look_for_keys'] = False
            if 'port' in options.keys():
                params['port'] = options['port']
            self.client.connect(**params)

        self.transport = paramiko.Transport((options['hostname'], 22))
        self.transport.connect()

    def closeConnection(self):
        if (self.client != None):
            self.client.close()
            self.transport.close()

    def openShell(self):
        self.shell = self.client.invoke_shell()

    def sendShell(self, command):
        if (self.shell):
            self.shell.send(command + "\n")
            time.sleep(1)
        else:
            print("Shell not opened.")

    def recv(self):
        while self.shell != None and not self.shell.recv_ready():
            pass

        if self.shell != None and self.shell.recv_ready():
            alldata = self.shell.recv(1024)
            while self.shell.recv_ready():
                alldata += self.shell.recv(1024)
                time.sleep(0.01)
            strdata = str(alldata).replace('\r', '')
        return strdata


class Cache(object):
    def __init__(self, cache_folder, cache_staleness_threshold_in_minutes):

        self._cache_file = os.path.join(cache_folder, 'cache.txt')
        try:
            os.makedirs(cache_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        if not os.path.isfile(self._cache_file):
            with open(self._cache_file, 'w') as cf:
                cloudpickle.dump({}, cf)

        with open(self._cache_file, 'r') as cf:
            try:
                self._content = cloudpickle.load(cf)
            except Exception as e:
                print(
                    'There is an error with reading cache file. You '
                    'can delete the corrupt file: {cache_file}.'.format(
                        cache_file=self._cache_file))
                exit(1)

        self._cache_staleness_threshodl = datetime.timedelta(
            minutes=cache_staleness_threshold_in_minutes)
        self._lock = threading.Lock()

    def get(self, key):
        self._lock.acquire()

        timestamp, val = self._content.get(key, (None, None))
        self._lock.release()
        if timestamp:
            if timestamp >= datetime.datetime.now() - self._cache_staleness_threshodl:
                return val
        else:
            return None

    def put(self, key, val):
        self._lock.acquire()
        self._content[key] = (datetime.datetime.now(), val)
        try:
            with open(self._cache_file, 'w') as cf:
                cloudpickle.dump(self._content, cf)
        finally:
            self._lock.release()


def check_success_msg_and_extract_result(response, success_msg, pattern):
    if success_msg not in response:
        return None

    m = re.search(pattern, response)
    res = None
    if m:
        try:
            res = m.group(1)
            res = codec.loads_base64(res)
        except AttributeError:
            res = None

    return res


def run_function_over_ssh(fn, arg, host):
    fn_arg_obj = codec.dumps_base64((fn, arg))
    ssh = Ssh(host)
    try:
        ssh.openShell()
        command = '{python} -m {remote_script_file_path} {fn_arg_obj}'.format(
            python=sys.executable,
            host=host,
            remote_script_file_path=REMOTE_EXECUTER_SCRIPT,
            fn_arg_obj=fn_arg_obj)

        ssh.sendShell(command)
        response = ssh.recv()
        result = \
            check_success_msg_and_extract_result(response,
                                                 'FUNCTION SUCCESSFULLY EXECUTED',
                                                 'RESULT: (.+?) EOM')
    finally:
        ssh.closeConnection()

    if result is None:
        raise RuntimeError(
            "remote execution was "
            "unsuccessful on host: {host}.".format(host=host))

    return result


def execute_function_multithreaded(fn, args_list, max_concurrent_executions):
    result_queue = Queue.Queue()
    worker_queue = Queue.Queue()

    for i, arg in enumerate(args_list):
        arg.append(i)
        worker_queue.put(arg)

    def fn_execute():
        while True:
            try:
                arg = worker_queue.get(block=False)
            except Queue.Empty:
                return
            exec_index = arg[-1]
            res = fn(*arg[:-1])
            result_queue.put((exec_index, res))

    threads = []
    number_of_threads = min(max_concurrent_executions, len(args_list))

    for _ in range(number_of_threads):
        thread = threading.Thread(target=fn_execute)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    results = {}
    while not result_queue.empty():
        item = result_queue.get()
        results[item[0]] = item[1]

    if len(results) != len(args_list):
        raise RuntimeError(
            'Some threads for func {func} did not complete '
            'successfully.'.format(func=fn.__name__))

    return results


def execute_function_over_ssh_multithreaded(fn, args_list,
                                            max_concurrent_executions):
    args_list = [[fn] + arg for arg in args_list]
    result_queue = execute_function_multithreaded(run_function_over_ssh,
                                                  args_list,
                                                  max_concurrent_executions)

    return result_queue
