# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import print_function
import cloudpickle
import tempfile
import os
import sys
import six
import shutil
import textwrap

from horovod.run.common.util import safe_shell_exec

_PICKLED_RESULT_PREFIX = "result.pkl"

_LOCAL_PICKLED_PROC_FN_FILENAME = "local_proc_fn.pkl"
_PICKLED_PROC_FN_FILENAME = "proc_fn.pkl"

_LOCAL_PROC_LAUNCHER_FILENAME = "local_launch.sh"
_PROC_LAUNCHER_FILENAME = "launch.sh"


def run_func(
        fn,
        num_proc,
        host=None,
        ssh_port=None,
        disable_cache=False,
        start_timeout=None,
        env=None,
        verbose=False):
    """
    Run horovod inside python program.
    Run `num_proc` processes executing `fn` on specified hosts.

    :param fn: Function to run. The function take no arguments and its return value will be
               collected and return by `run_func`.
               We can use a lambda function to wrap functions with arguments, such as:
               `run_func(fn=lambda: fn1(arg1, arg2, ..), num_proc=2, ...)`
    :param num_proc: Number of Horovod processes.
    :param host: To specify the list of host names as well as the
                 number of available slots on each host for
                 training processes using the following format:
                 <hostname>:<number of slots>,... .
                 E.g., host1:2,host2:4,host3:1 indicates that 2 processes can run on
                 host1, 4 processes on host2, and 1 process on host3.
                 If None, launch all mpi processes on local node.
    :param ssh_port: SSH port on all the hosts.
                     If None, use default SSH port.
    :param disable_cache: If the flag is not set, horovodrun will perform
                          the initialization checks only once every 60
                          minutes -- if the checks successfully pass.
                          Otherwise, all the checks will run every time
                          horovodrun is called."
    :param start_timeout: Horovodrun has to perform all the checks and
                          start the processes before the specified
                          timeout. The default value is 30 seconds.
                          Alternatively, The environment variable
                          HOROVOD_START_TIMEOUT can also be used to
                          specify the initialization timeout.
    :param env: Environment dictionary to use in Horovod run.
                If None, use `os.environ`.
    :param verbose: If this flag is set, extra messages will be printed.

    :return: A list contains returned result from each MPI process. The ith value of the list
             corresponds to the returned result of rank i process.
    """

    try:
        wdir = tempfile.mkdtemp()
        if verbose:
            print("horovod.run_func working dir is " + wdir)

        if ssh_port:
            ssh_port_opt = "-p " + str(ssh_port)
            scp_port_opt = "-P " + str(ssh_port)
        else:
            ssh_port_opt = ""
            scp_port_opt = ""

        def wrapped_proc_fn(rank=0):
            return_value = fn()

            result_file = "{prefix}.{rank}".format(prefix=_PICKLED_RESULT_PREFIX, rank=rank)
            with open(result_file, 'wb') as fp:
                try:
                    cloudpickle.dump(return_value, fp)
                except Exception as e:
                    raise RuntimeError("Caught an excpetion while pickling "
                                       "return value: {}".format(str(return_value)), e)

        local_pickled_proc_fn_path = os.path.join(wdir, _LOCAL_PICKLED_PROC_FN_FILENAME)
        with open(local_pickled_proc_fn_path, 'wb') as fp:
            try:
                cloudpickle.dump(wrapped_proc_fn, fp)
            except Exception as e:
                raise RuntimeError("Caught an excpetion while pickling "
                                   "wrapped fn: {}".format(str(wrapped_proc_fn)), e)

        local_launcher_path = os.path.join(wdir, _LOCAL_PROC_LAUNCHER_FILENAME)
        with open(local_launcher_path, 'w') as f:
            f.write(textwrap.dedent("""
            set -e
            cd "{wdir}"
            rank=${{OMPI_COMM_WORLD_RANK:-0}}
            {python_exec} -c \
            "import cloudpickle; cloudpickle.load(open('{fn_file}', 'rb'))(rank=$rank)"
            """.format(wdir=wdir,
                       python_exec=sys.executable,
                       fn_file=_PICKLED_PROC_FN_FILENAME)))

        if host is not None:
            all_host_names = set(x.split(':')[0] for x in host.split(','))
        else:
            # If host is None, launch all MPI processes on local node.
            all_host_names = ['127.0.0.1']

        # copy pickled function and launcher script to all hosts via scp
        # TODO: parallelize the scp copy, note that when scp in parallel, there could be
        #       race conditions because different hosts might have the same NFS mount.
        for hostname in all_host_names:
            scp_cmd = textwrap.dedent(
                """
                ssh {ssh_port_opt} -o StrictHostKeyChecking=no {hostname} mkdir -p {wdir} && \
                scp -r -q {scp_port_opt} -o StrictHostKeyChecking=no \
                {wdir}/{local_fn_file} {hostname}:{wdir}/{fn_file} && \
                scp -r -q {scp_port_opt} -o StrictHostKeyChecking=no \
                {wdir}/{local_launcher} {hostname}:{wdir}/{launcher}
                """).format(ssh_port_opt=ssh_port_opt, scp_port_opt=scp_port_opt,
                            wdir=wdir, hostname=hostname,
                            local_fn_file=_LOCAL_PICKLED_PROC_FN_FILENAME,
                            fn_file=_PICKLED_PROC_FN_FILENAME,
                            local_launcher=_LOCAL_PROC_LAUNCHER_FILENAME,
                            launcher=_PROC_LAUNCHER_FILENAME)

            safe_shell_exec.execute_swallow_output(scp_cmd)

        cmd = ["horovodrun", "-np", str(num_proc)]
        if ssh_port:
            cmd += ["-p", str(ssh_port)]
        if host:
            cmd += ["-H", str(host)]
        if disable_cache:
            cmd += ["--disable-cache"]
        if start_timeout:
            cmd += ["--start-timeout", str(start_timeout)]
        if verbose:
            cmd += ["--verbose"]

        cmd += ["bash", os.path.join(wdir, _PROC_LAUNCHER_FILENAME)]

        if verbose:
            print("Run command: " + " ".join(cmd))

        if env is None:
            env = os.environ

        exit_code = safe_shell_exec.execute(cmd, stdout=sys.stdout, stderr=sys.stderr,
                                            env=env, disable_stdin=True)
        if exit_code != 0:
            raise RuntimeError("Executing command {cmd} failed, exit code is {exit_code}"
                               .format(cmd=" ".join(cmd), exit_code=exit_code))

        # copy result file back via scp
        # TODO: parallelize the scp copy, and if different hosts might have the same NFS mount,
        #       avoid copying back the same result files multiple times.
        for hostname in all_host_names:
            scp_cmd = textwrap.dedent(
                """
                mkdir -p {wdir}/result
                scp -r -q {scp_port_opt} -o StrictHostKeyChecking=no \
                {hostname}:{wdir}/{prefix}.* {wdir}/result/
                """).format(scp_port_opt=scp_port_opt, wdir=wdir,
                            hostname=hostname, prefix=_PICKLED_RESULT_PREFIX)

            safe_shell_exec.execute_swallow_output(scp_cmd)

        results = [None] * num_proc
        for i in range(num_proc):
            result_path = "{wdir}/result/{prefix}.{rank}".format(
                wdir=wdir, prefix=_PICKLED_RESULT_PREFIX, rank=i)
            try:
                with open(result_path, 'rb') as fp:
                    results[i] = cloudpickle.loads(fp)
            except Exception as e:
                raise RuntimeError("Caught an excpetion while loading pickled results file: "
                                   "{}".format(result_path), e)
        return results
    finally:
        # cleanup working dir on each node.
        # do local cleanup first.
        # In case ssh broken we can cleanup locally.
        shutil.rmtree(wdir)
        for hostname in all_host_names:
            ssh_del_cmd = "ssh {ssh_port_opt} -o StrictHostKeyChecking=no " \
                          "{hostname} rm -rf {wdir}".format(hostname=hostname, wdir=wdir)
            safe_shell_exec.execute_swallow_output(ssh_del_cmd, swallow_exception=True)

