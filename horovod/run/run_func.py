import cloudpickle
import tempfile
import os
import sys
import six
import subprocess
import collections
import textwrap
import time

from horovod.run.util import network
from horovod.run.common.util import safe_shell_exec

_TAIL_LINES_TO_KEEP = 100

_LOCAL_PICKLED_RESULT_FILENAME = "local_result.pkl"
_PICKLED_RESULT_FILENAME = "result.pkl"
_PICKLED_PROC_FN_FILENAME = "proc_fn.pkl"
_PROC_LAUNCHER_FILENAME = "launch.sh"


def run_func(
        proc_fn,
        num_proc,
        host,
        ssh_port=None,
        disable_cache=False,
        start_timeout=None,
        verbose=False):
    wdir = tempfile.mkdtemp()
    wdir_par = os.path.dirname(wdir)
    if verbose:
        print("mpirun working dir is " + wdir)

    if ssh_port:
        ssh_port_opt = "-p " + str(ssh_port)
    else:
        ssh_port_opt = ""

    # Invokes proc_fn with args. So we don't need to pickle them separately.
    def wrapped_proc_fn(rank=0):
        return_value = proc_fn()
        if rank == 0:
            with open(_LOCAL_PICKLED_RESULT_FILENAME, 'wb') as f:
                try:
                    cloudpickle.dump(return_value, f)
                except Exception as e:
                    raise RuntimeError("Caught an excpetion while pickling "
                                       "return value: {}".format(repr(e)))

    pickled_proc_fn_str = cloudpickle.dumps(wrapped_proc_fn)
    pickled_proc_fn_path = os.path.join(wdir, _PICKLED_PROC_FN_FILENAME)
    with open(pickled_proc_fn_path, 'wb') as f:
        f.write(pickled_proc_fn_str)

    local_ip = network.get_local_ip_addr()
    launcher_path = os.path.join(wdir, _PROC_LAUNCHER_FILENAME)
    with open(launcher_path, 'w') as f:
        f.write(textwrap.dedent("""
        set -e
        cd {wdir}
        rank=${{OMPI_COMM_WORLD_RANK:-0}}
        {exec} -c "import cloudpickle; cloudpickle.load(open('{fn_file}', 'rb'))(rank=$rank)"
        if [[ "$rank" -eq 0 ]]; then
        scp -q {ssh_port_opt} -o StrictHostKeyChecking=no \
        {wdir}/{local_result} {local_ip}:{wdir}/{result}
        fi
        """.format(wdir=wdir,
                   exec=sys.executable,
                   ssh_port_opt=ssh_port_opt,
                   fn_file=_PICKLED_PROC_FN_FILENAME,
                   local_ip=local_ip,
                   local_result=_LOCAL_PICKLED_RESULT_FILENAME,
                   result=_PICKLED_RESULT_FILENAME)))

    os.chmod(launcher_path, 0o777)

    all_host_names = list(set(x.split(':')[0] for x in host.split(',')))
    remote_host_names = network.filter_local_addresses(all_host_names)

    for remote_host in remote_host_names:
        scp_cmd = textwrap.dedent(
            """
            ssh {ssh_port_opt} -o StrictHostKeyChecking=no {remote_host} mkdir -p {wdir_par} && \
            scp -r -q {ssh_port_opt} -o StrictHostKeyChecking=no \
            {wdir} {remote_host}:{wdir_par}
            """).format(ssh_port_opt=ssh_port_opt, wdir=wdir,
                        remote_host=remote_host, wdir_par=wdir_par)
        output = six.StringIO()
        exit_code = safe_shell_exec.execute(scp_cmd, stdout=output, stderr=output)
        if exit_code != 0:
            output_msg = output.getvalue()
            raise RuntimeError("Copy working dir to remote host {remote_host} failed."
                               "Error message is {msg}"
                               .format(remote_host=remote_host, msg=output_msg))

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

    cmd += ["bash", launcher_path]

    # Use `os.environ` to preserve the environ to support nested MPI jobs
    task = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            stdin=subprocess.PIPE,
                            env=os.environ)
    task.stdin.close()

    tail = collections.deque(maxlen=_TAIL_LINES_TO_KEEP)
    try:
        for line in task.stdout:
            decoded = line.decode()
            tail.append(decoded)
            sys.stdout.write(decoded)
        task.wait()
    finally:
        if task.poll() is None:
            try:
                task.terminate()  # SIGTERM
                time.sleep(0.5)
                if task.poll() is None:
                    task.kill()  # SIGKILL
            except OSError:
                pass

    if task.returncode != os.EX_OK:
        if len(tail) == _TAIL_LINES_TO_KEEP:
            last_n_msg = "last %d lines of the task output are" % _TAIL_LINES_TO_KEEP
        else:
            last_n_msg = "task output is"
        raise RuntimeError(
            "Command %s failed with return code %d.\n" % (cmd, task.returncode) +
            """
            The %s included below.
            """ % last_n_msg + "\n%s\n" % "".join(tail))

    result_path = os.path.join(wdir, _PICKLED_RESULT_FILENAME)
    with open(result_path, 'rb') as f:
        result_bytes = f.read()

    return cloudpickle.loads(result_bytes)
