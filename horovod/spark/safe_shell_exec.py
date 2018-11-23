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

import os
import psutil
import signal
import subprocess
import sys
import threading
import time


GRACEFUL_TERMINATION_TIME = 5


def terminate_executor_shell_and_children(pid):
    p = psutil.Process(pid)

    # Terminate children gracefully.
    for child in p.children():
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    # Wait for graceful termination.
    time.sleep(GRACEFUL_TERMINATION_TIME)

    # Send STOP to executor shell to stop progress.
    p.send_signal(signal.SIGSTOP)

    # Kill children recursively.
    for child in p.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    # Kill shell itself.
    p.kill()


def forward_stream(src, dst):
    while True:
        line = src.readline()
        if not line:
            break
        dst.write(line)


def execute(command, env=None, stdout=None, stderr=None):
    (r, w) = os.pipe()
    middleman_pid = os.fork()
    if middleman_pid == 0:
        os.close(w)
        os.setpgid(0, 0)

        # Redirect command stdout & stderr to provided streams or sys.stdout/sys.stderr.
        # This is useful for Jupyter Notebook that uses custom sys.stdout/sys.stderr or
        # redirecting to a file on disk.
        executor_shell = subprocess.Popen(command, shell=True, env=env,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if stdout is None:
            stdout = sys.stdout
        stdout_fwd = threading.Thread(target=forward_stream, args=(executor_shell.stdout, stdout))
        stdout_fwd.start()

        if stderr is None:
            stderr = sys.stderr
        stderr_fwd = threading.Thread(target=forward_stream, args=(executor_shell.stderr, stderr))
        stderr_fwd.start()

        sigterm_received = threading.Event()

        def set_sigterm_received(signum, frame):
            sigterm_received.set()

        signal.signal(signal.SIGINT, set_sigterm_received)
        signal.signal(signal.SIGTERM, set_sigterm_received)

        def kill_executor_children_if_parent_dies():
            os.read(r, 1)
            terminate_executor_shell_and_children(executor_shell.pid)

        bg = threading.Thread(target=kill_executor_children_if_parent_dies)
        bg.daemon = True
        bg.start()

        def kill_executor_children_if_sigterm_received():
            sigterm_received.wait()
            terminate_executor_shell_and_children(executor_shell.pid)

        bg = threading.Thread(target=kill_executor_children_if_sigterm_received)
        bg.daemon = True
        bg.start()

        exit_code = executor_shell.wait()
        stdout_fwd.join()
        stderr_fwd.join()
        os._exit(exit_code)

    os.close(r)
    try:
        _, status = os.waitpid(middleman_pid, 0)
    except:
        # interrupted, send middleman TERM signal which will terminate children
        os.kill(middleman_pid, signal.SIGTERM)
        while True:
            try:
                _, status = os.waitpid(middleman_pid, 0)
                break
            except:
                # interrupted, wait for middleman to finish
                pass

    exit_code = status >> 8
    return exit_code


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <command>' % sys.argv[0])
        sys.exit(1)
    sys.exit(execute(sys.argv[1]))
