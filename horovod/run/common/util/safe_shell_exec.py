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
import psutil
import re
import signal
import subprocess
import sys
import threading
import time


GRACEFUL_TERMINATION_TIME_S = 5


def terminate_executor_shell_and_children(pid):
    # If the shell already ends, no need to terminate its child.
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Terminate children gracefully.
    for child in p.children():
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    # Wait for graceful termination.
    time.sleep(GRACEFUL_TERMINATION_TIME_S)

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


def forward_stream(src_fd, dst_stream, prefix, index):
    with os.fdopen(src_fd, 'r') as src:
        line_buffer = ''
        while True:
            text = os.read(src.fileno(), 1000)
            if not isinstance(text, str):
                text = text.decode('utf-8')
            if not text:
                break

            for line in re.split('([\r\n])', text):
                line_buffer += line
                if line == '\r' or line == '\n':
                    if index is not None:
                        localtime = time.asctime(time.localtime(time.time()))
                        line_buffer = '{time}[{rank}]<{prefix}>:{line}'.format(
                            time=localtime,
                            rank=str(index),
                            prefix=prefix,
                            line=line_buffer
                        )

                    dst_stream.write(line_buffer)
                    dst_stream.flush()
                    line_buffer = ''


def execute(command, env=None, stdout=None, stderr=None, index=None, event=None):
    # Make a pipe for the subprocess stdout/stderr.
    (stdout_r, stdout_w) = os.pipe()
    (stderr_r, stderr_w) = os.pipe()

    # Make a pipe for notifying the child that parent has died.
    (r, w) = os.pipe()

    middleman_pid = os.fork()
    if middleman_pid == 0:
        # Close unused file descriptors to enforce PIPE behavior.
        os.close(w)
        os.setsid()

        executor_shell = subprocess.Popen(command, shell=True, env=env,
                                          stdout=stdout_w, stderr=stderr_w)

        sigterm_received = threading.Event()

        def set_sigterm_received(signum, frame):
            sigterm_received.set()

        signal.signal(signal.SIGINT, set_sigterm_received)
        signal.signal(signal.SIGTERM, set_sigterm_received)

        def kill_executor_children_if_parent_dies():
            # This read blocks until the pipe is closed on the other side
            # due to the process termination.
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
        os._exit(exit_code)

    # Close unused file descriptors to enforce PIPE behavior.
    os.close(r)
    os.close(stdout_w)
    os.close(stderr_w)

    # Redirect command stdout & stderr to provided streams or sys.stdout/sys.stderr.
    # This is useful for Jupyter Notebook that uses custom sys.stdout/sys.stderr or
    # for redirecting to a file on disk.
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr
    stdout_fwd = threading.Thread(target=forward_stream, args=(stdout_r, stdout, 'stdout', index))
    stderr_fwd = threading.Thread(target=forward_stream, args=(stderr_r, stderr, 'stderr', index))
    stdout_fwd.start()
    stderr_fwd.start()

    def kill_middleman_if_master_thread_terminate():
        event.wait()
        try:
            os.kill(middleman_pid, signal.SIGTERM)
        except:
            # The process has already been killed elsewhere
            pass

    # TODO: Currently this requires explicitly declaration of the event and signal handler to set
    #  the event (gloo_run.py:_launch_jobs()). Need to figure out a generalized way to hide this behind
    #  interfaces.
    if event is not None:
        bg_thread = threading.Thread(target=kill_middleman_if_master_thread_terminate)
        bg_thread.daemon = True
        bg_thread.start()

    try:
        res, status = os.waitpid(middleman_pid, 0)
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

    stdout_fwd.join()
    stderr_fwd.join()
    exit_code = status >> 8
    return exit_code
