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

import codecs
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import threading
import time

import psutil

from horovod.runner.util.threads import in_thread, on_event

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
    gone, alive = psutil.wait_procs(p.children(), timeout=GRACEFUL_TERMINATION_TIME_S)

    # Freeze the process to prevent it from spawning any new children.
    try:
        p.send_signal(signal.SIGSTOP)
    except psutil.NoSuchProcess:
        pass

    # Kill children recursively.
    for child in alive:
        try:
            for grandchild in child.children(recursive=True):
                try:
                    grandchild.kill()
                except psutil.NoSuchProcess:
                    pass
            child.kill()
        except psutil.NoSuchProcess:
            pass

    # Kill shell itself.
    try:
        p.terminate()
    except psutil.NoSuchProcess:
        pass

    try:
        p.wait(timeout=GRACEFUL_TERMINATION_TIME_S)
    except psutil.TimeoutExpired:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def prefix_connection(src_connection, dst_stream, prefix, index, prefix_output_with_timestamp):
    """
    Prefixes the given source connection with timestamp, a prefix and an index.
    Each line of the source will be prefix in this format, if index and prefix are not None:
        {time}[{index}]<{prefix}>:{line}
    The dst_stream must be text streams.

    :param src_connection: source pipe connection
    :param dst_stream: destination text stream
    :param prefix: prefix string
    :param index: index value
    :param prefix_output_with_timestamp: prefix lines in dst_stream with timestamp
    :return: None
    """
    def get_context(rank, prefix):
        localtime = time.asctime(time.localtime(time.time())) if prefix_output_with_timestamp else ''
        return '{time}[{rank}]<{prefix}>:'.format(
            time=localtime,
            rank=str(rank),
            prefix=prefix
        )

    def write(text):
        if index is not None and prefix is not None:
            context = get_context(index, prefix)
            text = context + text
        dst_stream.write(text)
        dst_stream.flush()

    # the incremental encoder allows us to decode chunks of utf8 bytes
    # with utf8 characters spread across the boundary chunks
    decoder = codecs.getincrementaldecoder('utf8')()

    line_buffer = ''
    while True:
        # read at most that many bytes, but do not wait until that many bytes become available
        # waits for the first bytes so this is not a busy loop
        buf = os.read(src_connection.fileno(), 1000)

        # turn the bytes into string decoding them as utf8
        # we need to use an incremental decoder as characters can span multiple bytes
        # where the last character might not be completely in buf
        # see https://github.com/horovod/horovod/issues/2367
        text = decoder.decode(buf or b'', final=not buf)

        # write line_buffer out when we reach an \n or \r
        # the latter is used to update the current line (e.g. progress bar)
        # which we want to flush out (and prefix) as soon as possible
        for line in re.split('([\r\n])', text):
            line_buffer += line
            if line == '\r' or line == '\n':
                write(line_buffer)
                line_buffer = ''

        # not an empty text indicates EOF but an empty buf
        if not buf:
            break

    # flush the line buffer if it is not empty
    if len(line_buffer):
        write(line_buffer)

    src_connection.close()


def _exec_middleman(command, env, exit_event, stdout, stderr, rw):
    stdout_r, stdout_w = stdout
    stderr_r, stderr_w = stderr
    r, w = rw

    # Close unused file descriptors to enforce PIPE behavior.
    stdout_r.close()
    stderr_r.close()
    w.close()
    os.setsid()

    executor_shell = subprocess.Popen(command, shell=True, env=env,
                                      stdout=stdout_w, stderr=stderr_w)

    # we don't bother stopping the on_event thread, this process sys.exits soon
    # so the on_event thread has to be a daemon thread
    on_event(exit_event, terminate_executor_shell_and_children, args=(executor_shell.pid,), daemon=True)

    def kill_executor_children_if_parent_dies():
        # This read blocks until the pipe is closed on the other side
        # due to parent process termination (for any reason, including -9).
        os.read(r.fileno(), 1)
        terminate_executor_shell_and_children(executor_shell.pid)

    in_thread(kill_executor_children_if_parent_dies)

    exit_code = executor_shell.wait()
    if exit_code < 0:
        # See: https://www.gnu.org/software/bash/manual/html_node/Exit-Status.html
        exit_code = 128 + abs(exit_code)

    sys.exit(exit_code)


def _create_event(ctx):
    # We need to expose this method for internal testing purposes, so we can mock it out to avoid
    # leaking semaphores.
    return ctx.Event()


def execute(command, env=None, stdout=None, stderr=None, index=None, events=None,
            prefix_output_with_timestamp=False):
    """
    Execute the given command and forward stdout and stderr of the command to the given
    stdout and stderr text streams, or sys.stdout and sys.stderr, respectively, if None given.
    Prefixes each line with index and timestamp if index is not None. The timestamp
    can be disabled with prefix_output_with_timestamp set False.
    The command will be terminated when any of the given events are set.

    :param command: command to execute
    :param env: environment variables to execute command with
    :param stdout: stdout text stream, sys.stdout if None
    :param stderr: stderr text stream, sys.stderr if None
    :param index: index used to prepend text streams
    :param events: events to terminate the command
    :param prefix_output_with_timestamp: prepend text streams with timestamp if True
    :return: command's exit code
    """
    ctx = multiprocessing.get_context('spawn')

    # When this event is set, signal to middleman to terminate its children and exit.
    exit_event = _create_event(ctx)

    # Make a pipe for the subprocess stdout/stderr.
    (stdout_r, stdout_w) = ctx.Pipe(duplex=False)
    (stderr_r, stderr_w) = ctx.Pipe(duplex=False)

    # This Pipe is how we ensure that the executed process is properly terminated (not orphaned) if
    # the parent process is hard killed (-9). If the parent (this process) is killed for any reason,
    # this Pipe will be closed, which can be detected by the middleman. When the middleman sees the
    # closed Pipe, it will issue a SIGTERM to the subprocess executing the command. The assumption
    # here is that users will be inclined to hard kill this process, not the middleman.
    (r, w) = ctx.Pipe(duplex=False)

    middleman = ctx.Process(target=_exec_middleman, args=(command, env, exit_event,
                                                          (stdout_r, stdout_w),
                                                          (stderr_r, stderr_w),
                                                          (r, w)))
    middleman.start()

    # Close unused file descriptors to enforce PIPE behavior.
    r.close()
    stdout_w.close()
    stderr_w.close()

    # Redirect command stdout & stderr to provided streams or sys.stdout/sys.stderr.
    # This is useful for Jupyter Notebook that uses custom sys.stdout/sys.stderr or
    # for redirecting to a file on disk.
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr

    stdout_fwd = in_thread(target=prefix_connection, args=(stdout_r, stdout, 'stdout', index, prefix_output_with_timestamp))
    stderr_fwd = in_thread(target=prefix_connection, args=(stderr_r, stderr, 'stderr', index, prefix_output_with_timestamp))

    # TODO: Currently this requires explicitly declaration of the events and signal handler to set
    #  the event (gloo_run.py:_launch_jobs()). Need to figure out a generalized way to hide this behind
    #  interfaces.
    stop = threading.Event()
    events = events or []
    for event in events:
        on_event(event, exit_event.set, stop=stop, silent=True)

    try:
        middleman.join()
    except:
        # interrupted, send middleman TERM signal which will terminate children
        exit_event.set()
        while True:
            try:
                middleman.join()
                break
            except:
                # interrupted, wait for middleman to finish
                pass
    finally:
        stop.set()

    stdout_fwd.join()
    stderr_fwd.join()

    return middleman.exitcode
