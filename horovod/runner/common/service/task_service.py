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

import abc
import threading

from horovod.runner.common import util
from horovod.runner.common.util import network, safe_shell_exec, timeout
from horovod.runner.util.streams import Pipe
from horovod.runner.util.threads import in_thread

WAIT_FOR_COMMAND_MIN_DELAY = 0.1


class RunCommandRequest(object):
    def __init__(self, command, env, capture_stdout=False, capture_stderr=False,
                 prefix_output_with_timestamp=False):
        self.command = command
        """Command to run."""
        self.env = env
        """Environment to use."""
        self.capture_stdout = capture_stdout
        """Captures stdout of command if True. Retrieve content via StreamCommandStdOutRequest."""
        self.capture_stderr = capture_stderr
        """Captures stderr of command if True. Retrieve content via StreamCommandErrOutRequest."""
        self.prefix_output_with_timestamp = prefix_output_with_timestamp


class StreamCommandOutputRequest(object, metaclass=abc.ABCMeta):
    pass


class StreamCommandStdOutRequest(StreamCommandOutputRequest):
    """Streams the command stdout to the client."""
    pass


class StreamCommandStdErrRequest(StreamCommandOutputRequest):
    """Streams the command stderr to the client."""
    pass


class CommandOutputNotCaptured(Exception):
    """Executed command does not capture its output."""
    pass


class AbortCommandRequest(object):
    """Aborts the command currently running."""
    pass


class CommandExitCodeRequest(object):
    """Get command result"""
    pass


class CommandExitCodeResponse(object):
    def __init__(self, terminated, exit_code):
        self.terminated = terminated
        """Yes/no"""
        self.exit_code = exit_code
        """Exit code returned from command if terminated, None otherwise"""


class WaitForCommandExitCodeRequest(object):
    """Wait for command exit code. Blocks until command terminated or connection closed."""

    def __init__(self, delay):
        """
        :param delay: delay in seconds
        :type delay: float
        """
        self.delay = delay
        """Delay in seconds between termination checks."""


class WaitForCommandExitCodeResponse(object):
    def __init__(self, exit_code):
        self.exit_code = exit_code
        """Exit code returned from command, None if connection closed."""


class NotifyInitialRegistrationCompleteRequest(object):
    """Notification that initial task registration has completed."""
    pass


class RegisterCodeResultRequest(object):
    """Register code execution results with task."""

    def __init__(self, result):
        self.result = result


class BasicTaskService(network.BasicService):
    def __init__(self, name, index, key, nics, command_env=None, verbose=0):
        super(BasicTaskService, self).__init__(name, key, nics)
        self._initial_registration_complete = False
        self._wait_cond = threading.Condition()
        self._index = index
        self._command_env = command_env
        self._command_stdout = None
        self._command_stderr = None
        self._command_abort = None
        self._command_exit_code = None
        self._verbose = verbose

        self._command_thread = None
        self._fn_result = None

    def _run_command(self, command, env, event,
                     stdout=None, stderr=None, index=None,
                     prefix_output_with_timestamp=False):
        self._command_exit_code = safe_shell_exec.execute(
            command,
            env=env,
            stdout=stdout, stderr=stderr,
            index=index,
            prefix_output_with_timestamp=prefix_output_with_timestamp,
            events=[event])
        if stdout:
            stdout.close()
        if stderr:
            stderr.close()

    def _add_envs(self, env, extra_env):
        """
        Adds extra_env to env.

        :param env: dict representing environment variables
        :param extra_env: additional variables to be added to env
        """
        for key, value in extra_env.items():
            if value is None:
                if key in env:
                    del env[key]
            else:
                env[key] = value

    def _handle(self, req, client_address):
        if isinstance(req, RunCommandRequest):
            self._wait_cond.acquire()
            try:
                if self._command_thread is None:
                    # we add req.env to _command_env and make this available to the executed command
                    if self._command_env:
                        env = self._command_env.copy()
                        self._add_envs(env, req.env)
                        req.env = env

                    if self._verbose >= 2:
                        print("Task service executes command: {}".format(req.command))
                        if self._verbose >= 3:
                            for key, value in req.env.items():
                                if 'SECRET' in key:
                                    value = '*' * len(value)
                                print("Task service env: {} = {}".format(key, value))

                    # We only permit executing exactly one command, so this is idempotent.
                    self._command_abort = threading.Event()
                    self._command_stdout = Pipe() if req.capture_stdout else None
                    self._command_stderr = Pipe() if req.capture_stderr else None
                    args = (req.command, req.env, self._command_abort,
                            self._command_stdout, self._command_stderr,
                            self._index,
                            req.prefix_output_with_timestamp)
                    self._command_thread = in_thread(self._run_command, args)
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, StreamCommandOutputRequest):
            # Wait for command to start
            self.wait_for_command_start()

            # We only expect streaming each command output stream once concurrently
            if isinstance(req, StreamCommandStdOutRequest):
                return self.stream_output(self._command_stdout)
            elif isinstance(req, StreamCommandStdErrRequest):
                return self.stream_output(self._command_stderr)
            else:
                return CommandOutputNotCaptured()

        if isinstance(req, AbortCommandRequest):
            self._wait_cond.acquire()
            try:
                if self._command_thread is not None:
                    self._command_abort.set()
                if self._command_stdout is not None:
                    self._command_stdout.close()
                if self._command_stderr is not None:
                    self._command_stderr.close()
            finally:
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, NotifyInitialRegistrationCompleteRequest):
            self._wait_cond.acquire()
            try:
                self._initial_registration_complete = True
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, CommandExitCodeRequest):
            self._wait_cond.acquire()
            try:
                terminated = (self._command_thread is not None and
                              not self._command_thread.is_alive())
                return CommandExitCodeResponse(terminated,
                                               self._command_exit_code if terminated else None)
            finally:
                self._wait_cond.release()

        if isinstance(req, WaitForCommandExitCodeRequest):
            self._wait_cond.acquire()
            try:
                while self._command_thread is None or self._command_thread.is_alive():
                    self._wait_cond.wait(max(req.delay, WAIT_FOR_COMMAND_MIN_DELAY))
                return WaitForCommandExitCodeResponse(self._command_exit_code)
            finally:
                self._wait_cond.release()

        if isinstance(req, RegisterCodeResultRequest):
            self._fn_result = req.result
            return network.AckResponse()

        return super(BasicTaskService, self)._handle(req, client_address)

    def fn_result(self):
        return self._fn_result

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while not self._initial_registration_complete:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('tasks to start')
        finally:
            self._wait_cond.release()

    def wait_for_command_start(self, timeout=None):
        """Waits for a command to start.

        Returns as soon as the command has started. This method raises an exception
        when given optional timeout.Timeout instance runs out.

        :param timeout: timeout.Timeout
        """
        self._wait_cond.acquire()
        try:
            while self._command_thread is None:
                if timeout:
                    self._wait_cond.wait(timeout.remaining())
                    timeout.check_time_out_for('command to run')
                else:
                    self._wait_cond.wait()
        finally:
            self._wait_cond.release()

    def check_for_command_start(self, seconds):
        """Checks that a command has started.

        Returns True as soon as the command has started, waits at most `seconds` seconds and
        returns False if command has not started in that time.

        :param seconds: seconds
        :return True or False indication command start
        """
        self._wait_cond.acquire()
        try:
            tmout = util.timeout.Timeout(seconds, 'Timed out waiting for {activity}')

            while self._command_thread is None:
                self._wait_cond.wait(tmout.remaining())
                if tmout.remaining() == 0:
                    return self._command_thread is not None
            return True
        finally:
            self._wait_cond.release()

    def wait_for_command_termination(self):
        self._command_thread.join()

    def command_exit_code(self):
        return self._command_exit_code

    def stream_output(self, stream):
        self._wait_cond.acquire()
        try:
            # Fail if command does not capture this stream
            if stream is None:
                return CommandOutputNotCaptured()
            return network.AckStreamResponse(), stream
        finally:
            self._wait_cond.release()


class BasicTaskClient(network.BasicClient):
    def __init__(self, service_name, task_addresses, key, verbose,
                 match_intf=False, attempts=3):
        super(BasicTaskClient, self).__init__(service_name,
                                              task_addresses, key, verbose,
                                              match_intf=match_intf,
                                              attempts=attempts)

    def run_command(self, command, env,
                    capture_stdout=False, capture_stderr=False,
                    prefix_output_with_timestamp=False):
        self._send(RunCommandRequest(command, env,
                                     capture_stdout, capture_stderr,
                                     prefix_output_with_timestamp))

    def stream_command_output(self, stdout=None, stderr=None):
        def send(req, stream):
            try:
                self._send(req, stream)
            except Exception as e:
                self.abort_command()
                raise e

        return (in_thread(send, (StreamCommandStdOutRequest(), stdout)) if stdout else None,
                in_thread(send, (StreamCommandStdErrRequest(), stderr)) if stderr else None)

    def abort_command(self):
        self._send(AbortCommandRequest())

    def notify_initial_registration_complete(self):
        self._send(NotifyInitialRegistrationCompleteRequest())

    def command_terminated(self):
        terminated, _ = self.command_result()
        return terminated

    def command_result(self):
        """
        Returns the command's result if terminated, or None.
        :return: terminated flag and exit code
        """
        resp = self._send(CommandExitCodeRequest())
        return resp.terminated, resp.exit_code

    def register_code_result(self, result):
        self._send(RegisterCodeResultRequest(result))

    def wait_for_command_termination(self, delay=1.0):
        """
        Wait for command termination. Blocks until command terminated or connection closed.

        :param delay: delay in seconds
        :type delay: float
        """
        self.wait_for_command_exit_code(delay)

    def wait_for_command_exit_code(self, delay=1.0):
        """
        Wait for command termination and retrieve exit code.
        Blocks until command terminated or connection closed.

        :param delay: delay in seconds
        :type delay: float
        :return: exit code
        """
        try:
            resp = self._send(WaitForCommandExitCodeRequest(delay))
            return resp.exit_code
        except:
            pass
