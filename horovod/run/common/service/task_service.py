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

import threading

from horovod.run.common.util import network
from horovod.run.common.util import safe_shell_exec
from horovod.run.util.threads import in_thread


class RunCommandRequest(object):
    def __init__(self, command, env):
        self.command = command
        """Command to run."""
        self.env = env
        """Environment to use."""


class CommandExitCodeRequest(object):
    """Get command result"""
    pass


class CommandExitCodeResponse(object):
    def __init__(self, terminated, exit_code):
        self.terminated = terminated
        """Yes/no"""
        self.exit_code = exit_code
        """Exit code returned from command if terminated, None otherwise"""


class AbortCommandRequest(object):
    """Aborts the command currently running."""
    pass


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
    def __init__(self, name, key, nics, command_env=None, verbose=0):
        super(BasicTaskService, self).__init__(name, key, nics)
        self._initial_registration_complete = False
        self._wait_cond = threading.Condition()
        self._command_env = command_env
        self._command_abort = None
        self._command_exit_code = None
        self._verbose = verbose

        self._command_thread = None
        self._fn_result = None

    def _run_command(self, command, env, event):
        self._command_exit_code = safe_shell_exec.execute(command, env=env, events=[event])

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
                        for key, value in req.env.items():
                            if 'SECRET' in key:
                                value = '*' * len(value)
                            print("Task service env: {} = {}".format(key, value))

                    # We only permit executing exactly one command, so this is idempotent.
                    self._command_abort = threading.Event()
                    self._command_thread = in_thread(
                        target=self._run_command,
                        args=(req.command, req.env, self._command_abort)
                    )
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, AbortCommandRequest):
            self._wait_cond.acquire()
            try:
                if self._command_thread is not None:
                    self._command_abort.set()
            finally:
                self._wait_cond.notify_all()
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
                    self._wait_cond.wait(req.delay if req.delay >= 1.0 else 1.0)
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

    def wait_for_command_start(self, timeout):
        self._wait_cond.acquire()
        try:
            while self._command_thread is None:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('command to run')
        finally:
            self._wait_cond.release()

    def wait_for_command_termination(self):
        self._command_thread.join()


class BasicTaskClient(network.BasicClient):
    def __init__(self, service_name, task_addresses, key, verbose,
                 match_intf=False, attempts=3):
        super(BasicTaskClient, self).__init__(service_name,
                                              task_addresses, key, verbose,
                                              match_intf=match_intf,
                                              attempts=attempts)

    def run_command(self, command, env):
        self._send(RunCommandRequest(command, env))

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
        :return: terminated flag and result tuple
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
        """
        try:
            resp = self._send(WaitForCommandExitCodeRequest(delay))
            return resp.exit_code
        except:
            pass
