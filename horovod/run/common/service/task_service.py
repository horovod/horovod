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
import time

from horovod.run.common.util import network
from horovod.run.common.util import safe_shell_exec


class RunCommandRequest(object):
    def __init__(self, command, env):
        self.command = command
        """Command to run."""
        self.env = env
        """Environment to use."""


class CommandTerminatedRequest(object):
    """Is command execution finished?"""
    pass


class CommandTerminatedResponse(object):
    def __init__(self, flag):
        self.flag = flag
        """Yes/no"""


class NotifyInitialRegistrationCompleteRequest(object):
    """Notification that initial task registration has completed."""
    pass


class RegisterCodeResultRequest(object):
    """Register code execution results with task."""

    def __init__(self, result):
        self.result = result


class BasicTaskService(network.BasicService):
    def __init__(self, name, key):
        super(BasicTaskService, self).__init__(name, key)
        self._initial_registration_complete = False
        self._wait_cond = threading.Condition()
        self._command_thread = None
        self._fn_result = None

    def _handle(self, req, client_address):
        if isinstance(req, RunCommandRequest):
            self._wait_cond.acquire()
            try:
                if self._command_thread is None:
                    # We only permit executing exactly one command, so this is idempotent.
                    self._command_thread = threading.Thread(
                        target=safe_shell_exec.execute,
                        args=(req.command, req.env))
                    self._command_thread.daemon = True
                    self._command_thread.start()
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

        if isinstance(req, CommandTerminatedRequest):
            self._wait_cond.acquire()
            try:
                terminated = (self._command_thread is not None and
                              not self._command_thread.is_alive())
            finally:
                self._wait_cond.release()
            return CommandTerminatedResponse(terminated)

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
                 match_intf=False, retries=3):
        super(BasicTaskClient, self).__init__(service_name,
                                              task_addresses, key, verbose,
                                              match_intf=match_intf,
                                              retries=retries)

    def run_command(self, command, env):
        self._send(RunCommandRequest(command, env))

    def notify_initial_registration_complete(self):
        self._send(NotifyInitialRegistrationCompleteRequest())

    def command_terminated(self):
        resp = self._send(CommandTerminatedRequest())
        return resp.flag

    def register_code_result(self, result):
        self._send(RegisterCodeResultRequest(result))

    def wait_for_command_termination(self, delay=1):
        try:
            while True:
                if self.command_terminated():
                    break
                time.sleep(delay)
        except:
            pass
