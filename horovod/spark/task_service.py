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

import threading

from horovod.spark.network import BasicService, BasicClient
from horovod.spark import safe_shell_exec


class RunCommandRequest(object):
    def __init__(self, command):
        self.command = command
        """Command to run."""


class RunCommandResponse(object):
    pass


class InitialRegistrationCompleteRequest(object):
    pass


class InitialRegistrationCompleteResponse(object):
    pass


class CodeResultRequest(object):
    def __init__(self, result):
        self.result = result


class CodeResultResponse(object):
    pass


class TaskService(BasicService):
    NAME_FORMAT = 'task service #%d'

    def __init__(self, index):
        super(TaskService, self).__init__(TaskService.NAME_FORMAT % index)
        self._initial_registration_complete = False
        self._wait_cond = threading.Condition()
        self._command_thread = None
        self._fn_result = None

    def _handle(self, req, client_address):
        if isinstance(req, RunCommandRequest):
            self._wait_cond.acquire()
            if self._command_thread is None:
                # We only permit executing exactly one command, so this is idempotent.
                self._command_thread = threading.Thread(target=safe_shell_exec.execute,
                                                        args=(req.command,))
                self._command_thread.daemon = True
                self._command_thread.start()
            self._wait_cond.notify_all()
            self._wait_cond.release()
            return RunCommandResponse()

        if isinstance(req, InitialRegistrationCompleteRequest):
            self._wait_cond.acquire()
            self._initial_registration_complete = True
            self._wait_cond.notify_all()
            self._wait_cond.release()
            return InitialRegistrationCompleteResponse()

        if isinstance(req, CodeResultRequest):
            self._fn_result = req.result
            return CodeResultResponse()

        return super(TaskService, self)._handle(req, client_address)

    def fn_result(self):
        return self._fn_result

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        while not self._initial_registration_complete:
            self._wait_cond.wait(timeout.remaining())
            if timeout.timed_out():
                raise Exception('Timed out waiting for tasks to start.')
        self._wait_cond.release()

    def wait_for_command_start(self, timeout):
        self._wait_cond.acquire()
        while self._command_thread is None:
            self._wait_cond.wait(timeout.remaining())
            if timeout.timed_out():
                raise Exception('Timed out waiting for command to run.')
        self._wait_cond.release()

    def wait_for_command_termination(self):
        self._command_thread.join()


class TaskClient(BasicClient):
    def __init__(self, index, task_addresses):
        super(TaskClient, self).__init__(TaskService.NAME_FORMAT % index,
                                         task_addresses)

    def run_command(self, command):
        self._send(RunCommandRequest(command))

    def notify_initial_registration_complete(self):
        self._send(InitialRegistrationCompleteRequest())

    def send_code_result(self, result):
        self._send(CodeResultRequest(result))
