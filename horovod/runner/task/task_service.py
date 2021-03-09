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

from horovod.runner.common.service import task_service


class TaskToTaskAddressCheckFinishedSignal(object):
    def __init__(self, index):
        self.index = index
        """Task index."""


class TaskToTaskAddressCheckFinishedSignalResponse(object):
    def __init__(self, index):
        self.index = index
        """Task index."""


class HorovodRunTaskService(task_service.BasicTaskService):
    NAME_FORMAT = 'horovod task service #%d'

    def __init__(self, index, key, nics):
        super(HorovodRunTaskService, self).__init__(
            HorovodRunTaskService.NAME_FORMAT % index,
            index, key, nics)
        self.index = index
        self._task_to_task_address_check_completed = False

    def _handle(self, req, client_address):

        if isinstance(req, TaskToTaskAddressCheckFinishedSignal):
            self._wait_cond.acquire()
            try:
                self._task_to_task_address_check_completed = True
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()

            return TaskToTaskAddressCheckFinishedSignalResponse(self.index)

        return super(HorovodRunTaskService, self)._handle(req, client_address)

    def wait_for_task_to_task_address_check_finish_signal(self, timeout):
        self._wait_cond.acquire()
        try:
            while not self._task_to_task_address_check_completed:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('Task to task address check')
        finally:
            self._wait_cond.release()


class HorovodRunTaskClient(task_service.BasicTaskClient):

    def __init__(self, index, task_addresses, key, verbose, match_intf=False, attempts=3):
        super(HorovodRunTaskClient, self).__init__(
            HorovodRunTaskService.NAME_FORMAT % index,
            task_addresses, key, verbose,
            match_intf=match_intf,
            attempts=attempts)
        self.index = index

    def task_to_task_address_check_completed(self):
        resp = self._send(TaskToTaskAddressCheckFinishedSignal(self.index))
        return resp.index
