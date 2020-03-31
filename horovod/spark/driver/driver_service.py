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

from horovod.run.common.service import driver_service


class TaskHostHashIndicesRequest(object):
    """Request task indices for a given host hash."""
    def __init__(self, host_hash):
        self.host_hash = host_hash


class TaskHostHashIndicesResponse(object):
    def __init__(self, indices):
        self.indices = indices
        """Task indices."""


class TaskIndexByRankRequest(object):
    """Request task index by Horovod rank."""
    def __init__(self, rank):
        self.rank = rank


class TaskIndexByRankResponse(object):
    def __init__(self, index):
        self.index = index
        """Task index."""


class CodeRequest(object):
    """Request Python function to execute."""
    pass


class CodeResponse(object):
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        """Function."""

        self.args = args
        """Function args."""

        self.kwargs = kwargs
        """Function kwargs."""


class SparkDriverService(driver_service.BasicDriverService):
    NAME = 'driver service'

    def __init__(self, num_proc, fn, args, kwargs, key, nics):
        super(SparkDriverService, self).__init__(num_proc,
                                                 SparkDriverService.NAME,
                                                 key, nics)

        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._ranks_to_indices = None
        self._spark_job_failed = False

    def _handle(self, req, client_address):

        if isinstance(req, TaskHostHashIndicesRequest):
            return TaskHostHashIndicesResponse(self._task_host_hash_indices[req.host_hash])

        if isinstance(req, TaskIndexByRankRequest):
            return TaskIndexByRankResponse(self._ranks_to_indices[req.rank])

        if isinstance(req, CodeRequest):
            return CodeResponse(self._fn, self._args, self._kwargs)

        return super(SparkDriverService, self)._handle(req, client_address)

    def set_ranks_to_indices(self, ranks_to_indices):
        self._ranks_to_indices = ranks_to_indices

    def notify_spark_job_failed(self):
        self._wait_cond.acquire()
        try:
            self._spark_job_failed = True
        finally:
            self._wait_cond.notify_all()
            self._wait_cond.release()

    def check_for_spark_job_failure(self):
        if self._spark_job_failed:
            raise Exception('Spark job has failed, see the error above.')

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._all_task_addresses) < self._num_proc:
                self.check_for_spark_job_failure()
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('Spark tasks to start')
        finally:
            self._wait_cond.release()

    def wait_for_task_to_task_address_updates(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._task_addresses_for_tasks) < self._num_proc:
                self.check_for_spark_job_failure()
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('Spark tasks to update task-to-task addresses')
        finally:
            self._wait_cond.release()


class SparkDriverClient(driver_service.BasicDriverClient):
    def __init__(self, driver_addresses, key, verbose, match_intf=False):
        super(SparkDriverClient, self).__init__(SparkDriverService.NAME,
                                                driver_addresses,
                                                key,
                                                verbose,
                                                match_intf=match_intf)

    def task_host_hash_indices(self, host_hash):
        resp = self._send(TaskHostHashIndicesRequest(host_hash))
        return resp.indices

    def task_index_by_rank(self, rank):
        resp = self._send(TaskIndexByRankRequest(rank))
        return resp.index

    def code(self):
        resp = self._send(CodeRequest())
        return resp.fn, resp.args, resp.kwargs
