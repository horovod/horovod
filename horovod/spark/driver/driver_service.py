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

from horovod.runner.common.service import driver_service
from horovod.runner.common.util import network


class TaskHostHashIndicesRequest(object):
    """Request task indices for a given host hash."""
    def __init__(self, host_hash):
        self.host_hash = host_hash


class TaskHostHashIndicesResponse(object):
    def __init__(self, indices):
        self.indices = indices
        """Task indices."""


class SetLocalRankToRankRequest(object):
    """Set local rank to rank."""
    def __init__(self, host_hash, local_rank, rank):
        self.host = host_hash
        """Host hash."""
        self.local_rank = local_rank
        """Local rank."""
        self.rank = rank
        """Rank for local rank on host."""


class SetLocalRankToRankResponse(object):
    def __init__(self, index):
        self.index = index
        """Index for rank given in request."""


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


class WaitForTaskShutdownRequest(object):
    """Request that blocks until all task services should shut down."""
    pass


class SparkDriverService(driver_service.BasicDriverService):
    NAME = 'driver service'

    def __init__(self, initial_np, num_proc, fn, args, kwargs, key, nics):
        super(SparkDriverService, self).__init__(num_proc,
                                                 SparkDriverService.NAME,
                                                 key, nics)
        self._initial_np = initial_np
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._key = key
        self._nics = nics
        self._ranks_to_indices = {}
        self._spark_job_failed = False
        self._lock = threading.Lock()
        self._task_shutdown = threading.Event()

    def _handle(self, req, client_address):

        if isinstance(req, TaskHostHashIndicesRequest):
            return TaskHostHashIndicesResponse(self._task_host_hash_indices[req.host_hash])

        if isinstance(req, SetLocalRankToRankRequest):
            self._lock.acquire()

            try:
                # get index for host and local_rank
                indices = self._task_host_hash_indices[req.host]
                index = indices[req.local_rank]

                # remove earlier rank for this index
                # dict.keys() and dict.values() have corresponding order
                # so we look up index in _ranks_to_indices.values() and use that position
                # to get the corresponding key (the rank) from _ranks_to_indices.keys()
                # https://stackoverflow.com/questions/835092/python-dictionary-are-keys-and-values-always-the-same-order
                values = list(self._ranks_to_indices.values())
                prev_pos = values.index(index) if index in values else None
                if prev_pos is not None:
                    prev_rank = list(self._ranks_to_indices.keys())[prev_pos]
                    del self._ranks_to_indices[prev_rank]

                # memorize rank's index
                self._ranks_to_indices[req.rank] = index
            finally:
                self._lock.release()
            return SetLocalRankToRankResponse(index)

        if isinstance(req, TaskIndexByRankRequest):
            self._lock.acquire()
            try:
                return TaskIndexByRankResponse(self._ranks_to_indices[req.rank])
            finally:
                self._lock.release()

        if isinstance(req, CodeRequest):
            return CodeResponse(self._fn, self._args, self._kwargs)

        if isinstance(req, WaitForTaskShutdownRequest):
            self._task_shutdown.wait()
            return network.AckResponse()

        return super(SparkDriverService, self)._handle(req, client_address)

    def set_ranks_to_indices(self, ranks_to_indices):
        self._lock.acquire()
        try:
            self._ranks_to_indices = ranks_to_indices.copy()
        finally:
            self._lock.release()

    def get_ranks_to_indices(self):
        self._lock.acquire()
        try:
            return self._ranks_to_indices.copy()
        finally:
            self._lock.release()

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
            while len(self._all_task_addresses) < self._initial_np:
                self.check_for_spark_job_failure()
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('Spark tasks to start')
        finally:
            self._wait_cond.release()

    def wait_for_task_to_task_address_updates(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._task_addresses_for_tasks) < self._initial_np:
                self.check_for_spark_job_failure()
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('Spark tasks to update task-to-task addresses')
        finally:
            self._wait_cond.release()

    def get_common_interfaces(self):
        if self._nics is not None:
            return self._nics

        nics = None
        if len(self._task_addresses_for_tasks) > 0:
            # in Elastic Horovod on Spark with auto-scaling
            # keys in task_addresses are in range(max_np or proc_num)
            # but not all keys may exist, so we don't do for index in range(proc_num)
            indices = list(self._task_addresses_for_tasks.keys())
            nics = set(self._task_addresses_for_tasks[indices[0]].keys())
            for index in indices[1:]:
                nics.intersection_update(self._task_addresses_for_tasks[index].keys())

        if not nics:
            raise Exception('Unable to find a set of common task-to-task communication interfaces: %s'
                            % [(index, self._task_addresses_for_tasks[index])
                               for index in self._task_addresses_for_tasks])

        return nics

    def shutdown_tasks(self):
        self._task_shutdown.set()

    def shutdown(self):
        self.shutdown_tasks()
        super(SparkDriverService, self).shutdown()


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

    def set_local_rank_to_rank(self, host_hash, local_rank, rank):
        resp = self._send(SetLocalRankToRankRequest(host_hash, local_rank, rank))
        return resp.index

    def task_index_by_rank(self, rank):
        resp = self._send(TaskIndexByRankRequest(rank))
        return resp.index

    def code(self):
        resp = self._send(CodeRequest())
        return resp.fn, resp.args, resp.kwargs

    def wait_for_task_shutdown(self):
        self._send(WaitForTaskShutdownRequest())
