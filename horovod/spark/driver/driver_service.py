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

from horovod.spark.util.network import BasicService, BasicClient, AckResponse


class RegisterTaskRequest(object):
    def __init__(self, index, task_addresses, host_hash):
        self.index = index
        """Task index."""

        self.task_addresses = task_addresses
        """Map of interface to list of (ip, port) pairs."""

        self.host_hash = host_hash
        """
        Hash of the host that helps to determine which tasks
        have shared memory access to each other.
        """


class RegisterTaskToTaskAddressesRequest(object):
    def __init__(self, index, task_addresses):
        self.index = index
        """Task index."""

        self.task_addresses = task_addresses
        """Map of interface to list of (ip, port) pairs."""


class AllTaskAddressesRequest(object):
    """Request all task addresses for a given index."""
    def __init__(self, index):
        self.index = index


class AllTaskAddressesResponse(object):
    def __init__(self, all_task_addresses):
        self.all_task_addresses = all_task_addresses
        """Map of interface to list of (ip, port) pairs."""


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


class DriverService(BasicService):
    NAME = 'driver service'

    def __init__(self, num_proc, fn, args, kwargs, key):
        super(DriverService, self).__init__(DriverService.NAME, key)
        self._num_proc = num_proc
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._all_task_addresses = {}
        self._task_addresses_for_driver = {}
        self._task_addresses_for_tasks = {}
        self._task_host_hash_indices = {}
        self._ranks_to_indices = None
        self._spark_job_failed = False
        self._wait_cond = threading.Condition()

    def _handle(self, req, client_address):
        if isinstance(req, RegisterTaskRequest):
            self._wait_cond.acquire()
            try:
                assert 0 <= req.index < self._num_proc
                self._all_task_addresses[req.index] = req.task_addresses
                # Just use source address for driver for fast probing.
                self._task_addresses_for_driver[req.index] = \
                    self._filter_by_ip(req.task_addresses, client_address[0])
                # Make host hash -> indices map.
                if req.host_hash not in self._task_host_hash_indices:
                    self._task_host_hash_indices[req.host_hash] = []
                self._task_host_hash_indices[req.host_hash].append(req.index)
                self._task_host_hash_indices[req.host_hash].sort()
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return AckResponse()

        if isinstance(req, RegisterTaskToTaskAddressesRequest):
            self._wait_cond.acquire()
            try:
                assert 0 <= req.index < self._num_proc
                self._task_addresses_for_tasks[req.index] = req.task_addresses
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return AckResponse()

        if isinstance(req, AllTaskAddressesRequest):
            return AllTaskAddressesResponse(self._all_task_addresses[req.index])

        if isinstance(req, TaskHostHashIndicesRequest):
            return TaskHostHashIndicesResponse(self._task_host_hash_indices[req.host_hash])

        if isinstance(req, TaskIndexByRankRequest):
            return TaskIndexByRankResponse(self._ranks_to_indices[req.rank])

        if isinstance(req, CodeRequest):
            return CodeResponse(self._fn, self._args, self._kwargs)

        return super(DriverService, self)._handle(req, client_address)

    def _filter_by_ip(self, addresses, target_ip):
        for intf, intf_addresses in addresses.items():
            for ip, port in intf_addresses:
                if ip == target_ip:
                    return {intf: [(ip, port)]}

    def task_addresses_for_driver(self, index):
        return self._task_addresses_for_driver[index]

    def task_addresses_for_tasks(self, index):
        return self._task_addresses_for_tasks[index]

    def task_host_hash_indices(self):
        return self._task_host_hash_indices

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


class DriverClient(BasicClient):
    def __init__(self, driver_addresses, key, match_intf=False):
        super(DriverClient, self).__init__(DriverService.NAME, driver_addresses, key,
                                           match_intf=match_intf)

    def register_task(self, index, task_addresses, host_hash):
        self._send(RegisterTaskRequest(index, task_addresses, host_hash))

    def all_task_addresses(self, index):
        resp = self._send(AllTaskAddressesRequest(index))
        return resp.all_task_addresses

    def register_task_to_task_addresses(self, index, task_addresses):
        self._send(RegisterTaskToTaskAddressesRequest(index, task_addresses))

    def task_host_hash_indices(self, host_hash):
        resp = self._send(TaskHostHashIndicesRequest(host_hash))
        return resp.indices

    def task_index_by_rank(self, rank):
        resp = self._send(TaskIndexByRankRequest(rank))
        return resp.index

    def code(self):
        resp = self._send(CodeRequest())
        return resp.fn, resp.args, resp.kwargs
