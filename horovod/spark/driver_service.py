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


class RegisterRequest(object):
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


class RegisterResponse(object):
    pass


class TaskAddressesRequest(object):
    def __init__(self, index):
        self.index = index


# TODO: clarify all these various task addresses
class TaskAddressesResponse(object):
    def __init__(self, all_task_addresses):
        self.all_task_addresses = all_task_addresses


class TaskHostHashIndicesRequest(object):
    def __init__(self, host_hash):
        self.host_hash = host_hash


class TaskHostHashIndicesResponse(object):
    def __init__(self, indices):
        self.indices = indices


class TaskIndexByRankRequest(object):
    def __init__(self, rank):
        self.rank = rank


class TaskIndexByRankResponse(object):
    def __init__(self, index):
        self.index = index


class CodeRequest(object):
    pass


class CodeResponse(object):
    def __init__(self, fn, args, kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class DriverService(BasicService):
    NAME = 'driver service'

    def __init__(self, num_proc, fn, args, kwargs):
        super(DriverService, self).__init__(DriverService.NAME)
        self._num_proc = num_proc
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._all_task_addresses = {}
        self._task_addresses_for_driver = {}
        self._task_addresses_for_tasks = {}
        self._task_host_hash_indices = {}
        self._ranks_to_indices = None
        self._wait_cond = threading.Condition()

    def _handle(self, req, client_address):
        if isinstance(req, RegisterRequest):
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
            return RegisterResponse()

        if isinstance(req, RegisterTaskToTaskAddressesRequest):
            self._wait_cond.acquire()
            try:
                assert 0 <= req.index < self._num_proc
                self._task_addresses_for_tasks[req.index] = req.task_addresses
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return RegisterResponse()

        if isinstance(req, TaskAddressesRequest):
            return TaskAddressesResponse(self._all_task_addresses[req.index])

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

    def all_task_addresses(self, index):
        return self._all_task_addresses[index]

    def task_addresses_for_driver(self, index):
        return self._task_addresses_for_driver[index]

    def task_addresses_for_tasks(self, index):
        return self._task_addresses_for_tasks[index]

    def task_host_hash_indices(self):
        return self._task_host_hash_indices

    def set_ranks_to_indices(self, ranks_to_indices):
        self._ranks_to_indices = ranks_to_indices

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._all_task_addresses) < self._num_proc:
                self._wait_cond.wait(timeout.remaining())
                if timeout.timed_out():
                    raise Exception('Timed out waiting for tasks to start.')
        finally:
            self._wait_cond.release()

    def wait_for_task_to_task_address_updates(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._task_addresses_for_tasks) < self._num_proc:
                self._wait_cond.wait(timeout.remaining())
                if timeout.timed_out():
                    raise Exception('Timed out waiting for tasks to update '
                                    'task-to-task addresses.')
        finally:
            self._wait_cond.release()


class DriverClient(BasicClient):
    def __init__(self, driver_addresses):
        super(DriverClient, self).__init__(DriverService.NAME, driver_addresses)

    def register(self, index, task_addresses, host_hash):
        self._send(RegisterRequest(index, task_addresses, host_hash))

    def all_task_addresses(self, index):
        resp = self._send(TaskAddressesRequest(index))
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
