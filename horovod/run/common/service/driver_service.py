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


class BasicDriverService(network.BasicService):
    def __init__(self, num_proc, name, key):
        super(BasicDriverService, self).__init__(name, key)
        self._num_proc = num_proc
        self._all_task_addresses = {}
        self._task_addresses_for_driver = {}
        self._task_addresses_for_tasks = {}
        self._task_host_hash_indices = {}
        self._wait_cond = threading.Condition()

    def _handle(self, req, client_address):
        if isinstance(req, RegisterTaskRequest):
            self._wait_cond.acquire()
            try:
                assert 0 <= req.index < self._num_proc
                self._all_task_addresses[req.index] = req.task_addresses
                # Just use source address for service for fast probing.
                self._task_addresses_for_driver[req.index] = \
                    self._filter_by_ip(req.task_addresses, client_address[0])
                if not self._task_addresses_for_driver[req.index]:
                    # No match is possible if one of the servers is behind NAT.
                    # We don't throw exception here, but will allow the following
                    # code fail with NoValidAddressesFound.
                    print('ERROR: Task {index} declared addresses {task_addresses}, '
                          'but has connected from a different address {source}. '
                          'This is not supported. Is the server behind NAT?'
                          ''.format(index=req.index, task_addresses=req.task_addresses,
                                    source=client_address[0]))
                # Make host hash -> indices map.
                if req.host_hash not in self._task_host_hash_indices:
                    self._task_host_hash_indices[req.host_hash] = []
                self._task_host_hash_indices[req.host_hash].append(req.index)
                self._task_host_hash_indices[req.host_hash].sort()
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, RegisterTaskToTaskAddressesRequest):
            self._wait_cond.acquire()
            try:
                assert 0 <= req.index < self._num_proc
                self._task_addresses_for_tasks[req.index] = req.task_addresses
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, AllTaskAddressesRequest):
            return AllTaskAddressesResponse(self._all_task_addresses[req.index])

        return super(BasicDriverService, self)._handle(req, client_address)

    def _filter_by_ip(self, addresses, target_ip):
        for intf, intf_addresses in addresses.items():
            for ip, port in intf_addresses:
                if ip == target_ip:
                    return {intf: [(ip, port)]}
        return {}

    def task_addresses_for_driver(self, index):
        return self._task_addresses_for_driver[index]

    def task_addresses_for_tasks(self, index):
        return self._task_addresses_for_tasks[index]

    def task_host_hash_indices(self):
        return self._task_host_hash_indices

    def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._all_task_addresses) < self._num_proc:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('tasks to start')
        finally:
            self._wait_cond.release()

    def wait_for_task_to_task_address_updates(self, timeout):
        self._wait_cond.acquire()
        try:
            while len(self._task_addresses_for_tasks) < self._num_proc:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for(
                    'tasks to update task-to-task addresses')
        finally:
            self._wait_cond.release()


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


class BasicDriverClient(network.BasicClient):
    def __init__(self, name, driver_addresses, key, verbose, match_intf=False):
        super(BasicDriverClient, self).__init__(name,
                                                driver_addresses,
                                                key,
                                                verbose,
                                                match_intf=match_intf)

    def register_task(self, index, task_addresses, host_hash):
        self._send(RegisterTaskRequest(index, task_addresses, host_hash))

    def all_task_addresses(self, index):
        resp = self._send(AllTaskAddressesRequest(index))
        return resp.all_task_addresses

    def register_task_to_task_addresses(self, index, task_addresses):
        self._send(RegisterTaskToTaskAddressesRequest(index, task_addresses))
