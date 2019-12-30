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

from __future__ import absolute_import

import os
import threading

from collections import defaultdict

from horovod.run.common.util import hosts, timeout


READY = 'READY'
SUCCESS = 'SUCCESS'
FAILURE = 'FAILURE'


class WorkersRecordedBarrier(object):
    def __init__(self, driver):
        self._driver = driver
        self._barrier = None
        self._lock = threading.Lock()
        self._states = {}
        self._action = self._action

    def reset(self, size):
        with self._lock:
            self._states.clear()
            self._barrier = threading.Barrier(parties=size, action=self._action)

    def record_ready(self, host, slot):
        self._record_state(host, slot, READY)

    def record_success(self, host, slot):
        self._record_state(host, slot, SUCCESS)

    def record_failure(self, host, slot):
        self._record_state(host, slot, FAILURE)

    def _record_state(self, host, slot, state):
        key = (host, slot)
        with self._lock:
            if key in self._states:
                self._barrier.reset()
            self._states[(host, slot)] = state

        while True:
            try:
                self._barrier.wait()
                return
            except threading.BrokenBarrierError:
                if self._barrier.broken():
                    # Timeout or other non-recoverable error, so exit
                    raise

                with self._lock:
                    saved_state = self._states.get(key, state)
                    if saved_state != state:
                        raise RuntimeError('State {} overridden by {}'.format(state, saved_state))

    def _action(self):
        self._driver.on_workers_recorded()
        self.reset(self._driver.world_size())


class ElasticDriver(object):
    def __init__(self, min_np, max_np, slots, start_timeout=None):
        self._min_np = min_np
        self._max_np = max_np
        self._slots = slots

        self._available_hosts = []
        self._blacklisted_hosts = set()
        self._assigned_hosts = set()
        self._host_assignments = {}
        self._world_size = 0

        self._wait_hosts_cond = threading.Condition()
        self._start_timeout = start_timeout or int(os.getenv('HOROVOD_ELASTIC_START_TIMEOUT', '600'))

        self._discovery_thread = threading.Thread(target=self._discover_hosts)
        self._discovery_thread.daemon = True

        self._create_worker_fn = None

        self._barrier = WorkersRecordedBarrier(self)
        self._shutdown = threading.Event()

    def start(self, create_worker_fn):
        self._discovery_thread.start()
        self._create_worker_fn = create_worker_fn
        self._activate_hosts()

    def stop(self):
        self._shutdown.set()

    def record_ready(self, host, slot):
        self._barrier.record_ready(host, slot)

    def on_workers_recorded(self):
        self._activate_hosts()

    def world_size(self):
        return self._world_size

    def local_size(self, host):
        return len(self._host_assignments[host])

    def get_slot_info(self, host, slot):
        return self._host_assignments[host][slot]

    def _activate_hosts(self):
        self._wait_for_available_hosts()
        new_assigned_hosts = self._update_assigned_hosts()
        for host in new_assigned_hosts:
            self._start_worker_processes(host)

    def _wait_for_available_hosts(self):
        tmout = timeout.Timeout(
            self._start_timeout,
            message='Timed out waiting for {{activity}}. Please check that you have '
                    'enough resources to run at least {min_np} Horovod processes.'.format(min_np=self._min_np))

        self._wait_hosts_cond.acquire()
        try:
            while len(self._available_hosts) < self._min_np:
                self._wait_hosts_cond.wait(tmout.remaining())
                tmout.check_time_out_for('minimum number of hosts to become available')
        finally:
            self._wait_hosts_cond.release()

    def _discover_hosts(self):
        while not self._shutdown.is_set():
            self._wait_hosts_cond.acquire()
            try:
                if self._update_available_hosts():
                    self._wait_hosts_cond.notify_all()
            finally:
                self._wait_hosts_cond.release()
            self._shutdown.wait(1.0)

    def _update_available_hosts(self):
        updated = False
        current_hosts = set(self._available_hosts)
        all_hosts = self._find_all_hosts()
        for host in all_hosts:
            if host not in current_hosts:
                self._available_hosts.append(host)
                updated = True
        return updated

    def _find_all_hosts(self):
        # TODO
        return []

    def _update_assigned_hosts(self):
        self._assigned_hosts = {host for host in self._assigned_hosts if host not in self._blacklisted_hosts}
        new_assigned_hosts = []
        for host in self._available_hosts:
            if host not in self._assigned_hosts and host not in self._blacklisted_hosts:
                new_assigned_hosts.append(host)
                self._assigned_hosts.add(host)
        self._update_host_assignments()
        return new_assigned_hosts

    def _update_host_assignments(self):
        host_list = [hosts.HostInfo(host, self._get_slots(host))
                     for host in self._available_hosts if host in self._assigned_hosts]
        host_assignments_list = hosts.get_host_assignments(host_list, self._min_np, self._max_np)
        host_assignments = defaultdict(list)
        for slot_info in host_assignments_list:
            host_assignments[slot_info.hostname].append(slot_info)
        self._host_assignments = host_assignments
        self._world_size = len(host_assignments_list)

    def _get_slots(self, host):
        # TODO: support per host slots
        return self._slots

    def _start_worker_processes(self, host):
        for slot_info in self._host_assignments[host]:
            self._start_worker_process(slot_info)

    def _start_worker_process(self, slot_info):
        create_worker_fn = self._create_worker_fn

        def run_worker():
            res = create_worker_fn(slot_info)
            exit_code, timestamp = res
            self._handle_worker_exit(slot_info, exit_code)

        thread = threading.Thread(target=run_worker)
        thread.daemon = True
        thread.start()

    def _handle_worker_exit(self, slot_info, exit_code):
        name = '{}:{}'.format(slot_info.hostname, slot_info.local_rank)
        if exit_code == 0:
            # Successful exit means training process is complete
            self._barrier.record_success(slot_info.hostname, slot_info.local_rank)
            return

        self._barrier.record_failure(slot_info.hostname, slot_info.local_rank)
