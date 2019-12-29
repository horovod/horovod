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


class ElasticDriver(object):
    def __init__(self, min_np, max_np, slots, start_timeout=None):
        self._min_np = min_np
        self._max_np = max_np
        self._slots = slots

        self._available_hosts = []
        self._blacklisted_hosts = set()
        self._active_hosts = set()
        self._host_assignments = {}

        self._wait_hosts_cond = threading.Condition()
        self._start_timeout = start_timeout or int(os.getenv('HOROVOD_ELASTIC_START_TIMEOUT', '600'))

        self._discovery_thread = threading.Thread(target=self._discover_hosts)
        self._discovery_thread.daemon = True

        self._shutdown = threading.Event()

    def start(self, create_worker_fn):
        self._discovery_thread.start()
        self._wait_for_available_hosts()
        for host in self._available_hosts:
            self._start_worker_processes(host, create_worker_fn)

    def stop(self):
        self._shutdown.set()

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
            if host not in current_hosts and host not in self._blacklisted_hosts:
                self._available_hosts.append(host)
                updated = True

        if updated:
            self._update_host_assignments()

        return updated

    def _find_all_hosts(self):
        # TODO
        return []

    def _update_host_assignments(self):
        host_list = [hosts.HostInfo(host, self._get_slots(host)) for host in self._available_hosts]
        host_assignments_list = hosts.get_host_assignments(host_list, self._min_np, self._max_np)
        host_assignments = defaultdict(list)
        for slot_info in host_assignments_list:
            host_assignments[slot_info.hostname].append(slot_info)
        self._host_assignments = host_assignments

    def _get_slots(self, host):
        # TODO: support per host slots
        return self._slots

    def _start_worker_processes(self, host, create_worker_fn):
        for slot_info in self._host_assignments[host]:
            create_worker_fn(slot_info)
        self._active_hosts.add(host)
