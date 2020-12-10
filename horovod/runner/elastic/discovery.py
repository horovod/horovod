# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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

import io
import logging
import threading

from collections import defaultdict

from horovod.runner.common.util import safe_shell_exec
from horovod.runner.elastic.worker import HostUpdateResult


class HostState(object):
    def __init__(self):
        self._event = threading.Event()

        # TODO(travis): blacklisted hosts should have a timeout period that increases with each failure
        self._blacklisted = False

    def get_event(self):
        if self._event.is_set():
            event = threading.Event()
            self._event = event
        return self._event

    def set_event(self):
        self._event.set()

    def blacklist(self):
        self._blacklisted = True
        self.set_event()

    def is_blacklisted(self):
        return self._blacklisted


class DiscoveredHosts(object):
    def __init__(self, host_slots, host_assignment_order):
        self._host_slots = host_slots
        self._host_assignment_order = host_assignment_order

    @property
    def host_slots(self):
        return self._host_slots

    @property
    def available_hosts(self):
        return set(self._host_assignment_order)

    @property
    def host_assignment_order(self):
        return self._host_assignment_order

    def get_slots(self, host):
        return self._host_slots.get(host, 0)

    def count_available_slots(self):
        # Use the host_assignment_order as it does not contain blacklisted hosts
        return sum([self.get_slots(host) for host in self._host_assignment_order])

    def update(self, hosts_state):
        self._host_assignment_order = [host for host in self._host_assignment_order
                                       if not hosts_state[host].is_blacklisted()]
        return self


class HostManager(object):
    def __init__(self, discovery):
        self._current_hosts = DiscoveredHosts(host_slots={}, host_assignment_order=[])
        self._hosts_state = defaultdict(HostState)
        self._discovery = discovery

    def update_available_hosts(self):
        # TODO(travis): also check for hosts removed from the blacklist in the future
        def check_update(cur_host_slots, prev_host_slots):
            res = HostUpdateResult.no_update

            for prev_h in prev_host_slots:
                if prev_h not in cur_host_slots:
                    # prev_h is a removed host
                    res |= HostUpdateResult.removed

            for h in cur_host_slots:
                if h not in prev_host_slots:
                    # h is an added host
                    res |= HostUpdateResult.added
                elif cur_host_slots[h] > prev_host_slots[h]:
                    # h has more slots added
                    res |= HostUpdateResult.added
                elif cur_host_slots[h] < prev_host_slots[h]:
                    # h has removed some slots
                    res |=  HostUpdateResult.removed
            return res

        prev_host_slots = self._current_hosts.host_slots
        prev_host_assignment_order = self._current_hosts.host_assignment_order
        host_slots = self._discovery.find_available_hosts_and_slots()
        if prev_host_slots != host_slots:
            available_hosts = set([host for host in host_slots.keys() if not self._hosts_state[host].is_blacklisted()])
            host_assignment_order = HostManager.order_available_hosts(available_hosts, prev_host_assignment_order)
            self._current_hosts = DiscoveredHosts(host_slots=host_slots,
                                                  host_assignment_order=host_assignment_order)
            return check_update(self._current_hosts.host_slots, prev_host_slots)
        else:
            return HostUpdateResult.no_update

    @property
    def current_hosts(self):
        return self._current_hosts.update(self._hosts_state)

    def blacklist(self, host):
        if not self._hosts_state[host].is_blacklisted():
            logging.warning('blacklist failing host: {}'.format(host))
        self._hosts_state[host].blacklist()

    def is_blacklisted(self, host):
        return self._hosts_state[host].is_blacklisted()

    def get_host_event(self, host):
        return self._hosts_state[host].get_event()

    @staticmethod
    def order_available_hosts(available_hosts, prev_host_assignment_order):
        # We need to ensure this list preserves relative order to ensure the oldest hosts are assigned lower ranks.
        host_assignment_order = [host for host in prev_host_assignment_order if host in available_hosts]
        known_hosts = set(host_assignment_order)
        for host in available_hosts:
            if host not in known_hosts:
                host_assignment_order.append(host)
        return host_assignment_order


class HostDiscovery(object):
    def find_available_hosts_and_slots(self):
        """Returns a dict mapping <hostname> -> <number of slots>."""
        raise NotImplementedError()


class HostDiscoveryScript(HostDiscovery):
    def __init__(self, discovery_script, slots):
        self._discovery_script = discovery_script
        self._default_slots = slots
        super(HostDiscoveryScript, self).__init__()

    def find_available_hosts_and_slots(self):
        stdout = io.StringIO()
        exit_code = safe_shell_exec.execute(self._discovery_script, stdout=stdout)
        if exit_code != 0:
            raise RuntimeError('Failed to execute discovery script: {}. Exit code: {}'
                               .format(self._discovery_script, exit_code))

        host_slots = {}
        lines = set(stdout.getvalue().strip().split('\n'))
        for line in lines:
            host = line
            if ':' in line:
                host, slots = line.split(':')
                host_slots[host] = int(slots)
            else:
                host_slots[host] = self._default_slots
        return host_slots


class FixedHosts(HostDiscovery):
    def __init__(self, host_slots):
        super(FixedHosts, self).__init__()
        self._host_slots = host_slots

    def find_available_hosts_and_slots(self):
        return self._host_slots

    def set(self, host_slots):
        self._host_slots = host_slots
