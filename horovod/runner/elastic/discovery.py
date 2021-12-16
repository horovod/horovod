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
import random
import threading
import time
from collections import defaultdict

from horovod.runner.common.util import safe_shell_exec
from horovod.runner.elastic.worker import HostUpdateResult

# The default lower bound for cooldown period. If a range is provided,
# the provided lower limit must be at or above this lower bound
DEFAULT_COOLDOWN_LOWER_LIMIT_SECONDS = 1
# The default upper bound for cooldown period. If a range is provided,
# the provided upper limit must be at or below this upper bound
DEFAULT_COOLDOWN_UPPER_LIMIT_SECONDS = 1 * 60 * 60

class HostState(object):

    def __init__(self, cooldown_range=None):
        self._event = threading.Event()

        self._blacklisted = False
        self._blacklist_count = 0
        if cooldown_range:
            HostState._validate_cooldown_range(cooldown_range)
            self._cooldown_lower_limit, self._cooldown_upper_limit = cooldown_range
        else:
            self._cooldown_lower_limit = -1
            self._cooldown_upper_limit = -1
        self._cooldown_period_end_ts = 0

    @staticmethod
    def _validate_cooldown_range(cooldown_range):
        cooldown_lower_limit, cooldown_upper_limit = cooldown_range

        if (cooldown_lower_limit < DEFAULT_COOLDOWN_LOWER_LIMIT_SECONDS):
            raise ValueError(f"Provided cooldown lower limit: {cooldown_lower_limit} \
                             cannot be lower than default cooldown lower limit: {DEFAULT_COOLDOWN_LOWER_LIMIT_SECONDS}")


        if (cooldown_upper_limit > DEFAULT_COOLDOWN_UPPER_LIMIT_SECONDS):
            raise ValueError(f"Provided cooldown upper limit: {cooldown_upper_limit} \
                             cannot be higher than default cooldown upper limit: {DEFAULT_COOLDOWN_UPPER_LIMIT_SECONDS}")

    def get_event(self):
        if self._event.is_set():
            event = threading.Event()
            self._event = event
        return self._event

    def set_event(self):
        self._event.set()

    def _in_cooldown_period(self, current_time):
        return self._cooldown_period_end_ts > current_time


    def _set_cooldown_period(self, current_time):
        if self._cooldown_lower_limit == -1 or self._cooldown_upper_limit == -1:
            return
        self._blacklist_count += 1

        cooldown_delay = self._cooldown_lower_limit * (1 << self._blacklist_count) + (random.uniform(0,1) * self._cooldown_lower_limit)
        logging.debug(f"{self._blacklist_count}:{self._cooldown_period_end_ts} cooldown_delay: {cooldown_delay}")
        # We need to ensure that the cooldown upper limit is the upper bound of the delay
        cooldown_delta_seconds = max(self._cooldown_lower_limit, min(self._cooldown_upper_limit, cooldown_delay))

        self._cooldown_period_end_ts = current_time + cooldown_delta_seconds
        logging.debug(f"cooldown delta seconds: {cooldown_delta_seconds}")

    def blacklist(self):
        """Moves this host to a blacklist, and starts the cooldown period."""
        self._blacklisted = True
        now = time.time()
        if self._in_cooldown_period(now):
            return
        self._set_cooldown_period(now)
        self.set_event()

    def whitelist(self):
        """Ends the cooldown period and moves this host out of blacklist."""
        self._cooldown_period_end_ts = 0
        self._blacklisted = False

    def is_blacklisted(self):
        """Checks if the host is in the blacklist."""
        return self._blacklisted

    def is_resurrected(self):
        """Checks if host is in an expired cooldown period."""
        if self._cooldown_period_end_ts > 0:
            return not self._in_cooldown_period(time.time())
        return False



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

    def __str__(self):
        return f"slots: {self._host_slots} order: {self._host_assignment_order}"


class HostManager(object):
    def __init__(self, discovery, cooldown_range=None):
        self._current_hosts = DiscoveredHosts(host_slots={}, host_assignment_order=[])
        self._hosts_state = defaultdict(lambda: HostState(cooldown_range))
        self._discovery = discovery

    def update_available_hosts(self):
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
                elif self._hosts_state[h].is_resurrected():
                    res |= HostUpdateResult.added
            return res

        prev_host_slots = self._current_hosts.host_slots
        prev_host_assignment_order = self._current_hosts.host_assignment_order
        host_slots = self._discovery.find_available_hosts_and_slots()

        def whitelist_all_hosts():
            for host in host_slots.keys():
                if self._hosts_state[host].is_resurrected():
                    self._hosts_state[host].whitelist()

        def has_resurrected_hosts():
            resurrected_hosts = [host for host in host_slots.keys() if self._hosts_state[host].is_resurrected()]
            return len(resurrected_hosts) > 0

        if prev_host_slots != host_slots or has_resurrected_hosts():
            available_hosts = set([host for host in host_slots.keys() \
                if not (self._hosts_state[host].is_blacklisted() and not self._hosts_state[host].is_resurrected())])
            host_assignment_order = HostManager.order_available_hosts(available_hosts, prev_host_assignment_order)
            self._current_hosts = DiscoveredHosts(host_slots=host_slots,
                                                  host_assignment_order=host_assignment_order)
            host_update_state = check_update(self._current_hosts.host_slots, prev_host_slots)
            whitelist_all_hosts()
            return host_update_state
        else:
            return HostUpdateResult.no_update

    @property
    def current_hosts(self):
        return self._current_hosts.update(self._hosts_state)

    def blacklist(self, host):
        if not self._hosts_state[host].is_blacklisted():
            logging.info('blacklist failing host: {}'.format(host))
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
        result = self._execute_discovery_script()

        host_slots = {}
        lines = set(result.strip().split('\n'))
        for line in lines:
            host = line
            if ':' in line:
                host, slots = line.split(':')
                host_slots[host] = int(slots)
            # Make sure the host is not empty. The discovery script might
            # return empty string when all workers are not ready or available.
            elif host:
                host_slots[host] = self._default_slots
        return host_slots

    def _execute_discovery_script(self):
        stdout = io.StringIO()
        exit_code = safe_shell_exec.execute(
            self._discovery_script, stdout=stdout)
        if exit_code != 0:
            raise RuntimeError(
                'Failed to execute discovery script: {}. Exit code: {}' .format(
                    self._discovery_script, exit_code))
        return stdout.getvalue()


class FixedHosts(HostDiscovery):
    def __init__(self, host_slots):
        super(FixedHosts, self).__init__()
        self._host_slots = host_slots

    def find_available_hosts_and_slots(self):
        return self._host_slots

    def set(self, host_slots):
        self._host_slots = host_slots
