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

from __future__ import absolute_import

import logging
import threading

from collections import defaultdict

READY = 'READY'
SUCCESS = 'SUCCESS'
FAILURE = 'FAILURE'


class WorkerStateRegistry(object):
    def __init__(self, driver, discovered_hosts, verbose=False):
        self._driver = driver
        self._discovered_hosts = discovered_hosts
        self._lock = threading.Lock()
        self._states = {}
        self._workers = defaultdict(set)
        self._barrier = None
        self._rendezvous_id = 0
        self._verbose = verbose
        self._size = 0

    def get(self, state):
        return self._workers[state]

    def count(self, state):
        return len(self._workers[state])

    def reset(self, size):
        with self._lock:
            logging.info('reset workers: {}'.format(size))
            self._states.clear()
            self._workers.clear()
            self._barrier = threading.Barrier(parties=size, action=self._action)
            self._rendezvous_id += 1
            self._size = size

    def size(self):
        return self._size

    def last_rendezvous(self):
        return self._rendezvous_id

    def record_ready(self, host, slot):
        return self._record_state(host, slot, READY)

    def record_success(self, host, slot):
        return self._record_state(host, slot, SUCCESS)

    def record_failure(self, host, slot):
        return self._record_state(host, slot, FAILURE)

    def _record_state(self, host, slot, state):
        if self._driver.finished():
            logging.info('driver finished, ignoring registration: {}[{}] = {}'.format(host, slot, state))
            return self._rendezvous_id

        key = (host, slot)
        with self._lock:
            if key in self._states:
                logging.info('key exists, reset barrier: {}[{}] = {}'.format(host, slot, state))
                self._barrier.reset()
            logging.info('record state: {}[{}] = {}'.format(host, slot, state))
            self._states[key] = state
            self._workers[state].add(key)
            rendezvous_id = self._rendezvous_id

        rendezvous_id = self._wait(key, state, rendezvous_id)
        return rendezvous_id

    def _wait(self, key, state, rendezvous_id):
        while True:
            try:
                self._barrier.wait()
                return rendezvous_id
            except threading.BrokenBarrierError:
                if self._barrier.broken:
                    # Timeout or other non-recoverable error, so exit
                    raise

                with self._lock:
                    rendezvous_id = self._rendezvous_id
                    saved_state = self._states.get(key, state)
                    if saved_state != state:
                        raise RuntimeError('State {} overridden by {}'.format(state, saved_state))

    def _action(self):
        self._on_workers_recorded()

    def _on_workers_recorded(self):
        logging.info('all {} workers recorded'.format(self.size()))

        # Check for success state, if any process succeeded, shutdown all other processes
        if self.count(SUCCESS) > 0:
            logging.info('success count == {} -> stop running'.format(self.count(SUCCESS)))
            self._driver.stop()
            return

        # Check that all processes failed, indicating that processing should stop
        if self.count(FAILURE) == self._size:
            logging.error('failure count == {} -> stop running'.format(self._size))
            self._driver.stop()
            return

        # Check for failures, and add them to the blacklisted hosts list
        failures = self.get(FAILURE)
        for host, slot in failures:
            self._discovered_hosts.blacklist(host)

        # If there are no active hosts that aren't blacklisted, treat this as job failure
        blacklisted_slots = self._discovered_hosts.count_blacklisted_slots()
        if blacklisted_slots == self._size:
            logging.error('blacklisted slots count == {} -> stop running'.format(self._size))
            self._driver.stop()
            return

        try:
            self._driver.resume()
        except Exception:
            logging.exception('failed to activate new hosts -> stop running')
            self._driver.stop()
