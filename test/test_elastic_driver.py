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
from __future__ import division
from __future__ import print_function

import time
import unittest
import warnings

import mock

from horovod.run.util import network
from horovod.run.elastic.driver import ElasticDriver
from horovod.run.elastic.rendezvous import create_rendezvous_handler
from horovod.run.elastic.worker import WorkerNotificationManager
from horovod.run.http.http_server import RendezvousServer


def wait_for_one(events):
    while True:
        for event in events:
            if event.is_set():
                return
        time.sleep(0.01)


class ElasticDriverTests(unittest.TestCase):
    """
    Tests for async processing logic in horovod.elastic.
    """

    def __init__(self, *args, **kwargs):
        super(ElasticDriverTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_rank_and_size(self, mock_find_available_hosts_and_slots):
        """Tests two hosts, two slots each in standard happy path."""
        hosts = {'host-1', 'host-2'}
        slots = {'host-1': 2, 'host-2': 2}
        mock_find_available_hosts_and_slots.return_value = hosts, slots

        driver = ElasticDriver(None, min_np=2, max_np=4, slots=2)
        driver.wait_for_available_hosts(min_np=2)

        rank_results = {}

        def exec_command(slot_info, events):
            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            updated_slot_info = driver.get_slot_info(slot_info.hostname, slot_info.local_rank)
            rank_results[slot_info.rank] = (slot_info, updated_slot_info)
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results()
        assert len(res) == 4
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        assert len(rank_results) == 4
        for rank, (slot_info, updated_slot_info) in rank_results.items():
            assert slot_info.to_response_string() == updated_slot_info.to_response_string(), rank

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_rank_and_size_with_host_failure(self, mock_find_available_hosts_and_slots):
        """Tests two hosts, two slots each with second host failing before rendezvous completes."""
        hosts = {'host-1', 'host-2'}
        slots = {'host-1': 2, 'host-2': 2}
        mock_find_available_hosts_and_slots.return_value = hosts, slots

        driver = ElasticDriver(None, min_np=2, max_np=4, slots=2)
        driver.wait_for_available_hosts(min_np=2)

        rank_results = {}

        def exec_command(slot_info, events):
            if slot_info.hostname == 'host-2':
                return 1, time.time()

            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            updated_slot_info = driver.get_slot_info(slot_info.hostname, slot_info.local_rank)
            rank_results[slot_info.rank] = (slot_info, updated_slot_info)
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results()
        assert len(res) == 2
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        assert len(rank_results) == 2
        for rank, (slot_info, updated_slot_info) in rank_results.items():
            assert updated_slot_info.size == 2, rank
            assert updated_slot_info.rank == slot_info.rank % 2, rank
            assert updated_slot_info.local_size == slot_info.local_size, rank
            assert updated_slot_info.local_rank == slot_info.local_rank, rank
            assert updated_slot_info.cross_size == 1, rank
            assert updated_slot_info.cross_rank == 0, rank

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_rank_and_size_with_host_added(self, mock_find_available_hosts_and_slots):
        """Tests training starts with one host two slots, then a second host is added."""
        hosts = {'host-1'}
        slots = {'host-1': 2}
        mock_find_available_hosts_and_slots.return_value = hosts, slots

        def add_host():
            hosts = {'host-1', 'host-2'}
            slots = {'host-1': 2, 'host-2': 2}
            mock_find_available_hosts_and_slots.return_value = hosts, slots

        driver = ElasticDriver(None, min_np=2, max_np=4, slots=2)
        driver.wait_for_available_hosts(min_np=2)

        rank_results = {}

        def exec_command(slot_info, events):
            driver.record_ready(slot_info.hostname, slot_info.local_rank)

            if slot_info.hostname == 'host-1':
                if slot_info.rank == 0:
                    add_host()
                driver.wait_for_available_hosts(4)
                driver.record_ready(slot_info.hostname, slot_info.local_rank)

            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            updated_slot_info = driver.get_slot_info(slot_info.hostname, slot_info.local_rank)
            rank_results[slot_info.rank] = (slot_info, updated_slot_info)
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results()
        assert len(res) == 4
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        assert len(rank_results) == 4
        for rank, (slot_info, updated_slot_info) in rank_results.items():
            assert updated_slot_info.size == 4, rank
            assert updated_slot_info.rank == slot_info.rank, rank
            assert updated_slot_info.local_size == slot_info.local_size, rank
            assert updated_slot_info.local_rank == slot_info.local_rank, rank
            assert updated_slot_info.cross_size == 2, rank
            assert updated_slot_info.cross_rank == slot_info.cross_rank, rank

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_wait_for_available_hosts(self, mock_find_available_hosts_and_slots):
        """Tests that driver blocks until the min number of slots are available."""
        hosts = [{'host-1'},
                 {'host-1', 'host-2'},
                 {'host-1', 'host-2', 'host-3'}]
        slots = [{'host-1': 4},
                 {'host-1': 4, 'host-2': 8},
                 {'host-1': 4, 'host-2': 8, 'host-3': 4}]
        mock_find_available_hosts_and_slots.side_effect = zip(hosts, slots)

        driver = ElasticDriver(None, min_np=2, max_np=12, slots=0)
        driver.wait_for_available_hosts(min_np=10)
        assert driver._count_available_slots() >= 10

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_all_workers_fail(self, mock_find_available_hosts_and_slots):
        """Tests that training fails when all workers fail."""
        hosts = {'host-1', 'host-2'}
        slots = {'host-1': 2, 'host-2': 2}
        mock_find_available_hosts_and_slots.return_value = hosts, slots

        driver = ElasticDriver(None, min_np=2, max_np=4, slots=2)
        driver.wait_for_available_hosts(min_np=2)

        def exec_command(slot_info, events):
            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            return 1, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results()
        assert len(res) == 4
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 1, name

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_shutdown_on_success(self, mock_find_available_hosts_and_slots):
        """Tests that shutdown event is triggered when one worker succeeds but the others are still working."""
        hosts = {'host-1', 'host-2'}
        slots = {'host-1': 2, 'host-2': 2}
        mock_find_available_hosts_and_slots.return_value = hosts, slots

        driver = ElasticDriver(None, min_np=2, max_np=4, slots=2)
        driver.wait_for_available_hosts(min_np=2)

        def exec_command(slot_info, events):
            if slot_info.rank == 0:
                return 0, time.time()

            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            wait_for_one(events)
            return 1, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results()
        assert len(res) == 4

        exit_code_sum = 0
        for name, (exit_code, timestamp) in res.items():
            exit_code_sum += exit_code
        assert exit_code_sum == 3

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_host_shutdown_on_worker_failure(self, mock_find_available_hosts_and_slots):
        """Tests two hosts, two slots each with one process on second host failing, causing host shutdown."""
        hosts = {'host-1', 'host-2'}
        slots = {'host-1': 2, 'host-2': 2}
        mock_find_available_hosts_and_slots.return_value = hosts, slots

        driver = ElasticDriver(None, min_np=2, max_np=4, slots=2)
        driver.wait_for_available_hosts(min_np=2)

        rank_results = {}

        def exec_command(slot_info, events):
            if slot_info.hostname == 'host-1':
                if slot_info.local_rank == 0:
                    return 1, time.time()

                driver.record_ready(slot_info.hostname, slot_info.local_rank)
                wait_for_one(events)
                return 1, time.time()

            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            updated_slot_info = driver.get_slot_info(slot_info.hostname, slot_info.local_rank)
            rank_results[slot_info.rank] = (slot_info, updated_slot_info)
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results()
        assert len(res) == 2
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        assert len(rank_results) == 2
        for rank, (slot_info, updated_slot_info) in rank_results.items():
            assert updated_slot_info.size == 2, rank
            assert updated_slot_info.rank == slot_info.rank % 2, rank
            assert updated_slot_info.local_size == slot_info.local_size, rank
            assert updated_slot_info.local_rank == slot_info.local_rank, rank
            assert updated_slot_info.cross_size == 1, rank
            assert updated_slot_info.cross_rank == 0, rank

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_host_failure_in_rendezvous(self, mock_find_available_hosts_and_slots):
        """Tests that rendezvous will continue successfully if a host fails after it records ready."""
        return

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_worker_notification_manager(self, mock_find_available_hosts_and_slots):
        """Tests that host add events are sent to the worker notification service and consumed."""
        hosts = {'host-1'}
        slots = {'host-1': 2}
        mock_find_available_hosts_and_slots.return_value = hosts, slots

        rendezvous = RendezvousServer()
        driver = ElasticDriver(rendezvous, None, min_np=2, max_np=4, slots=2)
        driver.wait_for_available_hosts(min_np=2)
        handler = create_rendezvous_handler(driver)

        common_intfs = network.get_local_intfs()
        addr = network.get_driver_ip(common_intfs)
        port = rendezvous.start_server(handler)
        nic = list(common_intfs)[0]

        rank_results = {}

        class NotificationReceiver:
            def __init__(self):
                self.events = []

            def on_hosts_updated(self, timestamp):
                self.events.append(timestamp)

        def add_host():
            hosts = {'host-1', 'host-2'}
            slots = {'host-1': 2, 'host-2': 2}
            mock_find_available_hosts_and_slots.return_value = hosts, slots

        def remove_host():
            hosts = {'host-2'}
            slots = {'host-2': 2}
            mock_find_available_hosts_and_slots.return_value = hosts, slots

        def exec_command(slot_info, events):
            manager = WorkerNotificationManager()
            manager.init(rendezvous_addr=addr,
                         rendezvous_port=port,
                         nic=nic,
                         hostname=slot_info.hostname,
                         local_rank=slot_info.local_rank)

            notification_receiver = NotificationReceiver()
            manager.register_listener(notification_receiver)

            driver.record_ready(slot_info.hostname, slot_info.local_rank)

            if slot_info.rank == 0:
                add_host()
            driver.wait_for_available_hosts(4)

            if slot_info.rank == 0:
                remove_host()
            driver.wait_for_available_hosts(2, max_np=2)

            rank_results[slot_info.rank] = notification_receiver.events
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results()
        assert len(res) == 2
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        assert len(rank_results) == 2
        for rank, timestamps in rank_results.items():
            assert len(timestamps) == 2

        rendezvous.stop_server()
        driver.stop()


if __name__ == "__main__":
    unittest.main()
