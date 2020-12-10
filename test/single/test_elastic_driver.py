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

import time
import unittest
import warnings

import mock
import pytest

from horovod.runner.util import network
from horovod.runner.elastic.discovery import FixedHosts, HostManager
from horovod.runner.elastic.driver import ElasticDriver
from horovod.runner.elastic.rendezvous import create_rendezvous_handler
from horovod.runner.elastic.worker import HostUpdateResult, WorkerNotificationManager
from horovod.runner.http.http_server import RendezvousServer


def wait_for_one(events):
    while True:
        for event in events:
            if event.is_set():
                return
        time.sleep(0.01)


def sequence(lst):
    for v in lst:
        yield v
    while True:
        yield lst[-1]


class ElasticDriverTests(unittest.TestCase):
    """
    Tests for async processing logic in horovod.elastic.
    """

    def __init__(self, *args, **kwargs):
        super(ElasticDriverTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_rank_and_size(self):
        """Tests two hosts, two slots each in standard happy path."""
        slots = {'host-1': 2, 'host-2': 2}
        discovery = FixedHosts(slots)

        driver = ElasticDriver(mock.Mock(), discovery, min_np=2, max_np=4)
        driver.wait_for_available_slots(min_np=2)

        rank_results = {}

        def exec_command(slot_info, events):
            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            updated_slot_info = driver.get_slot_info(slot_info.hostname, slot_info.local_rank)
            rank_results[slot_info.rank] = (slot_info, updated_slot_info)
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results().worker_results
        driver.stop()

        assert len(res) == 4
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        assert len(rank_results) == 4
        for rank, (slot_info, updated_slot_info) in rank_results.items():
            assert slot_info.to_response_string() == updated_slot_info.to_response_string(), rank

    def test_rank_and_size_with_host_failure(self):
        """Tests two hosts, two slots each with second host failing before rendezvous completes."""
        slots = {'host-1': 2, 'host-2': 2}
        discovery = FixedHosts(slots)

        driver = ElasticDriver(mock.Mock(), discovery, min_np=2, max_np=4)
        driver.wait_for_available_slots(min_np=2)

        rank_results = {}

        def exec_command(slot_info, events):
            if slot_info.hostname == 'host-2':
                return 1, time.time()

            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            updated_slot_info = driver.get_slot_info(slot_info.hostname, slot_info.local_rank)
            rank_results[slot_info.rank] = (slot_info, updated_slot_info)
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results().worker_results
        driver.stop()

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

    @mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_rank_and_size_with_host_added(self):
        """Tests training starts with one host two slots, then a second host is added."""
        slots = {'host-1': 2}
        discovery = FixedHosts(slots)

        def add_host():
            slots = {'host-1': 2, 'host-2': 2}
            discovery.set(slots)

        driver = ElasticDriver(mock.Mock(), discovery, min_np=2, max_np=4)
        driver.wait_for_available_slots(min_np=2)

        rank_results = {}

        def exec_command(slot_info, events):
            driver.record_ready(slot_info.hostname, slot_info.local_rank)

            if slot_info.hostname == 'host-1':
                if slot_info.rank == 0:
                    add_host()
                driver.wait_for_available_slots(4)
                driver.record_ready(slot_info.hostname, slot_info.local_rank)

            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            updated_slot_info = driver.get_slot_info(slot_info.hostname, slot_info.local_rank)
            rank_results[slot_info.rank] = (slot_info, updated_slot_info)
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results().worker_results
        driver.stop()

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

    @mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.runner.elastic.driver.ElasticDriver.get_coordinator_info')
    @mock.patch('horovod.runner.elastic.driver.ElasticDriver.get_worker_client')
    def test_wait_for_available_slots(self, mock_get_worker_client, mock_get_coordinator_info):
        """Tests that driver blocks until the min number of slots are available."""
        slots = [{'host-1': 4},
                 {'host-1': 4, 'host-2': 8},
                 {'host-1': 4, 'host-2': 8, 'host-3': 4}]
        mock_discovery = mock.Mock()
        mock_discovery.find_available_hosts_and_slots.side_effect = sequence(slots)

        driver = ElasticDriver(mock.Mock(), mock_discovery, min_np=8, max_np=20)
        driver.wait_for_available_slots(min_np=16)
        assert driver._host_manager.current_hosts.count_available_slots() >= 16
        driver.stop()

        # Notify coordinator 2 times, as the first time we are below min_np and the existing host assignments
        # are empty
        assert mock_get_worker_client.call_count == 2
        assert mock_get_coordinator_info.call_count == 2

    @mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_wait_for_min_hosts(self):
        """Tests that driver blocks until the min number of hosts and slots are available."""
        slots = [{'host-1': 4},
                 {'host-1': 4, 'host-2': 8},
                 {'host-1': 4, 'host-2': 8, 'host-3': 4}]
        mock_discovery = mock.Mock()
        mock_discovery.find_available_hosts_and_slots.side_effect = sequence(slots)

        driver = ElasticDriver(mock.Mock(), mock_discovery, min_np=2, max_np=12)
        driver.wait_for_available_slots(min_np=2, min_hosts=2)

        # Even though we only needed 2 slots, because we also needed 2 hosts, we will at least 12 slots total
        assert driver._host_manager.current_hosts.count_available_slots() >= 12
        driver.stop()

    def test_all_workers_fail(self):
        """Tests that training fails when all workers fail."""
        slots = {'host-1': 2, 'host-2': 2}
        discovery = FixedHosts(slots)

        driver = ElasticDriver(mock.Mock(), discovery, min_np=2, max_np=4)
        driver.wait_for_available_slots(min_np=2)

        def exec_command(slot_info, events):
            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            return 1, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results().worker_results
        driver.stop()

        assert len(res) == 4
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 1, name

    def test_shutdown_on_success(self):
        """Tests that shutdown event is triggered when one worker succeeds but the others are still working."""
        slots = {'host-1': 2, 'host-2': 2}
        discovery = FixedHosts(slots)

        driver = ElasticDriver(mock.Mock(), discovery, min_np=2, max_np=4)
        driver.wait_for_available_slots(min_np=2)

        def exec_command(slot_info, events):
            if slot_info.rank == 0:
                return 0, time.time()

            driver.record_ready(slot_info.hostname, slot_info.local_rank)
            wait_for_one(events)
            return 1, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results().worker_results
        driver.stop()

        assert len(res) == 4

        exit_code_sum = 0
        for name, (exit_code, timestamp) in res.items():
            exit_code_sum += exit_code
        assert exit_code_sum == 3

    def test_host_shutdown_on_worker_failure(self):
        """Tests two hosts, two slots each with one process on second host failing, causing host shutdown."""
        slots = {'host-1': 2, 'host-2': 2}
        discovery = FixedHosts(slots)

        driver = ElasticDriver(mock.Mock(), discovery, min_np=2, max_np=4)
        driver.wait_for_available_slots(min_np=2)

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
        res = driver.get_results().worker_results
        driver.stop()

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

    @mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_worker_notification_manager(self):
        """Tests that host add events are sent to the worker notification service and consumed."""
        slots = {'host-1': 2}
        discovery = FixedHosts(slots)

        rendezvous = RendezvousServer()
        driver = ElasticDriver(rendezvous, discovery, min_np=2, max_np=4)
        driver.wait_for_available_slots(min_np=2)
        handler = create_rendezvous_handler(driver)

        common_intfs = network.get_local_intfs()
        addr = network.get_driver_ip(common_intfs)
        port = rendezvous.start(handler)
        nic = list(common_intfs)[0]

        rank_results = {}

        class NotificationReceiver:
            def __init__(self):
                self.events = []

            def on_hosts_updated(self, timestamp, res):
                self.events.append((timestamp, res))

        def add_host():
            slots = {'host-1': 2, 'host-2': 2}
            discovery.set(slots)

        def remove_host():
            slots = {'host-2': 2}
            discovery.set(slots)

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
            driver.wait_for_available_slots(4)

            if slot_info.rank == 0:
                remove_host()

            # Busy wait for the number of available slots to decrease
            while driver._host_manager.current_hosts.count_available_slots() > 2:
                time.sleep(0.01)

            rank_results[slot_info.rank] = notification_receiver.events
            return 0, time.time()

        driver.start(np=2, create_worker_fn=exec_command)
        res = driver.get_results().worker_results
        driver.stop()

        assert len(res) == 2
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        assert len(rank_results) == 2
        for rank, events in rank_results.items():
            expected = 2 if rank == 0 else 0
            assert len(events) == expected, rank
            if rank == 0:
                # First update is an add
                assert events[0][1] == HostUpdateResult.added
                # Second update is a removal
                assert events[1][1] == HostUpdateResult.removed

        rendezvous.stop()

    @mock.patch('horovod.runner.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.runner.elastic.driver.ElasticDriver.host_assignments')
    @mock.patch('horovod.runner.elastic.driver.ElasticDriver.get_coordinator_info')
    @mock.patch('horovod.runner.elastic.driver.ElasticDriver.get_worker_client')
    def test_send_notifications_without_assignments(self, mock_get_worker_client, mock_get_coordinator_info,
                                                    mock_host_assignments):
        """Tests that notifications are still sent correctly even if host assignments cannot be generated."""
        slots = [{'host-1': 8, 'host-2': 4},
                 {'host-1': 8, 'host-2': 4},
                 {'host-2': 4},
                 {'host-2': 4},
                 {'host-2': 4, 'host-3': 12}]
        discovery = mock.Mock()
        discovery.find_available_hosts_and_slots.side_effect = sequence(slots)

        driver = ElasticDriver(mock.Mock(), discovery, min_np=8, max_np=12)
        driver.wait_for_available_slots(min_np=16)
        driver.stop()

        # On the second call, we should see the number of slots dip below the minimum, but we still want to ensure
        # we notify workers every time there is a change, so in total we should observe 3 calls.
        assert mock_get_worker_client.call_count == 3
        assert mock_get_coordinator_info.call_count == 3

    def test_order_available_hosts(self):
        """Tests the order is preserved for host assignment as available hosts are updated."""
        # This will be a set in practice, but use a list here to guarantee order.
        available_hosts = ['a', 'b', 'c']
        ordered_hosts = []
        ordered_hosts = HostManager.order_available_hosts(available_hosts, ordered_hosts)
        assert ordered_hosts == available_hosts

        # We remove a host, add a host, and chance the order, but relative order should be preserved
        available_hosts = ['d', 'c', 'b']
        ordered_hosts = HostManager.order_available_hosts(available_hosts, ordered_hosts)
        assert ordered_hosts == ['b', 'c', 'd']

    def test_update_available_hosts(self):
        """Tests that the current hosts object is immutable, while fetching the latest is correctly updated."""
        mock_discovery = mock.Mock()
        mock_discovery.find_available_hosts_and_slots.side_effect = [
            {'a': 2},
            {'a': 2, 'b': 2},
            {'b': 2},
            {'b': 1, 'c': 1},
            {'b': 1, 'c': 1}
        ]
        host_manager = HostManager(mock_discovery)

        # Should be empty initially
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == set()
        assert current_hosts.count_available_slots() == 0

        # From empty to {'a': 2}, it is an add update
        assert host_manager.update_available_hosts() == HostUpdateResult.added

        # First, check that nothing changed with our existing object, which is immutable
        assert current_hosts.available_hosts == set()
        assert current_hosts.count_available_slots() == 0

        # Now verify that the new object has the correct sets
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == {'a'}
        assert current_hosts.count_available_slots() == 2

        # Now check again
        # It is an increase update
        assert host_manager.update_available_hosts() == HostUpdateResult.added
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == {'a', 'b'}
        assert current_hosts.count_available_slots() == 4

        # And again
        # It is a removal update
        assert host_manager.update_available_hosts() == HostUpdateResult.removed
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == {'b'}
        assert current_hosts.count_available_slots() == 2

        # Try one more time
        # It is a mix update
        assert host_manager.update_available_hosts() == HostUpdateResult.mixed
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == {'b', 'c'}
        assert current_hosts.count_available_slots() == 2

        # Finally
        # No change
        assert host_manager.update_available_hosts() == HostUpdateResult.no_update
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == {'b', 'c'}
        assert current_hosts.count_available_slots() == 2

    def test_blacklist_host(self):
        """Tests the hosts are blacklisted, resulting in changes to the available hosts."""
        mock_discovery = mock.Mock()
        mock_discovery.find_available_hosts_and_slots.return_value = {'a': 2, 'b': 2}
        host_manager = HostManager(mock_discovery)

        host_manager.update_available_hosts()

        # Sanity check before we blacklist
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == {'a', 'b'}
        assert current_hosts.count_available_slots() == 4

        # Now blacklist, our existing object should not change (immutable)
        host_manager.blacklist('a')
        assert current_hosts.available_hosts == {'a', 'b'}
        assert current_hosts.count_available_slots() == 4

        # Check the new object, make sure we've blacklisted the host
        current_hosts = host_manager.current_hosts
        assert current_hosts.available_hosts == {'b'}
        assert current_hosts.count_available_slots() == 2

    def test_shutdown_on_initial_discovery_failure(self):
        """Tests that the driver will shutdown immediately if initial host discovery fails."""
        discovery = mock.Mock()
        discovery.find_available_hosts_and_slots.side_effect = RuntimeError()

        discover_hosts = ElasticDriver._discover_hosts

        def wrapped_discover_hosts(obj):
            try:
                discover_hosts(obj)
            except RuntimeError:
                # Suppress the error message from the background discovery thread to clean up unit tests
                pass

        try:
            ElasticDriver._discover_hosts = wrapped_discover_hosts
            driver = ElasticDriver(mock.Mock(), discovery, min_np=2, max_np=4)
            with pytest.raises(RuntimeError):
                driver.wait_for_available_slots(min_np=2)
            assert driver.finished()
        finally:
            ElasticDriver._discover_hosts = discover_hosts


if __name__ == "__main__":
    unittest.main()
