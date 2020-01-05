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

from horovod.run.elastic.driver import ElasticDriver


class ElasticTests(unittest.TestCase):
    """
    Tests for async processing logic in horovod.elastic.
    """

    def __init__(self, *args, **kwargs):
        super(ElasticTests, self).__init__(*args, **kwargs)
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
        print(res)
        assert len(res) == 2
        for name, (exit_code, timestamp) in res.items():
            assert exit_code == 0, name

        for rank, (slot_info, updated_slot_info) in rank_results.items():
            assert updated_slot_info.size == 2, rank
            assert updated_slot_info.rank == slot_info.rank % 2
            assert updated_slot_info.local_size == slot_info.local_size, rank
            assert updated_slot_info.local_rank == slot_info.local_rank, rank
            assert updated_slot_info.cross_size == 1, rank
            assert updated_slot_info.cross_rank == 0, rank

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_rank_and_size_with_worker_failure(self, mock_find_available_hosts_and_slots):
        """Tests two hosts, two slots each with one process on second host failing, causing host to fail."""
        pass

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_rank_and_size_with_host_added(self, mock_find_available_hosts_and_slots):
        """Tests training starts with one host two losts, then a second host is added."""
        pass

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_wait_for_available_hosts(self, mock_find_available_hosts_and_slots):
        """Tests that driver blocks until the min number of slots are available."""
        pass

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_all_workers_fail(self, mock_find_available_hosts_and_slots):
        """Tests that training fails when all workers fail."""
        pass

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_one_worker_success(self, mock_find_available_hosts_and_slots):
        """Tests that training fails when one worker succeeds but the others are still working."""
        pass

    @mock.patch('horovod.run.elastic.driver.ElasticDriver._find_available_hosts_and_slots')
    def test_all_workers_success(self, mock_find_available_hosts_and_slots):
        """Tests that training succeeds when all workers succeed."""
        pass


if __name__ == "__main__":
    unittest.main()
