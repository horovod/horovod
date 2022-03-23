# Copyright 2022 Uber Technologies, Inc. All Rights Reserved.
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

import unittest

import mock
from parameterized import parameterized

from horovod.runner.elastic.discovery import HostDiscoveryScript

# Mocked default slots for testing.
DEFAULT_SLOTS = mock.Mock()


class HostDiscoveryScriptTests(unittest.TestCase):
    """
    Tests for host discovery script in horovod.elastic.
    """

    def __init__(self, *args, **kwargs):
        super(HostDiscoveryScriptTests, self).__init__(*args, **kwargs)

    @parameterized.expand([
        ["", {}],  # No host found.
        ["host-1", {"host-1": DEFAULT_SLOTS}],
        ["host-1:2", {"host-1": 2}],
        ["host-1\nhost-5", {"host-1": DEFAULT_SLOTS, "host-5": DEFAULT_SLOTS}],
        ["host-1:2\nhost-4:3", {"host-1": 2, "host-4": 3}],
        ["\nhost-1:2\nhost-4:3", {"host-1": 2, "host-4": 3}],
        ["host-1:2\n\nhost-4:3", {"host-1": 2, "host-4": 3}],
        ["host-1:2\nhost-4:3\n", {"host-1": 2, "host-4": 3}],
    ])
    def test_find_available_hosts_and_slots(
            self, script_result, expected_hosts_and_slots):
        """Tests that HostDiscoveryScript finds available hosts and slots."""
        mock_result = mock.Mock(return_value=script_result)
        host_discovery_script = HostDiscoveryScript(
            discovery_script=mock.Mock(),
            slots=DEFAULT_SLOTS
        )
        host_discovery_script._execute_discovery_script = mock_result

        hosts_and_slots = host_discovery_script.find_available_hosts_and_slots()
        assert hosts_and_slots == expected_hosts_and_slots
