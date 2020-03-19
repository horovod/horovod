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

import unittest
import warnings

from horovod.run.common.util.hosts import HostInfo, parse_hosts, get_host_assignments


class RunCommonTests(unittest.TestCase):
    """
    Tests for horovod.run.common.
    """

    def __init__(self, *args, **kwargs):
        super(RunCommonTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_parse_hosts(self):
        """Test that hosts string can get parsed."""
        hosts = parse_hosts("host-1:1,host-2:2")
        self.assertEqual("host-1", hosts[0].hostname)
        self.assertEqual("host-2", hosts[1].hostname)

        self.assertEqual(1, hosts[0].slots)
        self.assertEqual(2, hosts[1].slots)

    def test_host_assignment_with_homogeneous_nodes(self):
        """Test that hosts equi-sized hosts are assigned in given order."""
        hosts = [HostInfo("host1", 2), HostInfo("host2", 2), HostInfo("host3", 2)]
        assignment = get_host_assignments(hosts, 2, 3)

        self.assertEqual("host1", assignment[0].hostname)
        self.assertEqual(0, assignment[0].rank)
        self.assertEqual(3, assignment[0].size)
        self.assertEqual(0, assignment[0].local_rank)
        self.assertEqual(2, assignment[0].local_size)
        self.assertEqual(0, assignment[0].cross_rank)
        self.assertEqual(2, assignment[0].cross_size)

        self.assertEqual("host1", assignment[1].hostname)
        self.assertEqual(1, assignment[1].rank)
        self.assertEqual(3, assignment[1].size)
        self.assertEqual(1, assignment[1].local_rank)
        self.assertEqual(2, assignment[1].local_size)
        self.assertEqual(0, assignment[1].cross_rank)
        self.assertEqual(1, assignment[1].cross_size)

        self.assertEqual("host2", assignment[2].hostname)
        self.assertEqual(2, assignment[2].rank)
        self.assertEqual(3, assignment[2].size)
        self.assertEqual(0, assignment[2].local_rank)
        self.assertEqual(1, assignment[2].local_size)
        self.assertEqual(1, assignment[2].cross_rank)
        self.assertEqual(2, assignment[2].cross_size)

    def test_host_assignment_with_heterogeneous_nodes(self):
        """Test that larger hosts are assigned first."""
        hosts = [HostInfo("host1", 1), HostInfo("host2", 3), HostInfo("host3", 2)]
        assignment = get_host_assignments(hosts, 2, 4)

        self.assertEqual("host2", assignment[0].hostname)
        self.assertEqual(0, assignment[0].rank)
        self.assertEqual(4, assignment[0].size)
        self.assertEqual(0, assignment[0].local_rank)
        self.assertEqual(3, assignment[0].local_size)
        self.assertEqual(0, assignment[0].cross_rank)
        self.assertEqual(2, assignment[0].cross_size)

        self.assertEqual("host2", assignment[1].hostname)
        self.assertEqual(1, assignment[1].rank)
        self.assertEqual(4, assignment[1].size)
        self.assertEqual(1, assignment[1].local_rank)
        self.assertEqual(3, assignment[1].local_size)
        self.assertEqual(0, assignment[1].cross_rank)
        self.assertEqual(1, assignment[1].cross_size)

        self.assertEqual("host2", assignment[2].hostname)
        self.assertEqual(2, assignment[2].rank)
        self.assertEqual(4, assignment[2].size)
        self.assertEqual(2, assignment[2].local_rank)
        self.assertEqual(3, assignment[2].local_size)
        self.assertEqual(0, assignment[2].cross_rank)
        self.assertEqual(1, assignment[2].cross_size)

        self.assertEqual("host3", assignment[3].hostname)
        self.assertEqual(3, assignment[3].rank)
        self.assertEqual(4, assignment[3].size)
        self.assertEqual(0, assignment[3].local_rank)
        self.assertEqual(1, assignment[3].local_size)
        self.assertEqual(1, assignment[3].cross_rank)
        self.assertEqual(2, assignment[3].cross_size)
