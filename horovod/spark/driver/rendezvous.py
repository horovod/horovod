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

from horovod.runner.http.http_server import RendezvousServer


class SparkRendezvousServer(RendezvousServer):
    def __init__(self, driver, verbose):
        super(SparkRendezvousServer, self).__init__(verbose)
        self._driver = driver

    def init(self, host_alloc_plan):
        super(SparkRendezvousServer, self).init(host_alloc_plan)

        # tell the SparkDriverService about the new host allocation plan
        # in the form of rank-to-index
        ranks_to_indices = {}
        host_indices = self._driver.task_host_hash_indices()
        for slot_info in host_alloc_plan:
            ranks_to_indices[slot_info.rank] = host_indices[slot_info.hostname][slot_info.local_rank]
        self._driver.set_ranks_to_indices(ranks_to_indices)

    def stop(self):
        self._driver.shutdown_tasks()
        super(SparkRendezvousServer, self).stop()
