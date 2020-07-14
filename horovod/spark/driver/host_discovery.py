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

from horovod.spark.driver.driver_service import SparkDriverService
from horovod.runner.elastic.discovery import HostDiscovery


class SparkDriverHostDiscovery(HostDiscovery):
    def __init__(self, driver):
        """
        :param driver: Spark driver service
        :type driver: SparkDriverService
        """
        super(SparkDriverHostDiscovery, self).__init__()
        self._driver = driver

    def find_available_hosts_and_slots(self):
        host_hash_indices = self._driver.task_host_hash_indices()
        slots = dict([(host, len(indices))
                      for host, indices in host_hash_indices.items()
                      if len(indices)])
        return slots
