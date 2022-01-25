# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from horovod.runner.common.util.settings import BaseSettings


class ElasticSettings(BaseSettings):
    def __init__(self, discovery, min_np, max_np, elastic_timeout, reset_limit, cooldown_range=None, **kwargs):
        """
        :param discovery: object used to detect and manage available hosts
        :type discovery: horovod.runner.elastic.discovery.HostDiscovery
        :param min_np: minimum number of processes
        :type min_np: int
        :param max_np: maximum number of processes
        :type max_np: int
        :param elastic_timeout: timeout for elastic initialisation after re-scaling in seconds
        :type elastic_timeout: int
        :param reset_limit: maximum number of resets after which the job is terminated
        :type reset_limit: int
        :param cooldown_range: maximum number of resets after which the job is terminated
        :type cooldown_range: int
        """
        super(ElasticSettings, self).__init__(elastic=True, **kwargs)
        self.discovery = discovery
        self.min_np = min_np
        self.max_np = max_np
        self.elastic_timeout = elastic_timeout
        self.reset_limit = reset_limit
        self.cooldown_range=cooldown_range

    # we do not serialize the discovery instance
    # it is not needed on the worker and might not be serializable
    def __getstate__(self):
        result = self.__dict__.copy()
        result['discovery'] = None
        return result
