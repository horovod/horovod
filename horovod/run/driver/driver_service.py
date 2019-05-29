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

from horovod.run.common.service import driver_service


class HorovodRunDriverService(driver_service.BasicDriverService):
    NAME = 'horovodrun driver service'

    def __init__(self, num_hosts, key):
        super(HorovodRunDriverService, self).__init__(num_hosts,
                                                      HorovodRunDriverService.NAME,
                                                      key)


class HorovodRunDriverClient(driver_service.BasicDriverClient):
    def __init__(self, driver_addresses, key, verbose, match_intf=False):
        super(HorovodRunDriverClient, self).__init__(
            HorovodRunDriverService.NAME,
            driver_addresses,
            key,
            verbose,
            match_intf=match_intf)
