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

from horovod.run.common.service import task_service


class SparkTaskService(task_service.BasicTaskService):
    NAME_FORMAT = 'task service #%d'

    def __init__(self, index, key):
        super(SparkTaskService, self).__init__(SparkTaskService.NAME_FORMAT % index, key)


class SparkTaskClient(task_service.BasicTaskClient):

    def __init__(self, index, task_addresses, key, verbose, match_intf=False):
        super(SparkTaskClient, self).__init__(SparkTaskService.NAME_FORMAT % index,
                                              task_addresses, key, verbose,
                                              match_intf=match_intf)
