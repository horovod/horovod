# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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


class Timeout(object):
    def __init__(self, timeout):
        self._timeout_at = time.time() + timeout

    def remaining(self):
        return max(0, self._timeout_at - time.time())

    def timed_out(self):
        return time.time() > self._timeout_at

    def check_time_out_for(self, activity):
        if self.timed_out():
            raise Exception('Timed out waiting for %s. Please check that you have enough resources '
                            'to run all Horovod processes. Each Horovod process runs in a Spark task. '
                            'You may need to increase the start_timeout parameter to a larger value '
                            'if your Spark resources are allocated on-demand.' % activity)
