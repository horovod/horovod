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

import time


class Timeout(object):
    def __init__(self, timeout, message):
        self._timeout = timeout
        self._timeout_at = time.time() + timeout
        self._message = message

    def remaining(self):
        return max(0, self._timeout_at - time.time())

    def timed_out(self):
        return time.time() > self._timeout_at

    def check_time_out_for(self, activity):
        if self.timed_out():
            raise Exception(
                '{}{} Timeout after {} seconds.'.format(
                    self._message.format(activity=activity),
                    '.' if not self._message.rstrip().endswith('.') else '',
                    self._timeout
                )
            )
