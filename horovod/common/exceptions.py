# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright Microsoft
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


class HorovodInternalError(RuntimeError):
    """Internal error raised when a Horovod collective operation (e.g., allreduce) fails.

    This is handled in elastic mode as a recoverable error, and will result in a reset event.
    """
    pass


class HostsUpdatedInterrupt(RuntimeError):
    """Internal interrupt event indicating that the set of hosts in the job has changed.

    In elastic mode, this will result in a reset event without a restore to committed state.
    """
    pass
