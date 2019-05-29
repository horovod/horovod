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

import sys

from horovod.run.common.util import codec, host_hash
from horovod.run.driver import driver_service
from horovod.run.task import task_service


def _task_fn(index, driver_addresses, settings):
    task = task_service.HorovodRunTaskService(index, settings.key)
    try:
        driver = driver_service.HorovodRunDriverClient(
            driver_addresses, settings.key, settings.verbose)
        driver.register_task(index,
                             task.addresses(),
                             host_hash.host_hash())
        task.wait_for_initial_registration(settings.timeout)
        # Tasks ping each other in a circular fashion to determine interfaces
        # reachable within the cluster.
        next_task_index = (index + 1) % settings.num_hosts
        next_task_addresses = driver.all_task_addresses(next_task_index)
        # We request interface matching to weed out all the NAT'ed interfaces.
        next_task = task_service.HorovodRunTaskClient(
            next_task_index,
            next_task_addresses,
            settings.key,
            settings.verbose,
            match_intf=True,
            retries=10)
        driver.register_task_to_task_addresses(next_task_index,
                                               next_task.addresses())
        # Notify the next task that the address checks are completed.
        next_task.task_to_task_address_check_completed()
        # Wait to get a notification from previous task that its address checks
        # are completed as well.
        task.wait_for_task_to_task_address_check_finish_signal(settings.timeout)

    finally:
        task.shutdown()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(
                'Usage: %s <index> <service addresses> <settings>' %
                sys.argv[0])
        sys.exit(1)

    index = codec.loads_base64(sys.argv[1])
    driver_addresses = codec.loads_base64(sys.argv[2])
    settings = codec.loads_base64(sys.argv[3])

    _task_fn(index, driver_addresses, settings)
