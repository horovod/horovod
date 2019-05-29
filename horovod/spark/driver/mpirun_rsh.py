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

import os
import sys

from horovod.spark.task import task_service
from horovod.spark.driver import driver_service
from horovod.run.common.util import codec, secret


def main(driver_addresses, settings, host_hash, command):
    if ':' in host_hash:
        raise Exception('Illegal host hash provided. Are you using Open MPI 4.0.0+?')

    key = codec.loads_base64(os.environ[secret.HOROVOD_SECRET_KEY])
    driver_client = driver_service.SparkDriverClient(driver_addresses, key,
                                                     verbose=settings.verbose)
    task_indices = driver_client.task_host_hash_indices(host_hash)
    # Since tasks with the same host hash have shared memory, we will run only
    # one ORTED process on the first task.
    first_task_index = task_indices[0]
    task_addresses = driver_client.all_task_addresses(first_task_index)
    task_client = task_service.SparkTaskClient(first_task_index, task_addresses,
                                               key, verbose=settings.verbose)
    task_client.run_command(command, os.environ)


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: %s <service addresses> <settings> <host hash> '
              '<command...>' % sys.argv[0])
        sys.exit(1)
    main(codec.loads_base64(sys.argv[1]), codec.loads_base64(sys.argv[2]),
         sys.argv[3], " ".join(sys.argv[4:]))
