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
import time

from horovod.runner.util.threads import in_thread
from horovod.spark.task import task_info, task_service
from horovod.spark.task.task_info import get_available_devices
from horovod.spark.driver import driver_service
from horovod.runner.common.util import codec, secret


def _parent_process_monitor(initial_ppid):
    try:
        while True:
            if initial_ppid != os.getppid():
                # Parent process died, terminate
                os._exit(1)
            time.sleep(1)
    except:
        # Avoids an error message during Python interpreter shutdown.
        pass


def task_exec(driver_addresses, settings, rank_env, local_rank_env):
    # Die if parent process terminates
    in_thread(target=_parent_process_monitor, args=(os.getppid(),))

    key = codec.loads_base64(os.environ[secret.HOROVOD_SECRET_KEY])
    rank = int(os.environ[rank_env])
    local_rank = int(os.environ[local_rank_env])
    driver_client = driver_service.SparkDriverClient(driver_addresses, key,
                                                     verbose=settings.verbose)

    # tell driver about local rank and rank
    # in elastic mode the driver already knows this mapping
    # for simplicity we keep code paths the same for elastic and static mode
    host_hash = os.environ['HOROVOD_HOSTNAME']
    task_index = driver_client.set_local_rank_to_rank(host_hash, local_rank, rank)

    # gather available resources from task service
    task_addresses = driver_client.all_task_addresses(task_index)
    task_client = task_service.SparkTaskClient(task_index, task_addresses, key,
                                               verbose=settings.verbose)
    task_info.set_resources(task_client.resources())

    fn, args, kwargs = driver_client.code()
    result = fn(*args, **kwargs)
    task_client.register_code_result(result)
