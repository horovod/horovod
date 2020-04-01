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

import sys
import time

from horovod.run.gloo_run import launch_gloo
from horovod.run.common.util import codec, secret
from horovod.spark.driver.rsh import rsh


def _exec_command_fn(driver_addresses, settings, env):
    def _exec_command(command, alloc_info, event):
        host = alloc_info.hostname
        local_rank = alloc_info.local_rank
        rsh(driver_addresses, settings, host, command, env, local_rank)
        # this indicate successful command execution, not the result of the executed command
        # the result of each task is collected through Spark at the end of horovod.spark.run.run()
        return 0, time.time()
    return _exec_command


def gloo_run(settings, nics, driver, env):
    """
    Run distributed gloo jobs.

    :param settings: Settings for running the distributed jobs.
                     Note: settings.num_proc and settings.hosts must not be None.
    :param nics: Interfaces to use by gloo.
    :param driver: The Spark driver service that tasks are connected to.
    :param env: Environment dictionary to use for running gloo jobs.
    """
    # Each thread will use SparkTaskClient to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session.
    iface = list(nics)[0]
    server_ip = driver.addresses()[iface][0][0]
    command = (sys.executable,
               '-m', 'horovod.spark.task.gloo_exec_fn',
               codec.dumps_base64(driver.addresses()),
               codec.dumps_base64(settings))

    # Pass secret key through the environment variables.
    env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(settings.key)

    exec_command = _exec_command_fn(driver.addresses(), settings, env)
    launch_gloo(command, exec_command, settings, nics, {}, server_ip)
