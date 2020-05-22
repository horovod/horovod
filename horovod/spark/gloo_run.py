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


def _exec_command_fn(driver_addresses, key, settings, env):
    def _exec_command(command, alloc_info, event):
        host = alloc_info.hostname
        local_rank = alloc_info.local_rank
        verbose = settings.verbose
        result = rsh(driver_addresses, key, host, command, env, local_rank, verbose, False, event)
        return result, time.time()
    return _exec_command


def gloo_run(settings, nics, driver, env):
    """
    Run distributed gloo jobs.

    :param settings: Settings for running the distributed jobs.
                     Note: settings.num_proc and settings.hosts must not be None.
    :param nics: Interfaces to use by gloo.
    :param driver: The Spark driver service that tasks are connected to.
    :param env: Environment dictionary to use for running gloo jobs.  Can be None.
    """
    if env is None:
        env = {}

    if sys.version_info < (3, 0, 0):
        raise Exception('Horovod on Spark over Gloo only supported on Python3')

    # we don't want the key to be serialized along with settings from here on
    key = settings.key
    settings.key = None

    # Each thread will use SparkTaskClient to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session.
    iface = list(nics)[0]
    server_ip = driver.addresses()[iface][0][0]
    command = (sys.executable,
               '-m', 'horovod.spark.task.gloo_exec_fn',
               codec.dumps_base64(driver.addresses()),
               codec.dumps_base64(settings))

    exec_command = _exec_command_fn(driver.addresses(), key, settings, env)
    launch_gloo(command, exec_command, settings, nics, {}, server_ip)
