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

from horovod.runner.common.util import codec, secret
from horovod.runner.gloo_run import launch_gloo, launch_gloo_elastic
from horovod.spark.driver.rsh import rsh
from horovod.spark.driver.rendezvous import SparkRendezvousServer


def _exec_command_fn(driver, key, settings, env, stdout, stderr, prefix_output_with_timestamp):
    def _exec_command(command, slot_info, events):
        host = slot_info.hostname
        local_rank = slot_info.local_rank
        verbose = settings.verbose
        result = rsh(driver.addresses(), key, host, command, env, local_rank, verbose,
                     stdout, stderr, prefix_output_with_timestamp, False, events)
        return result, time.time()
    return _exec_command


def gloo_run(executable, settings, nics, driver, env, stdout=None, stderr=None):
    """
    Run distributed gloo jobs.

    :param executable: Executable to run when launching the workers.
    :param settings: Settings for running the distributed jobs.
                     Note: settings.num_proc and settings.hosts must not be None.
    :param nics: Interfaces to use by gloo.
    :param driver: The Spark driver service that tasks are connected to.
    :param env: Environment dictionary to use for running gloo jobs.  Can be None.
    :param stdout: Horovod stdout is redirected to this stream.
    :param stderr: Horovod stderr is redirected to this stream.
    """
    if env is None:
        env = {}

    # we don't want the key to be serialized along with settings from here on
    key = settings.key
    settings.key = None

    # Each thread will use SparkTaskClient to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session.
    iface = list(nics)[0]
    server_ip = driver.addresses()[iface][0][0]
    command = (executable,
               '-m', 'horovod.spark.task.gloo_exec_fn',
               codec.dumps_base64(driver.addresses()),
               codec.dumps_base64(settings))

    exec_command = _exec_command_fn(driver, key, settings, env,
                                    stdout, stderr, settings.prefix_output_with_timestamp)
    launch_gloo(command, exec_command, settings, nics, {}, server_ip)


def gloo_run_elastic(settings, driver, env, stdout=None, stderr=None):
    """
    Run distributed gloo jobs.

    :param settings: Settings for running the distributed jobs.
                     Note: settings.num_proc and settings.hosts must not be None.
    :param driver: The Spark driver service that tasks are connected to.
    :param env: Environment dictionary to use for running gloo jobs.  Can be None.
    :param stdout: Horovod stdout is redirected to this stream.
    :param stderr: Horovod stderr is redirected to this stream.
    """
    if env is None:
        env = {}

    # Each thread will use SparkTaskClient to launch the job on each remote host. If an
    # error occurs in one thread, entire process will be terminated. Otherwise,
    # threads will keep running and ssh session.
    command = (sys.executable,
               '-m', 'horovod.spark.task.gloo_exec_fn',
               codec.dumps_base64(driver.addresses()),
               codec.dumps_base64(settings))

    # Pass secret key through the environment variables.
    env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(settings.key)

    # get common interfaces from driver
    nics = driver.get_common_interfaces()

    exec_command = _exec_command_fn(driver, settings.key, settings, env,
                                    stdout, stderr, settings.prefix_output_with_timestamp)
    rendezvous = SparkRendezvousServer(driver, settings.verbose)
    launch_gloo_elastic(command, exec_command, settings, env, lambda _: nics, rendezvous, sys.executable)
