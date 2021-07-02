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

import copy
import sys

from horovod.runner.mpi_run import mpi_run as hr_mpi_run
from horovod.runner.common.util import codec, secret


def mpi_run(executable, settings, nics, driver, env, stdout=None, stderr=None):
    """
    Runs mpirun.

    :param executable: Executable to run when launching the workers.
    :param settings: Settings for running MPI.
                     Note: settings.num_proc and settings.hosts must not be None.
    :param nics: Interfaces to include by MPI.
    :param driver: The Spark driver service that tasks are connected to.
    :param env: Environment dictionary to use for running MPI.  Can be None.
    :param stdout: Stdout of the mpi process.
                   Only used when settings.run_func_mode is True.
    :param stderr: Stderr of the mpi process.
                   Only used when settings.run_func_mode is True.
    """
    env = {} if env is None else copy.copy(env)  # copy env so we do not leak env modifications

    # Pass secret key through the environment variables.
    env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(settings.key)
    # we don't want the key to be serialized along with settings from here on
    settings.key = None

    rsh_agent = (executable,
                 '-m', 'horovod.spark.driver.mpirun_rsh',
                 codec.dumps_base64(driver.addresses()),
                 codec.dumps_base64(settings))
    settings.extra_mpi_args = ('{extra_mpi_args} -x NCCL_DEBUG=INFO -mca plm_rsh_agent "{rsh_agent}"'
                               .format(extra_mpi_args=settings.extra_mpi_args if settings.extra_mpi_args else '',
                                       rsh_agent=' '.join(rsh_agent)))
    command = (executable,
               '-m', 'horovod.spark.task.mpirun_exec_fn',
               codec.dumps_base64(driver.addresses()),
               codec.dumps_base64(settings))
    hr_mpi_run(settings, nics, env, command, stdout=stdout, stderr=stderr)
