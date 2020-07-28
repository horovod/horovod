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

from horovod.runner.common.util import codec, secret
from horovod.spark.driver.rsh import rsh


if __name__ == '__main__':
    """
    Method run by MPI to connect to a host hash and execute the given command.

    The command is usually `orted` to setup the MPI cluster. That `orted` process
    is then used to spin-up the actual remote process, the Horovod user's Python method.
    The `orted` process will run on the lowest task index and all other tasks with the
    same host hash are expected to no-op (see `horovod.spark._task_fn`)
    and wait for the first task to terminate.

    :param driver_addresses: all IP addresses of the driver, base64 encoded
    :param settings: all settings, base64 encoded
    :param host_hash: the host hash to connect to
    :param command: the command and arguments to execute remotely
    """
    if len(sys.argv) < 5:
        print('Usage: %s <service addresses> <settings> <host hash> '
              '<command...>' % sys.argv[0])
        sys.exit(1)

    addresses = codec.loads_base64(sys.argv[1])
    key = codec.loads_base64(os.environ.get(secret.HOROVOD_SECRET_KEY))
    settings = codec.loads_base64(sys.argv[2])
    host_hash = sys.argv[3]
    command = " ".join(sys.argv[4:])
    env = {}  # orted does not need any env vars, the target training code gets env from mpirun

    # Since tasks with the same host hash have shared memory,
    # we will run only one orted process on the first task.
    rsh(addresses, key, host_hash, command, env, 0, settings.verbose)
