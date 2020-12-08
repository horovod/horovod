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

from horovod.runner.common.util import env as env_util

SSH_COMMAND_PREFIX = 'ssh -o PasswordAuthentication=no -o StrictHostKeyChecking=no'


def get_ssh_command(local_command, host, port=None, identity_file=None, timeout_s=None):
    port_arg = f'-p {port}' if port is not None else ''
    identity_file_arg = f'-i {identity_file}' if identity_file is not None else ''
    timeout_arg = f'-o ConnectTimeout={timeout_s}' if timeout_s is not None else ''
    return f'{SSH_COMMAND_PREFIX} {host} {port_arg} {identity_file_arg} {timeout_arg} {local_command}'


def get_remote_command(local_command, host, port=None, identity_file=None, timeout_s=None):
    return f'{env_util.KUBEFLOW_MPI_EXEC} {host} {local_command}' if env_util.is_kubeflow_mpi() \
        else get_ssh_command(local_command, host, port, identity_file, timeout_s)
