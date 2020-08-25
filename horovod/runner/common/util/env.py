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

import re
import os

from horovod.runner.common.util import secret

LOG_LEVEL_STR = ['FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE']

# List of regular expressions to ignore environment variables by.
IGNORE_REGEXES = {'BASH_FUNC_.*', 'OLDPWD', secret.HOROVOD_SECRET_KEY}

KUBEFLOW_MPI_EXEC = '/etc/mpi/kubexec.sh'


def is_exportable(v):
    return not any(re.match(r, v) for r in IGNORE_REGEXES)


def get_env_rank_and_size():
    rank_env = ['HOROVOD_RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK']
    size_env = ['HOROVOD_SIZE', 'OMPI_COMM_WORLD_SIZE', 'PMI_SIZE']

    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)
        elif rank is not None or size is not None:
            raise RuntimeError(
                'Could not determine process rank and size: only one of {} and {} '
                'found in environment'.format(rank_var, size_var))

    # Default to rank zero and size one if there are no environment variables
    return 0, 1


def is_kubeflow_mpi():
    rsh_agent = os.environ.get('OMPI_MCA_plm_rsh_agent')
    return rsh_agent == KUBEFLOW_MPI_EXEC
