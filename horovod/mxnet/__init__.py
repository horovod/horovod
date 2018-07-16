# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common import init
from horovod.common import size
from horovod.common import local_size
from horovod.common import rank
from horovod.common import local_rank
from horovod.common import mpi_threads_supported
from horovod.common import check_extension

# TODO(ctcyang): do we need this here?
#check_extension('horovod.mxnet', 'HOROVOD_WITH_MXNET',
#                __file__, 'mpi_lib', '_mpi_lib')

from horovod.mxnet.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.mxnet.mpi_ops import allgather, allgather_async
from horovod.mxnet.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.mxnet.mpi_ops import poll, synchronize

import mxnet

# This is where Horovod's DistributedOptimizer wrapper for PyTorch goes
