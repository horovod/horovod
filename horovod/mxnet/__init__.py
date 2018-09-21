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

from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import allgather
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import poll, synchronize

import mxnet as mx

# This is where Horovod's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def update(self, index, weight, grad, state):
        allreduce_(grad, average=False, name=None)
        return self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        allreduce_(grad, average=False, name=None)
        return self._optimizer.update_multi_precision(index, weight, grad, state)

    def set_learning_rate(self, lr):
        return self._optimizer.set_learning_rate(lr)

    def set_lr_mult(args_lr_mult):
        return self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(args_wd_mult):
        return self._optimizer.set_wd_mult(args_wd_mult)

def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.get_params()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run broadcasts.
    print("Begin broadcast!")
    for name, p in params:
        broadcast_(p, root_rank, name)
