# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from horovod.common import check_extension

check_extension('horovod.mxnet', 'HOROVOD_WITH_MXNET',
                __file__, 'mpi_lib')

from horovod.mxnet.mpi_ops import allgather
from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import size, local_size, rank, local_rank
from horovod.mxnet.mpi_ops import mpi_threads_supported

import mxnet as mx


# This is where Horovod's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_allreduce(self, index, grad):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                allreduce_(grad[i], average=True, name=str(index[i]))
        else:
            allreduce_(grad, average=True, name=str(index))

    def update(self, index, weight, grad, state):
        self._do_allreduce(index, grad)
        self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        self._do_allreduce(index, grad)
        self._optimizer.update_multi_precision(index, weight, grad, state)

    def set_learning_rate(self, lr):
        self._optimizer.set_learning_rate(lr)

    def set_lr_mult(self, args_lr_mult):
        self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        self._optimizer.set_wd_mult(args_wd_mult)


def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()` or the
    `Block.collect_params()`.

    Arguments:
        params: One of the following:
            - dict of parameters to broadcast
            - ParameterDict to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    tensors = []
    if isinstance(params, dict):
        tensors = [p for _, p in sorted(params.items())]
    elif isinstance(params, mx.gluon.parameter.ParameterDict):
        for _, p in sorted(params.items()):
            try:
                tensors.append(p.data())
            except mx.gluon.parameter.DeferredInitializationError:
                # skip broadcasting deferred init param
                pass
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run broadcasts.
    for i, tensor in enumerate(tensors):
        broadcast_(tensor, root_rank, str(i))

    # Make sure tensors pushed to MXNet engine get processed such that all
    # workers are synced before starting training.
    for tensor in tensors:
        tensor.wait_to_read()
