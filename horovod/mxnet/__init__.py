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

from horovod.common.util import check_extension

check_extension('horovod.mxnet', 'HOROVOD_WITH_MXNET',
                __file__, 'mpi_lib')

from horovod.mxnet.mpi_ops import allgather
from horovod.mxnet.mpi_ops import allreduce, allreduce_
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import size, local_size, rank, local_rank
from horovod.mxnet.mpi_ops import mpi_threads_supported

import mxnet as mx
import types
import warnings


# This is where Horovod's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        # Normalizing rescale_grad by Horovod size, which is equivalent to
        # performing average in allreduce, has better performance.
        self._optimizer.rescale_grad /= size()

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_allreduce(self, index, grad):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                allreduce_(grad[i], average=False,
                           name=str(index[i]), priority=-i)
        else:
            allreduce_(grad, average=False, name=str(index))

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


# DistributedTrainer, a subclass of MXNet gluon.Trainer.
# There are two differences between DistributedTrainer and Trainer:
# 1. DistributedTrainer calculates gradients using Horovod allreduce
#    API while Trainer does it using kvstore push/pull APIs;
# 2. DistributedTrainer performs allreduce(summation) and average
#    while Trainer only performs allreduce(summation).
class DistributedTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        super(DistributedTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        self._scale /= size()

    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=False,
                           name=str(i), priority=-i)


# Wrapper to inject Horovod broadcast after parameter initialization
def _append_broadcast_init(param, root_rank):
    init_impl = getattr(param, '_init_impl')
    def wrapped_init_impl(self, *args, **kwargs):
        init_impl(*args, **kwargs)
        broadcast_(self.data(), root_rank=root_rank)
        self.data().wait_to_read()
    return wrapped_init_impl


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
                # Inject wrapper method with post-initialization broadcast to
                # handle parameters with deferred initialization
                new_init = _append_broadcast_init(p, root_rank)
                p._init_impl = types.MethodType(new_init, p)
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run broadcasts.
    for i, tensor in enumerate(tensors):
        broadcast_(tensor, root_rank, str(i))

    # Make sure tensors pushed to MXNet engine get processed such that all
    # workers are synced before starting training.
    for tensor in tensors:
        tensor.wait_to_read()
