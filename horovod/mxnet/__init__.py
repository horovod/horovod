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

from horovod.common.util import check_extension, split_list

check_extension('horovod.mxnet', 'HOROVOD_WITH_MXNET',
                __file__, 'mpi_lib')

from horovod.mxnet.compression import Compression
from horovod.mxnet.functions import allgather_object, broadcast_object
from horovod.mxnet.mpi_ops import allgather
from horovod.mxnet.mpi_ops import allreduce, allreduce_, grouped_allreduce, grouped_allreduce_
from horovod.mxnet.mpi_ops import alltoall
from horovod.mxnet.mpi_ops import broadcast, broadcast_
from horovod.mxnet.mpi_ops import init, shutdown
from horovod.mxnet.mpi_ops import is_initialized, start_timeline, stop_timeline
from horovod.mxnet.mpi_ops import size, local_size, cross_size, rank, local_rank, cross_rank
from horovod.mxnet.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.mxnet.mpi_ops import gloo_enabled, gloo_built
from horovod.mxnet.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
from horovod.mxnet.mpi_ops import ProcessSet, global_process_set, add_process_set, remove_process_set

import mxnet as mx
from collections import OrderedDict, defaultdict
import types
import warnings


# This is where Horovod's DistributedOptimizer wrapper for MXNet goes
class DistributedOptimizer(mx.optimizer.Optimizer):
    def __init__(self, optimizer, gradient_predivide_factor=1.0, num_groups=0, process_set=global_process_set):
        if gradient_predivide_factor != 1.0 and rocm_built():
            raise ValueError('gradient_predivide_factor not supported yet with ROCm')

        self._optimizer = optimizer
        # Normalizing rescale_grad by Horovod size, which is equivalent to
        # performing average in allreduce, has better performance.
        self._optimizer.rescale_grad *= (gradient_predivide_factor / process_set.size())
        self._gradient_predivide_factor = gradient_predivide_factor
        self._num_groups = num_groups
        self._process_set = process_set

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state(self, index, weight):
        return self._optimizer.create_state(index, weight)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_allreduce(self, index, grad):
        if self._process_set.size() == 1: return

        if isinstance(index, (tuple, list)):
            if (self._num_groups > 0):
                grad_split = split_list(grad, self._num_groups)
                index_split = split_list(index, self._num_groups)

                for i, (grads, indices) in enumerate(zip(grad_split, index_split)):
                    grouped_allreduce_(tensors=grads, average=False, name="{}:{}".format(indices[0], indices[-1]), priority=-i,
                                       prescale_factor=1.0 / self._gradient_predivide_factor,
                                       process_set=self._process_set)
            else:
              for i in range(len(index)):
                  allreduce_(grad[i], average=False,
                             name=str(index[i]), priority=-i,
                             prescale_factor=1.0 / self._gradient_predivide_factor,
                             process_set=self._process_set)
        else:
            allreduce_(grad, average=False, name=str(index),
                       prescale_factor=1.0 / self._gradient_predivide_factor,
                       process_set=self._process_set)

    def update(self, index, weight, grad, state):
        if self._process_set.included():
            self._do_allreduce(index, grad)
        self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        if self._process_set.included():
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
    """The distributed trainer for data parallel training.

    Arguments:
        params: dict of parameters to train
        optimizer: mx.optim.Optimizer. the choice of optimizer
        optimizer_params: hyper-parameter of the chosen optimizer
        compression: Compression algorithm used during allreduce to reduce the amount
                     of data sent during the each parameter update step.  Defaults to
                     not using compression.
        gradient_predivide_factor: gradient_predivide_factor splits the averaging
              before and after the sum. Gradients are scaled by
              1.0 / gradient_predivide_factor before the sum and
              gradient_predivide_factor / size after the sum.
        prefix: the prefix of the parameters this trainer manages.
              If multiple trainers are used in the same program,
              they must be specified by different prefixes to avoid tensor name collision.
    """
    def __init__(self, params, optimizer, optimizer_params=None,
                 compression=Compression.none,
                 gradient_predivide_factor=1.0, prefix=None,
                 num_groups=0, process_set=global_process_set):
        self._compression = compression
        self._process_set = process_set

        if gradient_predivide_factor != 1.0 and rocm_built():
            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        # To ensure consistent parameter ordering across workers, sort params before
        # passing to base Trainer constructor. This logic is consistent with trainer.py
        # since v1.6 but we do it here for backwards compatability
        if isinstance(params, dict):
            params = OrderedDict(params)
        elif isinstance(params, (list, tuple)):
            params = sorted(params)

        super(DistributedTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by Horovod size, which is equivalent to performing
        # average in allreduce, has better performance. 
        self._scale *= (gradient_predivide_factor / process_set.size())
        self._gradient_predivide_factor = gradient_predivide_factor
        assert prefix is None or isinstance(prefix, str)
        self._prefix = prefix if prefix else ""
        self._num_groups = num_groups

    def _allreduce_grads(self):
        if self._process_set.size() == 1: return
        if not self._process_set.included(): return

        if (self._num_groups > 0):
            grads = []
            names = []
            tensors_compressed = []
            ctxs = []

            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    tensor_compressed, ctx = self._compression.compress(param.list_grad()[0])
                    grads.append(tensor_compressed)
                    tensors_compressed.append(tensor_compressed)
                    ctxs.append(ctx)
                    names.append(self._prefix + str(i))

            grads_split = split_list(grads, self._num_groups)
            names_split = split_list(names, self._num_groups)

            for i, (group_grads, group_names) in enumerate(zip(grads_split, names_split)):
                # For better performance, enqueue groups in separate grouped_allreduce calls by dtype.
                entries_by_dtype = defaultdict(list)
                for grad, name in zip(group_grads, group_names):
                    entries_by_dtype[grad.dtype].append((grad, name))

                for entries in entries_by_dtype.values():
                    grads, names = zip(*entries)
                    grouped_allreduce_(tensors=grads, average=False, name="{}:{}".format(names[0], names[-1]), priority=-i,
                                       prescale_factor=1.0 / self._gradient_predivide_factor,
                                       process_set=self._process_set)

            if self._compression != Compression.none:
                for i, param in enumerate(self._params):
                    if param.grad_req != 'null':
                        param.list_grad()[0][:] = self._compression.decompress(tensors_compressed.pop(0), ctxs.pop(0))
        else:
            # In MXNet 2.0, param.name is no longer unique.
            # Meanwhile, since horovod requires Python 3.6, there is no need to sort
            # self._params as enumerating a python dict is always deterministic.
            for i, param in enumerate(self._params):
                if param.grad_req != 'null':
                    tensor_compressed, ctx = self._compression.compress(param.list_grad()[0])
                    allreduce_(tensor_compressed, average=False,
                               name=self._prefix + str(i), priority=-i,
                               prescale_factor=1.0 / self._gradient_predivide_factor,
                               process_set=self._process_set)

                    if self._compression != Compression.none:
                        param.list_grad()[0][:] = self._compression.decompress(tensor_compressed, ctx)



# POS_Trainer is the stage one : partition optmizer status in DeepSpeed's work
# It can reduce memory consumption in distributed Trainer but slower
# since we can not solve overlapping problem when calling broadcast and optimize parameters.
# The usage of this trainer is totally same with original one
# I test some benchmark Here:
# For 4 V100 Gpu with 16GB memory, the maximum batch size for bert-large and bert-base:
# bert-large: Original: 16 Pos: 24
# bert-base: Original: 64 Pos: 80
# The ideal average saving memory for each GPU is: (N-1)/N * P * K
# where N is the GPU number, P is the parameter number and K is the memory
# multiplier of optimizer states(E.g. for Adam, K = 12)
#TODO add group_num
class POS_Trainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None,
                 gradient_predivide_factor=1.0, prefix=None):

        self._world_size = size()
        self._world_rank = rank()

        self._all_params = []
        self._all_param2idx = {}
        param_list = []
        if isinstance(params, (dict, OrderedDict)):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])
            params = param_list
        if not isinstance(params, (list, tuple)):
            raise ValueError(
                "First argument must be a list or dict of Parameters, " \
                "got %s." % (type(params)))
        for i, param in enumerate(params):
            if not isinstance(param, Parameter):
                raise ValueError(
                    "First argument must be a list or dict of Parameters, " \
                    "got list of %s." % (type(param)))
            if param._uuid in self._all_param2idx:
                # Shared parameters have same uuid; only need to store one of the shared versions
                continue
            self._all_param2idx[param._uuid] = i
            self._all_params.append(param)
        self._partition_params, self._param2rank = self._partition_parameters(self._all_params)
        self._own_part = self._partition_params[self._world_rank]
        super(POS_Trainer, self).__init__(
            self._own_part, optimizer, optimizer_params=optimizer_params, kvstore=None,
            num_groups=0, process_set=global_process_set)
        self._prefix = prefix if prefix else ""
        self._scale = gradient_predivide_factor / size()
        self._gradient_predivide_factor = gradient_predivide_factor


    def _partition_parameters(self, params):
        """
        partition all the parameters by their size and try to average them.
        """
        world_size = self._world_size
        ## list for rank each would be
        partition_params = [[] for _ in range(world_size)]
        param2rank = {}
        sizes = [0 for _ in range(world_size)]
        for param in params:
            if param.grad_req != 'null':
                current_rank = sizes.index(min(sizes))
                partition_params[current_rank].append(param)
                num = 1
                param2rank[param._uuid] = current_rank
                for p in param.shape:
                    num *= p
                sizes[current_rank] += num
        return partition_params, param2rank

    def _allreduce_grads(self):
        """
        rewrite allreduce here because we need to communicate using horovod.
        Actually we should use reduce here, but since it is not available yet,
        I use allreduce instead.
        """
        for i, param in enumerate(self._all_params):
            if param.grad_req != 'null':
                allreduce_(param.list_grad()[0], average=False,
                           name=self._prefix + str(i), priority=-i,
                           prescale_factor=1.0 / self._gradient_predivide_factor)





    def step(self, batch_size, ignore_stale_grad=False):
        """
        inherit from trainer, only call boardcast to make sure all parameter are consistent
        Makes one step of parameter update.
        Since each process main their own part, we need to brodcast after calculation
        """
        super(POS_Trainer, self).step(batch_size, ignore_stale_grad)
        self._broadcast_partition_params()

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

    def update(self, batch_size, ignore_stale_grad=False):
        '''
        assert not (self._kvstore and self._update_on_kvstore), \
            'update() when parameters are updated on kvstore ' \
            'is not supported. Try setting `update_on_kvstore` ' \
            'to False when creating trainer.'
        Since each process main their own part, we need to brodcast after calculation
        '''


        super(POS_Trainer, self).update(batch_size, ignore_stale_grad)
        self._broadcast_partition_params()

    def _broadcast_partition_params(self):
        """
        This function is to broadcast parameter since each process will maintain their own part
        """
        for param in self._all_params:
            broadcast_(param.data(), self._param2rank[param._uuid], name=str(self._all_param2idx[param._uuid]))

# Wrapper to inject Horovod broadcast after parameter initialization
def _append_broadcast_init(param, root_rank, name):
    init_impl = getattr(param, '_init_impl')
    def wrapped_init_impl(self, *args, **kwargs):
        init_impl(*args, **kwargs)
        broadcast_(self.data(), root_rank=root_rank, name=name)
    return wrapped_init_impl


def broadcast_parameters(params, root_rank=0, prefix=None):
    """Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()` or the
    `Block.collect_params()`.

    Arguments:
        params: One of the following:
            - dict of parameters to broadcast
            - ParameterDict to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
        prefix: The prefix of the parameters to broadcast.
              If multiple `broadcast_parameters` are called in the same program,
              they must be specified by different prefixes to avoid tensor name collision.
    """
    if size() == 1: return

    tensors = []
    names = []
    assert prefix is None or isinstance(prefix, str)
    prefix = prefix if prefix else ""
    try:
        from mxnet.gluon.parameter import ParameterDict
        valid_types = (dict, ParameterDict)
    except ImportError:
        valid_types = (dict,)
    if isinstance(params, valid_types):
        for name, p in sorted(params.items()):
            try:
                if isinstance(p, mx.gluon.parameter.Parameter):
                    tensors.append(p.data())
                else:
                    tensors.append(p)
                names.append(prefix + str(name))
            except mx.gluon.parameter.DeferredInitializationError:
                # Inject wrapper method with post-initialization broadcast to
                # handle parameters with deferred initialization
                # we use the key of params instead of param.name, since
                # param.name is no longer unique in MXNet 2.0
                new_init = _append_broadcast_init(p, root_rank, prefix + str(name))
                p._init_impl = types.MethodType(new_init, p)
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run broadcasts.
    for tensor, name in zip(tensors, names):
        broadcast_(tensor, root_rank, name=name)
