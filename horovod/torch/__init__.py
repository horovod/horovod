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

check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                __file__, 'mpi_lib', '_mpi_lib')

from horovod.torch import mpi_lib_impl
from horovod.torch import mpi_lib

import torch


# TODO (doc): add note in docs to use NCCL 2.1.15+ with PyTorch to avoid deadlock

class _DistributedOptimizer(torch.optim.Optimizer):
    # TODO (doc): make it clear that parameters are taken from optimizer,
    # TODO (doc): and named_parameters are only used for naming
    def __init__(self, params, named_parameters=None):
        super(self.__class__, self).__init__(params)

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        self._parameter_names = {v: k for k, v
                                 in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []

        if size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        def hook(*ignore):
            # TODO: better error messages
            assert p not in self._handles
            assert not p.grad.requires_grad
            name = self._parameter_names.get(p)
            handle = allreduce_async_(p.grad.data, average=True, name=name)
            self._handles[p] = handle
        return hook

    # TODO (doc): required for clip_grad_norm(all_params, 5), auto-detect if missing?
    def synchronize(self):
        for handle in self._handles.values():
            synchronize(handle)
        self._handles.clear()

    def step(self, closure=None):
        self.synchronize()
        return super(self.__class__, self).step(closure)


def DistributedOptimizer(optimizer, named_parameters=None):
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters)


def broadcast_global_variables(params, root_rank):
    if isinstance(params, dict):
        params = params.items()
    else:
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]

    # Run asynchronous broadcasts.
    handles = []
    for name, p in params:
        if isinstance(p, torch.autograd.Variable):
            p = p.data
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}


# Null pointer.
_NULL = mpi_lib._ffi.NULL


# TODO: split into multiple files
# TODO: check arguments, return good errors if they're not the right type
def _check_function(function_factory, tensor):
    function = function_factory(tensor)
    if not hasattr(mpi_lib, function):
        if tensor.is_cuda:
            cpu_function = function_factory(tensor.new().cpu())
            if hasattr(mpi_lib, cpu_function):
                raise ValueError('Horovod should be rebuilt with GPU support '
                                 'to enable support for CUDA tensors.')
        raise ValueError('Tensor type %s is not supported.' % tensor.type())
    return function


def _allreduce_function_factory(tensor):
    return 'horovod_torch_allreduce_async_' + tensor.type().replace('.', '_')


def _allreduce_async(tensor, output, average, name):
    function = _check_function(_allreduce_function_factory, tensor)
    handle = getattr(mpi_lib, function)(tensor, output, average,
                                        name.encode() if name is not None else _NULL)
    _handle_map[handle] = (tensor, output)
    return handle


def allreduce_async(tensor, average=True, name=None):
    # TODO check that tensor is contiguous
    assert tensor.is_contiguous()
    output = tensor.new(tensor.shape)
    return _allreduce_async(tensor, output, average, name)


def allreduce(tensor, average=True, name=None):
    handle = allreduce_async(tensor, average, name)
    return synchronize(handle)


def allreduce_async_(tensor, average=True, name=None):
    return _allreduce_async(tensor, tensor, average, name)


def allreduce_(tensor, average=True, name=None):
    handle = allreduce_async_(tensor, average, name)
    return synchronize(handle)


def _allgather_function_factory(tensor):
    return 'horovod_torch_allgather_async_' + tensor.type().replace('.', '_')


def _allgather_async(tensor, output, name):
    function = _check_function(_allgather_function_factory, tensor)
    handle = getattr(mpi_lib, function)(
        tensor, output, name.encode() if name is not None else _NULL)
    _handle_map[handle] = (tensor, output)
    return handle


def allgather_async(tensor, name=None):
    output = tensor.new()
    return _allgather_async(tensor, output, name)


def allgather(tensor, name=None):
    handle = allgather_async(tensor, name)
    return synchronize(handle)


def _broadcast_function_factory(tensor):
    return 'horovod_torch_broadcast_async_' + tensor.type().replace('.', '_')


def _broadcast_async(tensor, output, root_rank, name):
    function = _check_function(_broadcast_function_factory, tensor)
    handle = getattr(mpi_lib, function)(
        tensor, output, root_rank, name.encode() if name is not None else _NULL)
    _handle_map[handle] = (tensor, output)
    return handle


def broadcast_async(tensor, root_rank, name=None):
    output = tensor.new(tensor.shape)
    return _broadcast_async(tensor, output, root_rank, name)


def broadcast(tensor, root_rank, name=None):
    handle = broadcast_async(tensor, root_rank, name)
    return synchronize(handle)


def broadcast_async_(tensor, root_rank, name=None):
    return _broadcast_async(tensor, tensor, root_rank, name)


def broadcast_(tensor, root_rank, name=None):
    handle = broadcast_async_(tensor, root_rank, name)
    return synchronize(handle)


def poll(handle):
    return mpi_lib.horovod_torch_poll(handle)


def synchronize(handle):
    if handle not in _handle_map:
        return
    mpi_lib.horovod_torch_wait_and_clear(handle)
    _, output = _handle_map[handle]
    del _handle_map[handle]
    return output
