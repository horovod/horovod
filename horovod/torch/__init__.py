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

import collections

from horovod.common import init
from horovod.common import size
from horovod.common import local_size
from horovod.common import rank
from horovod.common import local_rank
from horovod.common import mpi_threads_supported
from horovod.common import check_extension

check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
                __file__, 'mpi_lib', '_mpi_lib')

from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import allgather, allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import poll, synchronize

import torch


class _DistributedOptimizer(torch.optim.Optimizer):
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
            assert p not in self._handles
            assert not p.grad.requires_grad
            name = self._parameter_names.get(p)
            handle = allreduce_async_(p.grad.data, average=True, name=name)
            self._handles[p] = handle
        return hook

    def synchronize(self):
        for handle in self._handles.values():
            synchronize(handle)
        self._handles.clear()

    def step(self, closure=None):
        self.synchronize()
        return super(self.__class__, self).step(closure)


def DistributedOptimizer(optimizer, named_parameters=None,
                         initialize_state=False):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.

    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.

    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```

    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
        initialize_state: Initialize the state variables for the purpose of receiving
                          their values from a broadcast sent by another rank.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    opt = cls(optimizer.param_groups, named_parameters)
    if initialize_state:
        for group in opt.param_groups:
            for p in group['params']:
                p.grad = torch.autograd.Variable(p.data.new(p.size()).zero_())
        opt.step()
    return opt


def broadcast_parameters(params, root_rank):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `model.state_dict()`,
    `model.named_parameters()`, or `model.parameters()`.

    Arguments:
        params: One of the following:
            - list of parameters to broadcast
            - dict of parameters to broadcast
            - torch.optim.Optimizer whose state will be broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    handles = []
    occurrences = collections.defaultdict(int)

    if isinstance(params, dict):
        params = sorted(params.items())
    elif isinstance(params, list):
        # support both named_parameters() and regular parameters()
        params = [p if isinstance(p, tuple) else (None, p) for p in params]
    elif isinstance(params, torch.optim.Optimizer):
        new_params = []
        for group in params.state_dict()['param_groups']:
            for pid in group['params']:
                param_state = params.state_dict()['state'][pid]
                for name, p in param_state.iteritems():
                    # Some parameter names may appear more than once, in which
                    # case we ensure they have a unique identifier defined by
                    # their order
                    occurrences[name] += 1
                    name = '%s.%d' % (str(name), occurrences[name])
                    new_params.append((name, p))
                    print(rank(), name, p.shape)
        params = new_params
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run asynchronous broadcasts.
    for name, p in params:
        if isinstance(p, torch.autograd.Variable):
            p = p.data
        handle = broadcast_async_(p, root_rank, name)
        handles.append(handle)

    # Wait for completion.
    for handle in handles:
        synchronize(handle)


def load_model(filepath, model, optimizer=None):
    """
    Loads a saved PyTorch model with a Horovod DistributedOptimizer.

    The DistributedOptimizer will wrap the underlying optimizer used to train
    the saved model, so that the optimizer state (params and weights) will
    be picked up for retraining.

    Arguments:
        filepath: The string path to the saved model.
        model: The model whose state will be loaded.
        optimizer: The (optional) underlying optimizer whose state will be
                   loaded and then wrapped by a DistributedOptimizer.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer = DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
