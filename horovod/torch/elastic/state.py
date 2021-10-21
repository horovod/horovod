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

import copy

import torch

from horovod.common.elastic import ObjectState
from horovod.torch.elastic.sampler import ElasticSampler
from horovod.torch.functions import allgather_object, \
    broadcast_object, broadcast_optimizer_state, broadcast_parameters
from horovod.torch.mpi_ops import rank


class TorchState(ObjectState):
    """State representation of a PyTorch training process.

    Multiple models and optimizers are supported by providing them as
    kwargs. During initialization, `TorchState` will assign attributes
    for every keyword argument, and handle its state synchronization.

    Args:
        model: Optional PyTorch model.
        optimizer: Optional PyTorch optimizer.
        kwargs: Attributes sync, will be exposed as attributes of the object. If a handler exists
                for the attribute type, it will be used to sync the object, otherwise it will be
                handled an ordinary Python object.
    """
    def __init__(self, model=None, optimizer=None, **kwargs):
        kwargs.update(dict(model=model, optimizer=optimizer))
        self._handlers, kwargs = _get_handlers(kwargs)
        for name, handler in self._handlers.items():
            setattr(self, name, handler.value)
        super(TorchState, self).__init__(bcast_object=broadcast_object,
                                         get_rank=rank,
                                         **kwargs)

    def save(self):
        for handler in self._handlers.values():
            handler.save()
        super(TorchState, self).save()

    def restore(self):
        for handler in self._handlers.values():
            handler.restore()
        super(TorchState, self).restore()

    def sync(self):
        for handler in self._handlers.values():
            handler.sync()
        super(TorchState, self).sync()

    def __setattr__(self, name, value):
        if hasattr(self, name) and name in self._handlers:
            self._handlers[name].set_value(value)
        super().__setattr__(name, value)


class StateHandler(object):
    def __init__(self, value):
        self.value = value

    def save(self):
        raise NotImplementedError()

    def restore(self):
        raise NotImplementedError()

    def sync(self):
        raise NotImplementedError()

    def set_value(self, value):
        self.value = value
        self.save()


class ModelStateHandler(StateHandler):
    def __init__(self, model):
        super().__init__(model)
        self._saved_model_state = copy.deepcopy(self.value.state_dict())

    def save(self):
        self._saved_model_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value.load_state_dict(self._saved_model_state)

    def sync(self):
        broadcast_parameters(self.value.state_dict(), root_rank=0)


class OptimizerStateHandler(StateHandler):
    def __init__(self, optimizer):
        super().__init__(optimizer)
        self._saved_optimizer_state = copy.deepcopy(self.value.state_dict())

    def save(self):
        self._saved_optimizer_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value.load_state_dict(self._saved_optimizer_state)

    def sync(self):
        broadcast_optimizer_state(self.value, root_rank=0)


class SamplerStateHandler(StateHandler):
    def __init__(self, sampler):
        super().__init__(sampler)
        self._saved_sampler_state = copy.deepcopy(self.value.state_dict())

    def save(self):
        self._saved_sampler_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value.load_state_dict(self._saved_sampler_state)

    def sync(self):
        state_dict = self.value.state_dict()

        # Broadcast and load the state to make sure we're all in sync
        self.value.load_state_dict(broadcast_object(state_dict))


def _union(sets):
    # Union a list of sets into a single set
    return set().union(*sets)


_handler_registry = [
    (torch.nn.Module, ModelStateHandler),
    (torch.optim.Optimizer, OptimizerStateHandler),
    (ElasticSampler, SamplerStateHandler),
]


def get_handler_registry():
    return _handler_registry


def set_handler_registry(registry):
    global _handler_registry
    _handler_registry = registry


def _get_handler(v):
    for handler_type, handler_cls in _handler_registry:
        if isinstance(v, handler_type):
            return handler_cls(v)
    return None


def _get_handlers(kwargs):
    handlers = {}
    remainder = {}
    for k, v in kwargs.items():
        handler = _get_handler(v)
        if handler:
            handlers[k] = handler
        else:
            remainder[k] = v
    return handlers, remainder
