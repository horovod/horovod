# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright Microsoft
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

import copy
import functools

from horovod.common.exceptions import HorovodInternalError, WorkersAvailableException
from horovod.torch import broadcast_object, broadcast_optimizer_state, broadcast_parameters
from horovod.torch.mpi_ops import init, rank, shutdown


class State(object):
    def __init__(self, saved_state):
        self._workers_available = []
        self._reset_callbacks = []
        self._saved_state = saved_state
        self.restore()

    def on_reset(self):
        for callback in self._reset_callbacks:
            callback()

    def register_reset_callbacks(self, callbacks):
        self._reset_callbacks.extend(callbacks)

    def commit(self):
        self.save()
        if self._workers_available:
            raise WorkersAvailableException()

    def save(self):
        raise NotImplementedError()

    def restore(self):
        raise NotImplementedError()

    def sync(self):
        raise NotImplementedError()


class ObjectState(State):
    def __init__(self, **kwargs):
        super(ObjectState, self).__init__(kwargs)

    def save(self):
        new_state = {}
        for attr in self._saved_state.keys():
            new_state[attr] = getattr(self, attr)
        self._saved_state = new_state

    def restore(self):
        for attr, value in self._saved_state.items():
            setattr(self, attr, value)

    def sync(self):
        if self._saved_state:
            synced_state = broadcast_object(self._saved_state, root_rank=0)
            if rank() != 0:
                self._saved_state = synced_state
                self.restore()


class TorchState(ObjectState):
    def __init__(self, model, optimizer, **kwargs):
        self.model = model
        self._saved_model_state = copy.deepcopy(model.state_dict())

        self.optimizer = optimizer
        self._saved_optimizer_state = copy.deepcopy(optimizer.state_dict())

        super(TorchState, self).__init__(**kwargs)

    def save(self):
        self._saved_model_state = copy.deepcopy(self.model.state_dict())
        self._saved_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        super(TorchState, self).save()

    def restore(self):
        self.model.load_state_dict(self._saved_model_state)
        self.optimizer.load_state_dict(self._saved_optimizer_state)
        super(TorchState, self).restore()

    def sync(self):
        broadcast_parameters(self.model.state_dict(), root_rank=0)
        broadcast_optimizer_state(self.optimizer, root_rank=0)
        super(TorchState, self).sync()


def run(func):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        reset_required = False
        while True:
            if reset_required:
                _reset(state)

            state.sync()
            try:
                return func(state, *args, **kwargs)
            except HorovodInternalError:
                state.restore()
            except WorkersAvailableException:
                pass
            reset_required = True
    return wrapper


def _reset(state):
    shutdown()
    init()
    state.on_reset()
