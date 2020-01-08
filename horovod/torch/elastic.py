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

from __future__ import absolute_import

import copy
import functools

from six.moves import queue

import horovod.torch

from horovod.common.exceptions import HorovodInternalError, WorkersAvailableException
from horovod.run.elastic.worker import WorkerNotificationManager
from horovod.torch.mpi_ops import init, rank, shutdown


notification_manager = WorkerNotificationManager()


class State(object):
    def __init__(self, saved_state):
        self._host_messages = queue.Queue()
        self._known_hosts = set()
        self._reset_callbacks = []
        self._saved_state = saved_state
        self.restore()

    def register_reset_callbacks(self, callbacks):
        self._reset_callbacks.extend(callbacks)

    def on_reset(self):
        for callback in self._reset_callbacks:
            callback()

    def on_hosts_added(self, hosts):
        for host in hosts:
            self._host_messages.put(host)

    def commit(self):
        self.save()
        self._update_known_hosts()

    def save(self):
        raise NotImplementedError()

    def restore(self):
        raise NotImplementedError()

    def sync(self):
        raise NotImplementedError()

    def _update_known_hosts(self):
        if not self._host_messages.empty():
            host = self._host_messages.get()
            if host not in self._known_hosts:
                self._known_hosts.add(host)
                raise WorkersAvailableException()


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
            synced_state = horovod.torch.broadcast_object(self._saved_state, root_rank=0)
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
        horovod.torch.broadcast_parameters(self.model.state_dict(), root_rank=0)
        horovod.torch.broadcast_optimizer_state(self.optimizer, root_rank=0)
        super(TorchState, self).sync()


def run(func):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        init()
        notification_manager.init()
        notification_manager.register_listener(state)

        try:
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
        finally:
            notification_manager.remove_listener(state)
    return wrapper


def _reset(state):
    shutdown()
    init()
    state.on_reset()
