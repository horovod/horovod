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

import horovod.torch as _hvd

from horovod.common.elastic import run_fn, AbstractObjectState, State


def run(func):
    return run_fn(func, _hvd)


class ObjectState(AbstractObjectState):
    def __init__(self, **kwargs):
        super(ObjectState, self).__init__(**kwargs)

    def sync(self):
        if self._saved_state:
            synced_state = _hvd.broadcast_object(self._saved_state, root_rank=0)
            if _hvd.rank() != 0:
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
        _hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        _hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        super(TorchState, self).sync()
