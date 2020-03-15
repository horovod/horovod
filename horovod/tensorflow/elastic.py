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

from distutils.version import LooseVersion

import tensorflow as tf

from tensorflow.python.framework import ops

import horovod.tensorflow as _hvd

from horovod.common.elastic import run_fn, AbstractObjectState, State


_IS_TF2 = LooseVersion(tf.__version__) >= LooseVersion('2.0.0')


def run(func):
    return run_fn(func, _hvd)


def _broadcast_global(session):
    bcast_op = _hvd.broadcast_global_variables(0)
    session.run(bcast_op)


def _broadcast_model(model, optimizer, session):
    if _hvd._executing_eagerly():
        # TensorFlow 2.0 or TensorFlow eager
        _hvd.broadcast_variables(model.variables, root_rank=0)
        _hvd.broadcast_variables(optimizer.variables(), root_rank=0)
    else:
        _broadcast_global(session)


def _model_built(model):
    return model.built if hasattr(model, 'build') else True


class ObjectState(AbstractObjectState):
    def __init__(self, **kwargs):
        super(ObjectState, self).__init__(**kwargs)

    def sync(self):
        if self._saved_state:
            synced_state = _hvd.broadcast_object(self._saved_state, root_rank=0)
            if _hvd.rank() != 0:
                self._saved_state = synced_state
                self.restore()


class TensorFlowKerasState(ObjectState):
    def __init__(self, model, optimizer=None, **kwargs):
        self.model = model
        if not _model_built(model):
            raise ValueError('Model must be built first. Run `model.build(input_shape)`.')

        self.optimizer = optimizer or model.optimizer
        self._save_model()

        super(TensorFlowKerasState, self).__init__(**kwargs)

    def save(self):
        self._save_model()
        super(TensorFlowKerasState, self).save()

    def restore(self):
        self._load_model()
        super(TensorFlowKerasState, self).restore()

    def sync(self):
        _broadcast_model(self.model, self.optimizer, session=None)
        super(TensorFlowKerasState, self).sync()

    def _save_model(self):
        self._saved_model_state = self.model.get_weights()
        self._saved_optimizer_state = self.optimizer.get_weights()

    def _load_model(self):
        self._saved_model_state = self.model.get_weights()
        self._saved_optimizer_state = self.optimizer.get_weights()


class TensorFlowSessionState(ObjectState):
    def __init__(self, session=None, **kwargs):
        self.session = session or ops.get_default_session()
        self._save_model()

        super(TensorFlowSessionState, self).__init__(**kwargs)

    def save(self):
        self._save_model()
        super(TensorFlowSessionState, self).save()

    def restore(self):
        self._load_model()
        super(TensorFlowSessionState, self).restore()

    def sync(self):
        _broadcast_global(self.session)
        super(TensorFlowSessionState, self).sync()

    def _save_model(self):
        self._values = [var.eval(self.session) for var in tf.global_variables()]

    def _load_model(self):
        for var, value in zip(tf.global_variables(), self._values):
            var.load(value, self.session)


class TensorFlowState(State):
    def __init__(self, model=None, optimizer=None, session=None, **kwargs):
        self._state = TensorFlowKerasState(model, optimizer, **kwargs) if model is not None else \
            TensorFlowSessionState(session, **kwargs)
        super(TensorFlowState, self).__init__()

    def save(self):
        self._state.save()

    def restore(self):
        self._state.restore()

    def sync(self):
        self._state.sync()
