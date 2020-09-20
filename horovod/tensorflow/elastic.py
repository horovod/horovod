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

from distutils.version import LooseVersion

import tensorflow as tf

from tensorflow.python.framework import ops

from horovod.common.elastic import run_fn, ObjectState
from horovod.common.exceptions import HorovodInternalError
from horovod.tensorflow.functions import broadcast_object, broadcast_object_fn, broadcast_variables
from horovod.tensorflow.mpi_ops import _executing_eagerly, init, rank, shutdown


_IS_TF2 = LooseVersion(tf.__version__) >= LooseVersion('2.0.0')


def run(func):
    """Decorator used to run the elastic training process.

    The purpose of this decorator is to allow for uninterrupted execution of the wrapped function
    across multiple workers in parallel, as workers come and go from the system. When a new worker is added,
    its state needs to be brought to the same point as the other workers, which is done by synchronizing
    the state object before executing `func`.

    When a worker is added or removed, other workers will raise an exception to bring them back to such a sync
    point before executing `func` again. This ensures that workers do not diverge when such reset events occur.

    It's important to note that collective operations (e.g., broadcast, allreduce) cannot be the call to
    the wrapped function. Otherwise, new workers could execute these operations during their initialization
    while other workers are attempting to sync state, resulting in deadlock.

    Args:
        func: a wrapped function taking any number of args or kwargs. The first argument
              must be a `horovod.common.elastic.State` object used to synchronize state across
              workers.
    """
    from tensorflow.python.framework.errors_impl import UnknownError

    def wrapper(state, *args, **kwargs):
        try:
            return func(state, *args, **kwargs)
        except UnknownError as e:
            if 'HorovodAllreduce' in e.message or \
                    'HorovodAllgather' in e.message or \
                    'HorovodBroadcast' in e.message:
                raise HorovodInternalError(e)
    return run_fn(wrapper, _reset)


def _reset():
    shutdown()
    init()


def _broadcast_model(model, optimizer, backend):
    if _executing_eagerly():
        # TensorFlow 2.0 or TensorFlow eager
        broadcast_variables(model.variables, root_rank=0)
        broadcast_variables(optimizer.variables(), root_rank=0)
    else:
        bcast_op = broadcast_variables(_global_variables(), root_rank=0)
        backend.get_session().run(bcast_op)


def _model_built(model):
    return model.built if hasattr(model, 'build') else True


def _global_variables():
    return tf.global_variables() if not _IS_TF2 else tf.compat.v1.global_variables()


def _default_session():
    return ops.get_default_session() if not _IS_TF2 else None


class TensorFlowKerasState(ObjectState):
    """State representation of a TensorFlow Keras model and optimizer.

    Supports TensorFlow 2 models and optimizers, as well as `keras` and `tf.keras`.

    Args:
        model: TensorFlow Keras model.
        optimizer: Optional optimizer, can be compiled into model instead.
        backend: For TensorFlow v1, backend used by Keras for obtaining the session.
        kwargs: Additional properties to sync, will be exposed as attributes of the object.
    """
    def __init__(self, model, optimizer=None, backend=None, **kwargs):
        self.model = model
        if not _model_built(model):
            raise ValueError('Model must be built first. Run `model.build(input_shape)`.')

        self.optimizer = optimizer or model.optimizer
        self.backend = backend
        self._save_model()

        if not backend or _executing_eagerly():
            self._bcast_model = lambda: _broadcast_model(self.model, self.optimizer, backend=self.backend)
            bcast_object = broadcast_object
        else:
            # For TensorFlow v1, we need to reuse the broadcast op to prevent incrementing the uids
            bcast_op = broadcast_variables(_global_variables(), root_rank=0)
            self._bcast_model = lambda: self.backend.get_session().run(bcast_op)
            bcast_object = broadcast_object_fn(session=self.backend.get_session())

        super(TensorFlowKerasState, self).__init__(bcast_object=bcast_object,
                                                   get_rank=rank,
                                                   **kwargs)

    def save(self):
        self._save_model()
        super(TensorFlowKerasState, self).save()

    def restore(self):
        self._load_model()
        super(TensorFlowKerasState, self).restore()

    def sync(self):
        self._bcast_model()
        self._save_model()
        super(TensorFlowKerasState, self).sync()

    def _save_model(self):
        if _executing_eagerly():
            self._saved_model_state = [tf.identity(var) for var in self.model.variables]
            self._saved_optimizer_state = [tf.identity(var) for var in self.optimizer.variables()]
        else:
            self._saved_model_state = self.model.get_weights()
            self._saved_optimizer_state = self.optimizer.get_weights()

    def _load_model(self):
        if _executing_eagerly():
            for var, saved_var in zip(self.model.variables, self._saved_model_state):
                var.assign(saved_var)
            for var, saved_var in zip(self.optimizer.variables(), self._saved_optimizer_state):
                var.assign(saved_var)
        else:
            self.model.set_weights(self._saved_model_state)
            self.optimizer.set_weights(self._saved_optimizer_state)


class TensorFlowState(ObjectState):
    """State representation of a list of TensorFlow variables.

    Supports both TensorFlow v1 and v2. For TensorFlow v2, can only be used when eager execution is enabled.

    Args:
        variables: List of `tf.Variable` objects to track (default: `tf.global_variables()`).
        session: For TensorFlow v1, session used to materialize variables (default: `ops.get_default_session()`).
        kwargs: Additional properties to sync, will be exposed as attributes of the object.
    """
    def __init__(self, variables=None, session=None, **kwargs):
        self.variables = variables or _global_variables()
        self.session = session or _default_session()
        self._bcast_op = broadcast_variables(self.variables, root_rank=0)
        self._eval_fn = self._to_numpy if _executing_eagerly() else self._eval_var
        self._assign_fn = self._assign_var if _IS_TF2 else self._load_var
        self._save_model()

        bcast_obj = broadcast_object_fn(session=session) if not _executing_eagerly() else broadcast_object

        def broadcast_object_with_session(obj):
            return bcast_obj(obj)

        super(TensorFlowState, self).__init__(bcast_object=broadcast_object_with_session,
                                              get_rank=rank,
                                              **kwargs)

    def save(self):
        self._save_model()
        super(TensorFlowState, self).save()

    def restore(self):
        self._load_model()
        super(TensorFlowState, self).restore()

    def sync(self):
        if self.session is not None:
            self.session.run(self._bcast_op)
        self._save_model()
        super(TensorFlowState, self).sync()

    def _save_model(self):
        self._values = [self._eval_fn(var) for var in self.variables]

    def _eval_var(self, var):
        return var.eval(self.session)

    def _to_numpy(self, var):
        return var.numpy()

    def _load_model(self):
        for var, value in zip(self.variables, self._values):
            self._assign_fn(var, value)

    def _load_var(self, var, value):
        var.load(value, self.session)

    def _assign_var(self, var, value):
        var.assign(value)


__all__ = [
    'TensorFlowKerasState',
    'TensorFlowState',
    'run',
]
