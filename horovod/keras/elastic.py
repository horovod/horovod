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

import keras

from horovod._keras import elastic as _impl
from horovod.tensorflow.elastic import TensorFlowKerasState, run


class KerasState(TensorFlowKerasState):
    """State representation of a `keras` model and optimizer.

    Args:
        model: Keras model.
        optimizer: Optional optimizer, can be compiled into model instead.
        kwargs: Additional properties to sync, will be exposed as attributes of the object.
    """
    def __init__(self, model, optimizer=None, **kwargs):
        super(KerasState, self).__init__(model, optimizer=optimizer, backend=keras.backend, **kwargs)


class CommitStateCallback(_impl.CommitStateCallbackImpl, keras.callbacks.Callback):
    """
    Keras Callback that will commit the `state` object every `batches_per_commit`
    batches at the end of each batch.
    """

    def __init__(self, state, batches_per_commit=1):
        """
        Constructs a new CommitStateCallback.

        Args:
            state: `horovod.common.elastic.State` object to be committed.
            batches_per_commit: Number of batches to complete between each commit (default: 1).
        """
        super(CommitStateCallback, self).__init__(keras.backend, state, batches_per_commit)


class UpdateBatchStateCallback(_impl.UpdateBatchStateCallbackImpl, keras.callbacks.Callback):
    """
    Keras Callback that will update the value of `state.batch` with the current batch number at
    the end of each batch. Batch will reset to 0 at the end of each epoch.

    If `steps_per_epoch` is set, then this callback will also ensure that the number of steps
    in the first epoch following a reset is shortened by the number of batches already processed.
    """

    def __init__(self, state):
        """
        Constructs a new UpdateBatchStateCallback.

        Args:
            state: `horovod.common.elastic.State` object to be updated.
        """
        super(UpdateBatchStateCallback, self).__init__(keras.backend, state)


class UpdateEpochStateCallback(_impl.UpdateEpochStateCallbackImpl, keras.callbacks.Callback):
    """
    Keras Callback that will update the value of `state.epoch` with the current epoch number at
    the end of each epoch.
    """

    def __init__(self, state):
        """
        Constructs a new UpdateEpochStateCallback.

        Args:
            state: `horovod.common.elastic.State` object to be updated.
        """
        super(UpdateEpochStateCallback, self).__init__(keras.backend, state)


__all__ = [
    'KerasState',
    'CommitStateCallback',
    'UpdateBatchStateCallback',
    'UpdateEpochStateCallback',
    'run',
]
