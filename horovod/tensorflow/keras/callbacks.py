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


import tensorflow as tf
from tensorflow import keras

from horovod.common.util  import is_version_greater_equal_than

if is_version_greater_equal_than(tf.__version__, "2.6.0"):
    from keras import backend as K
else:
    from tensorflow.python.keras import backend as K

from horovod._keras import callbacks as _impl


class BroadcastGlobalVariablesCallback(_impl.BroadcastGlobalVariablesCallbackImpl, keras.callbacks.Callback):
    """
    Keras Callback that will broadcast all global variables from root rank
    to all other processes during initialization.

    This is necessary to ensure consistent initialization of all workers when
    training is started with random weights or restored from a checkpoint.
    """

    def __init__(self, root_rank, device=''):
        """
        Construct a new BroadcastGlobalVariablesCallback that will broadcast all
        global variables from root rank to all other processes during initialization.

        Args:
            root_rank: Rank that will send data, other ranks will receive data.
            device: Device to be used for broadcasting. Uses GPU by default
                    if Horovod was build with HOROVOD_GPU_OPERATIONS.
        """
        super(BroadcastGlobalVariablesCallback, self).__init__(K, root_rank, device)


class MetricAverageCallback(_impl.MetricAverageCallbackImpl, keras.callbacks.Callback):
    """
    Keras Callback that will average metrics across all processes at the
    end of the epoch. Useful in conjuction with ReduceLROnPlateau,
    TensorBoard and other metrics-based callbacks.

    Note: This callback must be added to the callback list before the
    ReduceLROnPlateau, TensorBoard or other metrics-based callbacks.
    """

    def __init__(self, device=''):
        """
        Construct a new MetricAverageCallback that will average metrics
        across all processes at the end of the epoch.

        Args:
            device: Device to be used for allreduce. Uses GPU by default
                    if Horovod was build with HOROVOD_GPU_OPERATIONS.
        """
        super(MetricAverageCallback, self).__init__(K, device)


class LearningRateScheduleCallback(_impl.LearningRateScheduleCallbackImpl, keras.callbacks.Callback):
    """
    LearningRateScheduleCallback sets learning rate between epochs `start_epoch` and
    `end_epoch` to be `initial_lr * multiplier`.  `multiplier` can be a constant or
    a function `f(epoch) = lr'`.

    If `multiplier` is a function and `staircase=True`, learning rate adjustment will
    happen at the beginning of each epoch and the epoch passed to the `multiplier`
    function will be an integer.

    If `multiplier` is a function and `staircase=False`, learning rate adjustment will
    happen at the beginning of each batch and the epoch passed to the `multiplier`
    function will be a floating number: `epoch' = epoch + batch / steps_per_epoch`.
    This functionality is useful for smooth learning rate adjustment schedulers, such
    as `LearningRateWarmupCallback`.

    `initial_lr` is the learning rate of the model optimizer at the start of the training.
    """

    def __init__(self, initial_lr, multiplier, start_epoch=0, end_epoch=None, staircase=True,
                 momentum_correction=True, steps_per_epoch=None):
        """
        Construct a new LearningRateScheduleCallback.

        Args:
            initial_lr: Initial learning rate at the start of training.
            multiplier: A constant multiplier or a function `f(epoch) = lr'`
            start_epoch: The first epoch this adjustment will be applied to. Defaults to 0.
            end_epoch: The epoch this adjustment will stop applying (exclusive end).
                       Defaults to None.
            staircase: Whether to adjust learning rate at the start of epoch (`staircase=True`)
                       or at the start of every batch (`staircase=False`).
            momentum_correction: Apply momentum correction to optimizers that have momentum.
                                 Defaults to True.
            steps_per_epoch: The callback will attempt to autodetect number of batches per
                             epoch with Keras >= 2.0.0. Provide this value if you have an older
                             version of Keras.
        """
        super(LearningRateScheduleCallback, self).__init__(K, initial_lr, multiplier, start_epoch, end_epoch,
                                                           staircase, momentum_correction, steps_per_epoch)


class LearningRateWarmupCallback(_impl.LearningRateWarmupCallbackImpl, keras.callbacks.Callback):
    """
    Implements gradual learning rate warmup:

        `lr = initial_lr / hvd.size()` ---> `lr = initial_lr`

    `initial_lr` is the learning rate of the model optimizer at the start of the training.

    This technique was described in the paper "Accurate, Large Minibatch SGD: Training
    ImageNet in 1 Hour". See https://arxiv.org/pdf/1706.02677.pdf for details.

    Math recap:

    .. math::

        epoch &= full\\_epochs + \\frac{batch}{steps\\_per\\_epoch}

        lr'(epoch) &= \\frac{lr}{size} * (\\frac{size - 1}{warmup} * epoch + 1)

        lr'(epoch = 0) &= \\frac{lr}{size}

        lr'(epoch = warmup) &= lr
    """

    def __init__(self, initial_lr, warmup_epochs=5, momentum_correction=True, steps_per_epoch=None,
                 verbose=0):
        """
        Construct a new LearningRateWarmupCallback that will gradually warm up the learning rate.

        Args:
            initial_lr: Initial learning rate at the start of training.
            warmup_epochs: The number of epochs of the warmup phase. Defaults to 5.
            momentum_correction: Apply momentum correction to optimizers that have momentum.
                                 Defaults to True.
            steps_per_epoch: The callback will attempt to autodetect number of batches per
                             epoch with Keras >= 2.0.0. Provide this value if you have an older
                             version of Keras.
            verbose: verbosity mode, 0 or 1.
        """
        super(LearningRateWarmupCallback, self).__init__(K, initial_lr, warmup_epochs, momentum_correction,
                                                         steps_per_epoch, verbose)


class BestModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self,
                 monitor='val_loss',
                 verbose=0,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch'):
        super(BestModelCheckpoint, self).__init__(filepath=None,
                                                  monitor=monitor,
                                                  verbose=verbose,
                                                  save_best_only=True,
                                                  save_weights_only=save_weights_only,
                                                  mode=mode,
                                                  save_freq=save_freq)
