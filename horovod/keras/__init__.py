# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
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
import keras.backend as K
import tensorflow as tf

import horovod.tensorflow as hvd
from horovod.common import init
from horovod.common import shutdown
from horovod.common import size
from horovod.common import local_size
from horovod.common import rank
from horovod.common import local_rank
from horovod.common import mpi_threads_supported
from horovod.keras import callbacks


class _DistributedOptimizer(keras.optimizers.Optimizer):
    """
    This is an internal implementation class of Distributed Optimizer, not to be
    directly instantiated by end-users. See horovod.keras.DistributedOptimizer.
    """

    def __init__(self, name, device_dense, device_sparse, **kwargs):
        if name is None:
            name = "Distributed%s" % self.__class__.__base__.__name__
        self._name = name
        self._device_dense = device_dense
        self._device_sparse = device_sparse
        super(self.__class__, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
        """
        Compute gradients of all trainable variables.

        See Optimizer.get_gradients() for more info.

        In DistributedOptimizer, get_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = super(self.__class__, self).get_gradients(loss, params)
        if hvd.size() > 1:
            averaged_gradients = []
            with tf.name_scope(self._name + "_Allreduce"):
                for grad in gradients:
                    if grad is not None:
                        avg_grad = hvd.allreduce(grad, device_dense=self._device_dense,
                                                 device_sparse=self._device_sparse)
                        averaged_gradients.append(avg_grad)
                    else:
                        averaged_gradients.append(None)
                return averaged_gradients
        else:
            return gradients


def DistributedOptimizer(optimizer, name=None, device_dense='', device_sparse=''):
    """
    An optimizer that wraps another keras.optimizers.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Args:
        optimizer: Optimizer to use for computing gradients and applying updates.
        name: Optional name prefix for the operations created when applying
              gradients. Defaults to "Distributed" followed by the provided
              optimizer type.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was build with HOROVOD_GPU_ALLREDUCE.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was build with HOROVOD_GPU_ALLGATHER.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, device_dense, device_sparse, **optimizer.get_config())


def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.

    Arguments:
        root_rank: Rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    bcast_op = hvd.broadcast_global_variables(root_rank)
    return K.get_session().run(bcast_op)


def allreduce(value, name=None, average=True):
    """
    Perform an allreduce on a tensor-compatible value.

    Arguments:
        value: A tensor-compatible value to reduce.
               The shape of the input must be identical across all ranks.
        name: Optional name for the constants created by this operation.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
    """
    allreduce_op = hvd.allreduce(tf.constant(value, name=name), average=average)
    return K.get_session().run(allreduce_op)


def allgather(value, name=None):
    """
    Perform an allgather on a tensor-compatible value.

    The concatenation is done on the first dimension, so the input values on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Arguments:
        value: A tensor-compatible value to gather.
        name: Optional name prefix for the constants created by this operation.
    """
    allgather_op = hvd.allgather(tf.constant(value, name=name))
    return K.get_session().run(allgather_op)


def broadcast(value, root_rank, name=None):
    """
    Perform a broadcast on a tensor-compatible value.

    Arguments:
        value: A tensor-compatible value to reduce.
               The shape of the input must be identical across all ranks.
        root_rank: Rank of the process from which global variables will be
                   broadcasted to all other processes.
        name: Optional name for the constants created by this operation.
    """
    bcast_op = hvd.broadcast(tf.constant(value, name=name), root_rank)
    return K.get_session().run(bcast_op)


def load_model(filepath, custom_optimizers=None, custom_objects=None):
    """
    Loads a saved Keras model with a Horovod DistributedOptimizer.

    The DistributedOptimizer will wrap the underlying optimizer used to train
    the saved model, so that the optimizer state (params and weights) will
    be picked up for retraining.

    By default, all optimizers in the module `keras.optimizers` will be loaded
    and wrapped without needing to specify any `custom_optimizers` or
    `custom_objects`.

    # Arguments
        filepath: One of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        custom_optimizers: Optional list of Optimizer subclasses to support
            during loading.
        custom_objects: Optional dictionary mapping names (strings) to custom
            classes or functions to be considered during deserialization.

    # Returns
        A Keras model instance.

    # Raises
        ImportError: If h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    def wrap_optimizer(cls):
        return lambda **kwargs: DistributedOptimizer(cls(**kwargs))

    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ == 'keras.optimizers'
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
