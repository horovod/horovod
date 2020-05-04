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

from horovod.tensorflow import init
from horovod.tensorflow import shutdown
from horovod.tensorflow import size
from horovod.tensorflow import local_size
from horovod.tensorflow import rank
from horovod.tensorflow import local_rank
from horovod.tensorflow import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.tensorflow import gloo_enabled, gloo_built
from horovod.tensorflow import nccl_built, ddl_built, ccl_built
from horovod.tensorflow import Compression

from horovod.keras import callbacks
import horovod._keras as _impl


def DistributedOptimizer(optimizer, name=None,
                         device_dense='', device_sparse='',
                         compression=Compression.none,
                         sparse_as_dense=False):
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
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        sparse_as_dense: Treat all sparse gradients as dense tensors.  This can
                         help improve performance and memory utilization if
                         the original sparse gradient has high density.
                         Defaults to false.
    """
    return _impl.create_distributed_optimizer(keras, optimizer, name,
                                              device_dense, device_sparse, compression,
                                              sparse_as_dense)


def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.

    Arguments:
        root_rank: Rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    return _impl.broadcast_global_variables(K, root_rank)


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
    return _impl.allreduce(K, value, name, average)


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
    return _impl.allgather(K, value, name)


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
    return _impl.broadcast(K, value, root_rank, name)


def load_model(filepath, custom_optimizers=None, custom_objects=None, compression=Compression.none):
    """
    Loads a saved Keras model with a Horovod DistributedOptimizer.

    The DistributedOptimizer will wrap the underlying optimizer used to train
    the saved model, so that the optimizer state (params and weights) will
    be picked up for retraining.

    By default, all optimizers in the module `keras.optimizers` will be loaded
    and wrapped without needing to specify any `custom_optimizers` or
    `custom_objects`.

    Arguments:
        filepath: One of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        custom_optimizers: Optional list of Optimizer subclasses to support
            during loading.
        custom_objects: Optional dictionary mapping names (strings) to custom
            classes or functions to be considered during deserialization.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.

    Returns:
        A Keras model instance.

    Raises:
        ImportError: If h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    def wrap_optimizer(cls):
        return lambda **kwargs: DistributedOptimizer(cls(**kwargs), compression=compression)
    optimizer_modules = {keras.optimizers.Optimizer.__module__}
    return _impl.load_model(keras, wrap_optimizer, optimizer_modules, filepath, custom_optimizers, custom_objects)
