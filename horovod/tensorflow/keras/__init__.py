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

import inspect
import warnings

from packaging import version

import tensorflow as tf

from tensorflow import keras

from horovod.common.util  import is_version_greater_equal_than


if is_version_greater_equal_than(tf.__version__, "2.6.0"):
    from keras import backend as K
else:
    from tensorflow.python.keras import backend as K

from horovod.tensorflow import init
from horovod.tensorflow import shutdown
from horovod.tensorflow import is_initialized, start_timeline, stop_timeline
from horovod.tensorflow import size, local_size, cross_size, rank, local_rank, cross_rank
from horovod.tensorflow import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.tensorflow import gloo_enabled, gloo_built
from horovod.tensorflow.mpi_ops import ProcessSet, global_process_set, add_process_set, remove_process_set
from horovod.tensorflow import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
from horovod.tensorflow import Average, Sum
from horovod.tensorflow.compression import Compression


import horovod._keras as _impl
from horovod.tensorflow.keras import callbacks, elastic


try:
    # In later versions of TensorFlow, optimizers are spread across multiple modules. This set is used to distinguish
    # stock optimizers that come with tf.keras from custom optimizers that may need to be wrapped specially.
    optimizer_type = _impl.get_keras_optimizer_base_type(tf.keras)

    _OPTIMIZER_MODULES = set([obj.__module__ for name, obj in inspect.getmembers(tf.keras.optimizers)
                              if isinstance(obj, type(optimizer_type))])
except:
    _OPTIMIZER_MODULES = set()


def DistributedOptimizer(optimizer, name=None,
                         device_dense='', device_sparse='',
                         compression=Compression.none,
                         sparse_as_dense=False,
                         gradient_predivide_factor=1.0,
                         op=Average,
                         backward_passes_per_step=1,
                         average_aggregated_gradients=False,
                         num_groups=0,
                         groups=None,
                         process_set=global_process_set,
                         scale_local_gradients=True):
    """
    An optimizer that wraps another keras.optimizers.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.

    Args:
        optimizer: Optimizer to use for computing gradients and applying updates.
        name: Optional name prefix for the operations created when applying
              gradients. Defaults to "Distributed" followed by the provided
              optimizer type.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was build with HOROVOD_GPU_OPERATIONS.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was build with HOROVOD_GPU_OPERATIONS.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        sparse_as_dense: Treat all sparse gradients as dense tensors.  This can
                         help improve performance and memory utilization if
                         the original sparse gradient has high density.
                         Defaults to false.
        gradient_predivide_factor: gradient_predivide_factor splits the averaging
                                   before and after the sum. Gradients are scaled by
                                   1.0 / gradient_predivide_factor before the sum and
                                   gradient_predivide_factor / size after the sum.
        op: The reduction operation to use when combining gradients across
            different ranks. Defaults to Average.
        backward_passes_per_step: Number of backward passes to perform before calling
                                  hvd.allreduce. This allows accumulating updates over
                                  multiple mini-batches before reducing and applying them.
        average_aggregated_gradients: Whether to average the aggregated gradients that
                                      have been accumulated over multiple mini-batches.
                                      If true divides gradient updates by
                                      backward_passes_per_step.
                                      Only applicable for backward_passes_per_step > 1.
        num_groups: Number of groups to assign gradient allreduce ops to for explicit
                    grouping. Defaults to no explicit groups.
        groups: The parameter to group the gradient allreduce ops. Accept values is a
                non-negative integer or a list of list of tf.Variable.
                If groups is a non-negative integer, it is the number of groups to assign
                gradient allreduce ops to for explicit grouping.
                If groups is a list of list of tf.Variable. Variables in the same
                inner list will be assigned to the same group, while parameter that does
                not appear in any list will form a group itself.
                Defaults as None, which is no explicit groups.
        process_set: Gradients will only be reduced over Horovod processes belonging
                   to this process set. Defaults to the global process set.
        scale_local_gradients: Whether to scale the gradients of local variables. Default is set to True.

    """
    if gradient_predivide_factor != 1.0 and rocm_built():
            raise ValueError('gradient_predivide_factor not supported yet with ROCm')

    if op != Average and op != Sum:
        raise ValueError('op currently only supports Average and Sum')

    if num_groups != 0:
        warnings.warn('Parameter `num_groups` has been replaced by `groups` '
                      'and will be removed in v0.23.0.', DeprecationWarning)
        if groups is None:
            groups = num_groups

    if groups is not None:
        if not (isinstance(groups, list) or groups > 0):
            raise ValueError('groups should be a non-negative integer or '
                            'a list of list of tf.Variable.')

    return _impl.create_distributed_optimizer(
        keras=keras,
        optimizer=optimizer,
        name=name,
        device_dense=device_dense,
        device_sparse=device_sparse,
        compression=compression,
        sparse_as_dense=sparse_as_dense,
        gradient_predivide_factor=gradient_predivide_factor,
        op=op,
        backward_passes_per_step=backward_passes_per_step,
        average_aggregated_gradients=average_aggregated_gradients,
        groups=groups,
        process_set=process_set,
        scale_local_gradients=scale_local_gradients
    )


def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.

    Arguments:
        root_rank: Rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    return _impl.broadcast_global_variables(K, root_rank)


def allreduce(value, name=None, average=None,
              prescale_factor=1.0,
              postscale_factor=1.0,
              op=None,
              compression=Compression.none):
    """
    Perform an allreduce on a tensor-compatible value.

    Arguments:
        value: A tensor-compatible value to reduce.
               The shape of the input must be identical across all ranks.
        name: Optional name for the constants created by this operation.
        average:
            .. warning:: .. deprecated:: 0.19.0

               Use `op` instead. Will be removed in v0.21.0.

        prescale_factor: Multiplicative factor to scale tensor before allreduce.
        postscale_factor: Multiplicative factor to scale tensor after allreduce.
        op: The reduction operation to combine tensors across different ranks.
            Defaults to Average if None is given.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
    """
    return _impl.allreduce(
        backend=K,
        value=value,
        name=name,
        average=average,
        prescale_factor=prescale_factor,
        postscale_factor=postscale_factor,
        op=op,
        compression=compression)


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


def reducescatter(value, name=None, op=Average):
    """
    Perform a reducescatter on a tensor-compatible value.

    Arguments:
        value: A tensor-compatible value to reduce and scatter.
               The shape of the input must be identical across all ranks.
        name: Optional name for the constants created by this operation.
        op: The reduction operation to combine tensors across different ranks.
            Defaults to Average.
    """
    return _impl.reducescatter(K, value, name, op)


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
    return _impl.load_model(keras, wrap_optimizer, _OPTIMIZER_MODULES, filepath, custom_optimizers, custom_objects)
