# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
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
# pylint: disable=g-short-docstring-punctuation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common.util import check_extension

check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')

from horovod.tensorflow.compression import Compression
from horovod.tensorflow.mpi_ops import allgather, broadcast, _allreduce
from horovod.tensorflow.mpi_ops import init, shutdown
from horovod.tensorflow.mpi_ops import size, local_size, rank, local_rank
from horovod.tensorflow.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.tensorflow.mpi_ops import gloo_enabled, gloo_built
from horovod.tensorflow.util import _executing_eagerly, _make_subgraph, _cache

import tensorflow as tf


def allreduce(tensor, average=True, device_dense='', device_sparse='',
              compression=Compression.none):
    """Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
                The shape of the input must be identical across all ranks.
        average: If True, computes the average over all ranks.
                 Otherwise, computes the sum over all ranks.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was built with HOROVOD_GPU_ALLREDUCE.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was built with HOROVOD_GPU_ALLGATHER.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.

    Returns:
        A tensor of the same shape and type as `tensor`, summed across all
        processes.
    """
    if isinstance(tensor, tf.IndexedSlices):
        with tf.device(device_sparse):
            # For IndexedSlices, do two allgathers instead of an allreduce.
            horovod_size = tf.cast(size(), tensor.values.dtype)
            values = allgather(tensor.values)
            indices = allgather(tensor.indices)

            # To make this operation into an average, divide allgathered values by
            # the Horovod size.
            new_values = (values / horovod_size) if average else values
        return tf.IndexedSlices(new_values, indices,
                                dense_shape=tensor.dense_shape)
    else:
        with tf.device(device_dense):
            horovod_size = tf.cast(size(), dtype=tensor.dtype)
            tensor_compressed, ctx = compression.compress(tensor)
            summed_tensor_compressed = _allreduce(tensor_compressed)
            summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
            new_tensor = (summed_tensor / horovod_size) if average else summed_tensor
        return new_tensor


@_cache
def _make_broadcast_group_fn():
    if _executing_eagerly():
        # Eager mode will parallelize independent control flow
        def broadcast_group(variables, root_rank):
            for var in variables:
                var.assign(broadcast(var, root_rank))

        return _make_subgraph(broadcast_group)
    else:
        # Graph mode requires an Op
        def broadcast_group(variables, root_rank):
            return tf.group(*[var.assign(broadcast(var, root_rank))
                              for var in variables])

        return broadcast_group


def broadcast_variables(variables, root_rank):
    """Broadcasts variables from root rank to all other processes.

    Arguments:
        variables: variables for broadcast
        root_rank: rank of the process from which global variables will be broadcasted
                   to all other processes.
    """
    broadcast_group = _make_broadcast_group_fn()
    return broadcast_group(variables, root_rank)


try:
    _global_variables = tf.global_variables
except AttributeError:
    try:
        _global_variables = tf.compat.v1.global_variables
    except AttributeError:
        _global_variables = None

if _global_variables is not None:
    def broadcast_global_variables(root_rank):
        """Broadcasts all global variables from root rank to all other processes.

        **NOTE:** deprecated in TensorFlow 2.0.

        Arguments:
            root_rank: rank of the process from which global variables will be broadcasted
                       to all other processes.
        """
        if _executing_eagerly():
            raise RuntimeError(
                "Eager Execution is not supported by `hvd.BroadcastGlobalVariablesHook`\n"
                "We recommend using `hvd.DistributedGradientTape` instead"
            )

        return broadcast_variables(_global_variables(), root_rank)

try:
    _get_default_graph = tf.get_default_graph
except AttributeError:
    try:
        _get_default_graph = tf.compat.v1.get_default_graph
    except AttributeError:
        _get_default_graph = None

try:
    _SessionRunHook = tf.estimator.SessionRunHook
except AttributeError:
    try:
        _SessionRunHook = tf.train.SessionRunHook
    except AttributeError:
        _SessionRunHook = None

if _SessionRunHook is not None and _get_default_graph is not None:
    class BroadcastGlobalVariablesHook(_SessionRunHook):
        """
        SessionRunHook that will broadcast all global variables from root rank
        to all other processes during initialization.

        This is necessary to ensure consistent initialization of all workers when
        training is started with random weights or restored from a checkpoint.

        **NOTE:** deprecated in TensorFlow 2.0.
        """

        def __init__(self, root_rank, device=''):
            """Construct a new BroadcastGlobalVariablesHook that will broadcast all
            global variables from root rank to all other processes during initialization.

            Args:
              root_rank:
                Rank that will send data, other ranks will receive data.
              device:
                Device to be used for broadcasting. Uses GPU by default
                if Horovod was built with HOROVOD_GPU_BROADCAST.
            """
            super(BroadcastGlobalVariablesHook, self).__init__()
            self.root_rank = root_rank
            self.bcast_op = None
            self.device = device

        def begin(self):
            if not self.bcast_op or self.bcast_op.graph != _get_default_graph():
                with tf.device(self.device):
                    self.bcast_op = broadcast_global_variables(self.root_rank)

        def after_create_session(self, session, coord):
            session.run(self.bcast_op)


@_cache
def _make_allreduce_grads_fn(name, device_dense, device_sparse,
                             compression, sparse_as_dense):
    def allreduce_grads(grads):
        with tf.name_scope(name + "_Allreduce"):
            if sparse_as_dense:
                grads = [tf.convert_to_tensor(grad)
                         if grad is not None and isinstance(grad, tf.IndexedSlices)
                         else grad for grad in grads]

            return [allreduce(grad,
                              device_dense=device_dense,
                              device_sparse=device_sparse,
                              compression=compression)
                    if grad is not None else grad
                    for grad in grads]

    if _executing_eagerly():
        return _make_subgraph(allreduce_grads)
    else:
        return allreduce_grads


try:
    # TensorFlow 2.x
    _LegacyOptimizer = tf.compat.v1.train.Optimizer
except AttributeError:
    try:
        # TensorFlow 1.x
        _LegacyOptimizer = tf.train.Optimizer
    except AttributeError:
        # Future TensorFlow versions
        _LegacyOptimizer = None

if _LegacyOptimizer is not None:
    class _DistributedOptimizer(_LegacyOptimizer):
        """An optimizer that wraps another tf.Optimizer, using an allreduce to
        average gradient values before applying gradients to model weights."""

        def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                    device_sparse='', compression=Compression.none,
                    sparse_as_dense=False):
            if name is None:
                name = "Distributed{}".format(type(optimizer).__name__)
            super(_DistributedOptimizer, self).__init__(name=name, use_locking=use_locking)

            self._optimizer = optimizer
            self._allreduce_grads = _make_allreduce_grads_fn(
                name, device_dense, device_sparse, compression, sparse_as_dense)

        def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.

            See Optimizer.compute_gradients() for more info.

            In DistributedOptimizer, compute_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = self._optimizer.compute_gradients(*args, **kwargs)
            if size() > 1:
                grads, vars = zip(*gradients)
                avg_grads = self._allreduce_grads(grads)
                return list(zip(avg_grads, vars))
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.apply_gradients(*args, **kwargs)

        def get_slot(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot(*args, **kwargs)

        def get_slot_names(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot_names(*args, **kwargs)

        def variables(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.variables(*args, **kwargs)


def DistributedOptimizer(optimizer, name=None, use_locking=False, device_dense='',
                         device_sparse='', compression=Compression.none,
                         sparse_as_dense=False):
    """Construct a new DistributedOptimizer, which uses another optimizer
    under the hood for computing single-process gradient values and
    applying gradient updates after the gradient values have been averaged
    across all the Horovod ranks.

    Args:
      optimizer:
        Optimizer to use for computing gradients and applying updates.
      name:
        Optional name prefix for the operations created when applying
        gradients. Defaults to "Distributed" followed by the provided
        optimizer type.
      use_locking:
        Whether to use locking when updating variables.
        See Optimizer.__init__ for more info.
      device_dense:
        Device to be used for dense tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_ALLREDUCE.
      device_sparse:
        Device to be used for sparse tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_ALLGATHER.
      compression:
        Compression algorithm used during allreduce to reduce the amount
        of data sent during each parameter update step.  Defaults to
        not using compression.
      sparse_as_dense:
        Treat all sparse gradients as dense tensors.  This can help improve
        performance and memory utilization if the original sparse gradient
        has high density.  Defaults to false.
    """
    if isinstance(optimizer, _LegacyOptimizer):
        return _DistributedOptimizer(optimizer, name, use_locking, device_dense,
                                     device_sparse, compression, sparse_as_dense)
    elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
        import horovod.tensorflow.keras as hvd_k
        return hvd_k.DistributedOptimizer(optimizer, name, device_dense, device_sparse,
                                          compression, sparse_as_dense)
    else:
        raise ValueError('Provided optimizer doesn\'t inherit from either legacy '
                         'TensorFlow or Keras optimizer: %s' % optimizer)


if hasattr(tf, 'GradientTape'):
    class _DistributedGradientTape(tf.GradientTape):
        def __init__(self, tape, device_dense, device_sparse, compression, sparse_as_dense,
                     persistent=False, watch_accessed_variables=True):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(persistent, watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)

            self._tape = tape
            self._allreduce_grads = _make_allreduce_grads_fn(
                'DistributedGradientTape', device_dense, device_sparse, compression,
                sparse_as_dense)

        def gradient(self, target, sources, output_gradients=None):
            gradients = super(self.__class__, self).gradient(target, sources, output_gradients)
            if size() > 1:
                return self._allreduce_grads(gradients)
            else:
                return gradients


    def DistributedGradientTape(gradtape, device_dense='', device_sparse='',
                                compression=Compression.none, sparse_as_dense=False):
        """A tape that wraps another tf.GradientTape, using an allreduce to
        average gradient values before applying gradients to model weights.

        Args:
          gradtape:
            GradientTape to use for computing gradients and applying updates.
          device_dense:
            Device to be used for dense tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_ALLREDUCE.
          device_sparse:
            Device to be used for sparse tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_ALLGATHER.
          compression:
            Compression algorithm used during allreduce to reduce the amount
            of data sent during each parameter update step.  Defaults to
            not using compression.
          sparse_as_dense:
            Treat all sparse gradients as dense tensors.  This can help improve
            performance and memory utilization if the original sparse gradient
            has high density.  Defaults to false.
        """
        cls = type(gradtape.__class__.__name__, (gradtape.__class__,),
                   dict(_DistributedGradientTape.__dict__))
        if hasattr(gradtape, '_watch_accessed_variables'):
            return cls(gradtape._tape, device_dense, device_sparse, compression,
                       sparse_as_dense, gradtape._persistent,
                       gradtape._watch_accessed_variables)
        else:
            return cls(gradtape._tape, device_dense, device_sparse, compression,
                       sparse_as_dense, gradtape._persistent)
