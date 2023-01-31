# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
# Modifications copyright Microsoft
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

from packaging import version
import os
import warnings

from horovod.common.util import check_extension, gpu_available, split_list

check_extension('horovod.tensorflow', 'HOROVOD_WITH_TENSORFLOW', __file__, 'mpi_lib')

from horovod.tensorflow import elastic
from horovod.tensorflow.compression import Compression
from horovod.tensorflow.functions import allgather_object, broadcast_object, broadcast_object_fn, broadcast_variables
from horovod.tensorflow.mpi_ops import allgather, grouped_allgather, broadcast, broadcast_, _allreduce, \
    _grouped_allreduce, alltoall, _reducescatter, _grouped_reducescatter
from horovod.tensorflow.mpi_ops import init, shutdown
from horovod.tensorflow.mpi_ops import is_initialized, start_timeline, stop_timeline
from horovod.tensorflow.mpi_ops import size, local_size, cross_size, rank, local_rank, cross_rank, is_homogeneous
from horovod.tensorflow.mpi_ops import rank_op, local_rank_op, size_op, local_size_op, process_set_included_op
from horovod.tensorflow.mpi_ops import mpi_threads_supported, mpi_enabled, mpi_built
from horovod.tensorflow.mpi_ops import gloo_enabled, gloo_built
from horovod.tensorflow.mpi_ops import nccl_built, ddl_built, ccl_built, cuda_built, rocm_built
from horovod.tensorflow.mpi_ops import ProcessSet, global_process_set, add_process_set, remove_process_set
from horovod.tensorflow.mpi_ops import Average, Sum, Adasum, Min, Max, Product
from horovod.tensorflow.mpi_ops import handle_average_backwards_compatibility, check_num_rank_power_of_2
from horovod.tensorflow.util import _executing_eagerly, _make_subgraph, _cache, vars_to_refs, refs_to_vars
from horovod.tensorflow.mpi_ops import join
from horovod.tensorflow.sync_batch_norm import SyncBatchNormalization
from horovod.tensorflow.gradient_aggregation import LocalGradientAggregationHelper
from horovod.common.util import support_non_legacy_keras_optimizers

import tensorflow as tf
_IS_TF2 = version.parse(tf.__version__) >= version.parse('2.0.0')

# @DEKHTIARJonathan: Do not remove, this fixes issues:
# - https://github.com/tensorflow/tensorflow/issues/38516
# - https://github.com/tensorflow/tensorflow/issues/39894
if tf.__version__.startswith('2.2.'):
  from tensorflow.python.keras.mixed_precision.experimental import device_compatibility_check
  device_compatibility_check.log_device_compatibility_check = lambda policy_name, skip_local: None


def allreduce(tensor, average=None, device_dense='', device_sparse='',
              compression=Compression.none, op=None,
              prescale_factor=1.0, postscale_factor=1.0,
              name=None, process_set=global_process_set,
              ignore_name_scope=False):
    """Perform an allreduce on a tf.Tensor or tf.IndexedSlices.

    This function performs a bandwidth-optimal ring allreduce on the input
    tensor. If the input is an tf.IndexedSlices, the function instead does an
    allgather on the values and the indices, effectively doing an allreduce on
    the represented tensor.

    Arguments:
        tensor: tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
                The shape of the input must be identical across all ranks.
        average:
            .. warning:: .. deprecated:: 0.19.0

                Use `op` instead. Will be removed in v1.0.

        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was built with HOROVOD_GPU_OPERATIONS.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was built with HOROVOD_GPU_OPERATIONS.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        op: The reduction operation to combine tensors across different ranks.
            Supported op values are Sum, Average, Min, Max, and Product. Defaults
            to Average if None is given.
        prescale_factor: Multiplicative factor to scale tensor before allreduce.
        postscale_factor: Multiplicative factor to scale tensor after allreduce.
        process_set: Process set object to limit this operation to a subset of
            Horovod processes. Default is the global process set.
        name: A name of the allreduce operation
        ignore_name_scope: If True, ignores any outer name scope applied by
                           TensorFlow in the name used by the Horovod operation.

    Returns:
        A tensor of the same shape and type as `tensor`, summed across all
        processes.
    """
    op = handle_average_backwards_compatibility(op, average)

    if isinstance(tensor, tf.IndexedSlices):
        # TODO: Need to fix this to actuall call Adasum
        if op == Adasum:
            raise NotImplementedError('The Adasum reduction does not currently support sparse tensors. As a '
                                      'workaround please pass sparse_as_dense=True to DistributedOptimizer')
        if op != Sum and op != Average:
            raise NotImplementedError('Only Sum and Average ops are supported with tf.IndexedSlices')


        with tf.device(device_sparse):
            # For IndexedSlices, do two allgathers instead of an allreduce.
            horovod_size = tf.cast(size_op(process_set_id=process_set.process_set_id)
                                   if int(os.environ.get("HOROVOD_ELASTIC", 0)) else process_set.size(),
                                   dtype=tensor.values.dtype)
            values = allgather(tensor.values, process_set=process_set, ignore_name_scope=ignore_name_scope)
            indices = allgather(tensor.indices, process_set=process_set, ignore_name_scope=ignore_name_scope)

            # To make this operation into an average, divide allgathered values by
            # the Horovod size.
            new_values = (values / horovod_size) if op == Average else values
            if (prescale_factor != 1.0 or postscale_factor != 1.0):
                raise NotImplementedError("Pre/postscale_factor are not supported with tf.IndexedSlices")
        return tf.IndexedSlices(new_values, indices,
                                dense_shape=tensor.dense_shape)
    else:
        average_in_framework = False
        if rocm_built():
            # For ROCm, perform averaging at framework level
            average_in_framework = op == Average or op == Adasum
            op = Sum if op == Average else op

        with tf.device(device_dense):
            horovod_size = tf.cast(size_op(process_set_id=process_set.process_set_id)
                                   if int(os.environ.get("HOROVOD_ELASTIC", 0)) else process_set.size(),
                                   dtype=tensor.dtype)
            tensor_compressed, ctx = compression.compress(tensor)
            summed_tensor_compressed = _allreduce(tensor_compressed, op=op,
                                                  prescale_factor=prescale_factor,
                                                  postscale_factor=postscale_factor,
                                                  name=name, process_set=process_set,
                                                  ignore_name_scope=ignore_name_scope)
            summed_tensor = compression.decompress(summed_tensor_compressed, ctx)
            if op == Adasum:
                if process_set != global_process_set:
                    raise NotImplementedError("Adasum does not support non-global process sets yet.")
                if 'CPU' not in tensor.device and gpu_available('tensorflow'):
                    if nccl_built():
                        if not is_homogeneous:
                            raise NotImplementedError(
                                'Running GPU Adasum on heterogeneous cluster is not supported yet.')
                        elif not check_num_rank_power_of_2(int(size() / local_size())):
                            raise NotImplementedError(
                                'Running GPU Adasum with non-power of 2 nodes is not supported yet.')
                        if rocm_built():
                            horovod_local_size = tf.cast(local_size_op() if int(os.environ.get("HOROVOD_ELASTIC", 0)) else local_size(),
                                                         dtype=tensor.dtype)
                            new_tensor = summed_tensor / horovod_local_size
                        else:
                            new_tensor = summed_tensor
                    else:
                        warnings.warn('Adasum reduction does not currently support GPU reduction using MPI. Tensors '
                                      'are copied to CPU memory instead. To use Adasum for GPU reduction, please '
                                      'compile Horovod with HOROVOD_GPU_OPERATIONS=NCCL.')
                        new_tensor = summed_tensor
                else:
                    if not check_num_rank_power_of_2(size()):
                        raise NotImplementedError('Running Adasum with non-power of 2 ranks is not supported yet.')
                    new_tensor = summed_tensor
            else:
                if rocm_built():
                    new_tensor = (summed_tensor / horovod_size) if average_in_framework else summed_tensor
                else:
                    new_tensor = summed_tensor
        return new_tensor


def reducescatter(tensor, device_dense='', compression=Compression.none, op=Average,
                  name=None, process_set=global_process_set,
                  ignore_name_scope=False,
                  prescale_factor=1.0, postscale_factor=1.0):
    """Perform a reducescatter on a tf.Tensor.

    This function performs a bandwidth-optimal reduce and scatter on the input
    tensor.

    Arguments:
        tensor: tf.Tensor or tf.Variable to reduce.
                The shape of the input must be identical across all ranks.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was built with HOROVOD_GPU_REDUCESCATTER.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        op: The reduction operation to combine tensors across different ranks.
            Defaults to Average.
        process_set: Process set object to limit this operation to a subset of
            Horovod processes. Default is the global process set.
        name: A name of the reduce_scatter operation
        ignore_name_scope: If True, ignores any outer name scope applied by
                           TensorFlow in the name used by the Horovod operation.
        prescale_factor: Multiplicative factor to scale tensor before reducescatter.
        postscale_factor: Multiplicative factor to scale tensor after reducescatter.

    Returns:
        A tensor of the same rank and type as `tensor`, summed across all processes.
        The shape is identical to the input shape, except for the first dimension,
        which will be divided across the different Horovod processes.
    """
    if rocm_built() and op == Average:
        # Need to average in framework code
        true_op = Sum
    else:
        true_op = op

    with tf.device(device_dense):
        tensor_compressed, ctx = compression.compress(tensor)
        reduced_tensor_compressed = _reducescatter(tensor_compressed, op=true_op, name=name, process_set=process_set,
                                                   ignore_name_scope=ignore_name_scope, prescale_factor=prescale_factor,
                                                   postscale_factor=postscale_factor)
        reduced_tensor = compression.decompress(reduced_tensor_compressed, ctx)
        if op == Average and true_op == Sum:
            horovod_size = tf.cast(size_op(process_set_id=process_set.process_set_id)
                                   if int(os.environ.get("HOROVOD_ELASTIC", 0)) else process_set.size(),
                                   dtype=tensor.dtype)
            new_tensor = reduced_tensor / horovod_size
            return new_tensor
        else:
            return reduced_tensor


def grouped_allreduce(tensors, average=None, device_dense='', device_sparse='',
                      compression=Compression.none, op=None,
                      prescale_factor=1.0, postscale_factor=1.0,
                      process_set=global_process_set,
                      ignore_name_scope=False,
                      name=None):
    """Perform grouped allreduces on a sequence of tf.Tensor or tf.IndexedSlices.

    Arguments:
        tensors: Sequence of tf.Tensor, tf.Variable, or tf.IndexedSlices to reduce.
                 The tensor type and shape must be the same on all Horovod processes
                 for tensors sharing positions in `tensors`.
        average:
            .. warning:: .. deprecated:: 0.19.0

                Use `op` instead. Will be removed in v1.0.

        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was built with HOROVOD_GPU_OPERATIONS.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was built with HOROVOD_GPU_OPERATIONS.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        op: The reduction operation to combine tensors across different ranks.
            Supported op values are Sum, Average, Min, Max, and Product. Defaults
            to Average if None is given.
        prescale_factor: Multiplicative factor to scale tensors before allreduce.
        postscale_factor: Multiplicative factor to scale tensors after allreduce.
        process_set: Process set object to limit this operation to a subset of
            Horovod processes. Default is the global process set.
        name: A name of the reduce_scatter operation
        ignore_name_scope: If True, ignores any outer name scope applied by
                           TensorFlow in the name used by the Horovod operation.

    Returns:
        A list of tensors of the same shape and type as those in `tensors`,
        reduced across all processes.
    """
    if not tensors:
        return tensors

    op = handle_average_backwards_compatibility(op, average)

    average_in_framework = False
    if rocm_built():
        # For ROCm, perform averaging at framework level
        average_in_framework = op == Average or op == Adasum
        op = Sum if op == Average else op

    # Split list of tensors into indexed slices and normal tensors to handle separately.
    tensor_list, tensor_list_idx = [], []
    indexed_slices_list, indexed_slices_list_idx = [], []
    new_tensors_merged = [None] * len(tensors)

    for i, t in enumerate(tensors):
        if isinstance(t, tf.IndexedSlices):
            indexed_slices_list.append(t)
            indexed_slices_list_idx.append(i)
        else:
            tensor_list.append(t)
            tensor_list_idx.append(i)

    if indexed_slices_list:
        # TODO: Need to fix this to actuall call Adasum
        if op == Adasum:
            raise NotImplementedError('The Adasum reduction does not currently support sparse tensors. As a '
                                      'workaround please pass sparse_as_dense=True to DistributedOptimizer')
        if op != Sum and op != Average:
            raise NotImplementedError('Only Sum and Average ops are supported with tf.IndexedSlices')

        with tf.device(device_sparse):
            new_values = []
            new_indices = []
            # For IndexedSlices, do two grouped_allgathers instead of a grouped_allreduce.
            values = grouped_allgather([x.values for x in indexed_slices_list], process_set=process_set,
                                       ignore_name_scope=ignore_name_scope)
            new_indices = grouped_allgather([x.indices for x in indexed_slices_list], process_set=process_set,
                                            ignore_name_scope=ignore_name_scope)

            # To make this operation into an average, divide allgathered values by
            # the Horovod size.
            for x in values:
                horovod_size = tf.cast(size_op(process_set_id=process_set.process_set_id)
                                       if int(os.environ.get("HOROVOD_ELASTIC", 0)) else process_set.size(),
                                       dtype=x.dtype)
                new_values.append(x / horovod_size if op == Average else x)
            if (prescale_factor != 1.0 or postscale_factor != 1.0):
                raise NotImplementedError("Pre/postscale_factor are not supported with tf.IndexedSlices")
        new_indexed_slices = [tf.IndexedSlices(x, i,
                                               dense_shape=t.dense_shape) for x,i,t in zip(new_values, new_indices, tensors)]

        for idx, indexed_slice in zip(indexed_slices_list_idx, new_indexed_slices):
            new_tensors_merged[idx] = indexed_slice

    if tensor_list:
        with tf.device(device_dense):
            tensors_compressed, ctxs = zip(*[compression.compress(tensor) for tensor in tensor_list])
            summed_tensors_compressed = _grouped_allreduce(tensors_compressed, op=op,
                                                           prescale_factor=prescale_factor,
                                                           postscale_factor=postscale_factor,
                                                           process_set=process_set,
                                                           name=name,
                                                           ignore_name_scope=ignore_name_scope)
            summed_tensors = [compression.decompress(t, ctx) for t, ctx in zip(summed_tensors_compressed, ctxs)]
            if op == Adasum:
                if process_set != global_process_set:
                    raise NotImplementedError("Adasum does not support non-global process sets yet.")
                if 'CPU' not in tensor.device and gpu_available('tensorflow'):
                    if nccl_built():
                        if not is_homogeneous:
                            raise NotImplementedError(
                                'Running GPU Adasum on heterogeneous cluster is not supported yet.')
                        elif not check_num_rank_power_of_2(int(size() / local_size())):
                            raise NotImplementedError(
                                'Running GPU Adasum with non-power of 2 nodes is not supported yet.')
                        if rocm_built():
                            new_tensors = []
                            for tensor in summed_tensors:
                              horovod_local_size = tf.cast(local_size_op() if int(os.environ.get("HOROVOD_ELASTIC", 0)) else local_size(),
                                                           dtype=tensor.dtype)
                              new_tensors += tensor / horovod_local_size
                        else:
                            new_tensors = summed_tensors
                    else:
                        warnings.warn('Adasum reduction does not currently support GPU reduction using MPI. Tensors '
                                      'are copied to CPU memory instead. To use Adasum for GPU reduction, please '
                                      'compile Horovod with HOROVOD_GPU_OPERATIONS=NCCL.')
                        new_tensors = summed_tensors
                else:
                    if not check_num_rank_power_of_2(size()):
                        raise NotImplementedError('Running Adasum with non-power of 2 ranks is not supported yet.')
                    new_tensors = summed_tensors
            else:
                if rocm_built():
                    new_tensors = []
                    for tensor in summed_tensors:
                        horovod_size = tf.cast(size_op(process_set_id=process_set.process_set_id)
                                               if int(os.environ.get("HOROVOD_ELASTIC", 0)) else process_set.size(),
                                               dtype=tensor.dtype)
                        new_tensors += (tensor / horovod_size) if average_in_framework else tensor
                else:
                    new_tensors = summed_tensors

        for idx, tensor in zip(tensor_list_idx, new_tensors):
            new_tensors_merged[idx] = tensor

    return new_tensors_merged


def _allreduce_cond(tensor, *args, process_set=global_process_set, **kwargs):
    def allreduce_fn():
        return allreduce(tensor, *args, process_set=process_set, **kwargs)

    def id_fn():
        return tensor

    return tf.cond(tf.logical_and(
        tf.equal(process_set_included_op(process_set.process_set_id), 1),
        tf.greater(size_op(process_set.process_set_id), 1))
                   if int(os.environ.get("HOROVOD_ELASTIC", 0)) else (
        tf.convert_to_tensor(process_set.included() and process_set.size() > 1)),
                   allreduce_fn, id_fn)


def _grouped_allreduce_cond(tensors, *args, process_set=global_process_set, **kwargs):
    def allreduce_fn():
        return grouped_allreduce(tensors, *args, process_set=process_set, **kwargs)

    def id_fn():
        return tensors

    return tf.cond(tf.logical_and(
        tf.equal(process_set_included_op(process_set.process_set_id), 1),
        tf.greater(size_op(process_set.process_set_id), 1))
                   if int(os.environ.get("HOROVOD_ELASTIC", 0)) else (
        tf.convert_to_tensor(process_set.included() and process_set.size() > 1)),
                   allreduce_fn, id_fn)


def grouped_reducescatter(tensors, device_dense='', compression=Compression.none, op=Average,
                          process_set=global_process_set, prescale_factor=1.0, postscale_factor=1.0):
    """Perform grouped reducescatters on a sequence of tf.Tensor.

    Arguments:
        tensors: Sequence of tf.Tensor or tf.Variable to reduce.
                 The shape must be the same on all Horovod processes
                 for inputs sharing positions in `tensors`.
        device_dense: Device to be used for dense tensors. Uses GPU by default
                      if Horovod was built with HOROVOD_GPU_OPERATIONS.
        device_sparse: Device to be used for sparse tensors. Uses GPU by default
                       if Horovod was built with HOROVOD_GPU_OPERATIONS.
        compression: Compression algorithm used to reduce the amount of data
                     sent and received by each worker node.  Defaults to not
                     using compression.
        op: The reduction operation to combine tensors across different ranks.
            Defaults to Average if None is given.
        process_set: Process set object to limit this operation to a subset of
            Horovod processes. Default is the global process set.
        prescale_factor: Multiplicative factor to scale tensors before reducescatter.
        postscale_factor: Multiplicative factor to scale tensors after reducescatter.

    Returns:
        A list of tensors of the same rank and type as those in `tensors`,
        reduced across all processes. For each returned tensor the shape is
        identical to the corresponding input shape, except for the first
        dimension, which will be divided across the different Horovod processes.
    """
    if not tensors:
        return tensors
    if rocm_built() and op == Average:
        # Need to average in framework code
        true_op = Sum
    else:
        true_op = op
    with tf.device(device_dense):
        tensors_compressed, ctxs = zip(*[compression.compress(tensor) for tensor in tensors])
        reduced_tensors_compressed = _grouped_reducescatter(tensors_compressed, op=true_op, process_set=process_set,
                                                            prescale_factor=prescale_factor,
                                                            postscale_factor=postscale_factor)
        reduced_tensors = [compression.decompress(t, ctx) for t, ctx in zip(reduced_tensors_compressed, ctxs)]
        if op == Average and true_op == Sum:
            dtype = tensors[0].dtype  # HorovodGroupedReducescatterOp requires all input tensors to have the same dtype
            horovod_size = tf.cast(size_op(process_set_id=process_set.process_set_id)
                                   if int(os.environ.get("HOROVOD_ELASTIC", 0)) else process_set.size(),
                                   dtype=dtype)
            new_tensors = [(rt / horovod_size) for rt in reduced_tensors]
            return new_tensors
        else:
            return reduced_tensors


try:
    _global_variables = tf.compat.v1.global_variables
except AttributeError:
    try:
        _global_variables = tf.global_variables
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
                "hvd.broadcast_global_variables() does not support eager execution. "
                "Please use `hvd.broadcast_variables(<model/optimizer variables>)` instead."
            )

        return broadcast_variables(_global_variables(), root_rank)

try:
    _get_default_graph = tf.compat.v1.get_default_graph
except AttributeError:
    try:
        _get_default_graph = tf.get_default_graph
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
                if Horovod was built with HOROVOD_GPU_OPERATIONS.
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
def _make_cached_allreduce_grads_fn(name, device_dense, device_sparse,
                                    compression, sparse_as_dense, op,
                                    gradient_predivide_factor, groups,
                                    process_set):
    groups = refs_to_vars(groups) if isinstance(groups, tuple) else groups
    if op == Average:
        # Split average operation across pre/postscale factors
        # C++ backend will apply additional 1 / size() factor to postscale_factor for op == Average.
        prescale_factor = 1.0 / gradient_predivide_factor
        postscale_factor = gradient_predivide_factor
    else:
        prescale_factor = 1.0
        postscale_factor = 1.0

    def allreduce_grads(grads, vars=None, use_generic_names=False):
        with tf.name_scope(name + "_Allreduce"):
            if sparse_as_dense:
                grads = [tf.convert_to_tensor(grad)
                         if grad is not None and isinstance(grad, tf.IndexedSlices)
                         else grad for grad in grads]

            if groups is not None:
                if isinstance(groups, list):
                    var_name2grad = {}
                    for i in range(len(vars)):
                        var = vars[i]
                        grad = grads[i]
                        if grad is not None:
                            var_name2grad[var.name] = (i, grad)
                    grads_split = []
                    for group in groups:
                        grad_group = []
                        for var in group:
                            if var.name in var_name2grad:
                                grad_group.append(var_name2grad[var.name])
                                del var_name2grad[var.name]
                        grads_split.append(grad_group)
                    for _, grad in var_name2grad.items():
                        grads_split.append([grad])
                elif groups > 0:
                    grads_clean = [(i, grad) for i, grad in enumerate(grads) if grad is not None]
                    grads_split = split_list(grads_clean, groups)

                reduce_ops = [None] * len(vars)
                for i, group in enumerate(grads_split):
                    index_group, grad_group = [list(t) for t in zip(*group)]
                    reduce_ops_group = _grouped_allreduce_cond(grad_group,
                                                               device_dense=device_dense,
                                                               device_sparse=device_sparse,
                                                               compression=compression,
                                                               op=op,
                                                               prescale_factor=prescale_factor,
                                                               postscale_factor=postscale_factor,
                                                               process_set=process_set,
                                                               name=f"grad_{i}" if use_generic_names else None,
                                                               ignore_name_scope=use_generic_names)
                    for i in range(len(index_group)):
                        reduce_ops[index_group[i]] = reduce_ops_group[i]
                return reduce_ops

            return [_allreduce_cond(grad,
                                    device_dense=device_dense,
                                    device_sparse=device_sparse,
                                    compression=compression,
                                    op=op,
                                    prescale_factor=prescale_factor,
                                    postscale_factor=postscale_factor,
                                    process_set=process_set,
                                    name=f"grad_{i}" if use_generic_names else None,
                                    ignore_name_scope=use_generic_names)
                    if grad is not None else grad
                    for i, grad in enumerate(grads)]

    if _executing_eagerly():
        return _make_subgraph(allreduce_grads)
    else:
        return allreduce_grads


def _make_allreduce_grads_fn(name, device_dense, device_sparse,
                             compression, sparse_as_dense, op,
                             gradient_predivide_factor, groups,
                             process_set):
    groups = vars_to_refs(groups) if isinstance(groups, list) else groups
    return _make_cached_allreduce_grads_fn(name, device_dense, device_sparse,
                                           compression, sparse_as_dense, op,
                                           gradient_predivide_factor, groups,
                                           process_set)


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
        combine gradient values before applying gradients to model weights."""

        def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                    device_sparse='', compression=Compression.none,
                    sparse_as_dense=False, op=Average, gradient_predivide_factor=1.0,
                    backward_passes_per_step=1, average_aggregated_gradients=False,
                    groups=None, process_set=global_process_set, scale_local_gradients=True):
            if name is None:
                name = "Distributed{}".format(type(optimizer).__name__)
            super(_DistributedOptimizer, self).__init__(name=name, use_locking=use_locking)

            self._optimizer = optimizer
            self._allreduce_grads = _make_allreduce_grads_fn(
                name, device_dense, device_sparse, compression, sparse_as_dense, op,
                gradient_predivide_factor, groups, process_set=process_set)

            self._local_vars = set()
            self.process_set = process_set
            self.scale_local_gradients = scale_local_gradients
            self._agg_helper = None
            if backward_passes_per_step > 1:
                if _executing_eagerly():
                    raise ValueError(
                        "backward_passes_per_step > 1 is not yet supported "
                        "for _LegacyOptimizer with eager execution."
                    )

                self._agg_helper = LocalGradientAggregationHelper(
                    backward_passes_per_step=backward_passes_per_step,
                    allreduce_func=self._allreduce_grads,
                    sparse_as_dense=sparse_as_dense,
                    average_aggregated_gradients=average_aggregated_gradients,
                    rank=rank(),
                    optimizer_type=LocalGradientAggregationHelper._OPTIMIZER_TYPE_LEGACY,
                    process_set=process_set,
                    scale_local_gradients=scale_local_gradients,
                    name=name,
                )

        def register_local_var(self, var):
            """Registers a source/variable as worker local. Horovod will not perform any global
            operations on gradients corresponding to these sources and will instead return the local
            gradient."""
            if self._agg_helper:
                self._agg_helper.register_local_var(var)
            elif _IS_TF2:
                self._local_vars.add(var.ref())
            else:
                self._local_vars.add(var)

        def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.

            See Optimizer.compute_gradients() for more info.

            In DistributedOptimizer, compute_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = self._optimizer.compute_gradients(*args, **kwargs)
            grads, vars = zip(*gradients)
            if self._agg_helper:
                avg_grads = self._agg_helper.compute_gradients(grads, vars)
            else:
                def _filtered_reduce_grads(grads, vars):
                    rv = []
                    rg = []
                    if _IS_TF2:
                        v2g = {var.ref(): grad for var, grad in zip(vars, grads)}
                        for var, grad in zip(vars, grads):
                            if var.ref() not in self._local_vars:
                                rv.append(var)
                                rg.append(grad)
                    else:
                        v2g = {var: grad for var, grad in zip(vars, grads)}
                        for var, grad in zip(vars, grads):
                            if var not in self._local_vars:
                                rv.append(var)
                                rg.append(grad)

                    rg = self._allreduce_grads(rg, rv)
                    horovod_size = size_op(process_set_id=self.process_set.process_set_id) if int(os.environ.get("HOROVOD_ELASTIC", 0)) else self.process_set.size()
                    if _IS_TF2:
                        for rv,rg in zip(rv, rg):
                            v2g[rv.ref()] = rg

                        if self.scale_local_gradients and len(self._local_vars):
                            # Scale local gradients by a size factor. See pull/3695 and discussions/3705 for context.
                            for v_ref in v2g:
                                if v_ref in self._local_vars and v2g[v_ref]:
                                    grad = v2g[v_ref]
                                    if isinstance(grad, tf.IndexedSlices):
                                        grad = tf.IndexedSlices(grad.values / horovod_size, grad.indices, grad.dense_shape)
                                    else:
                                        grad /= horovod_size
                                    v2g[v_ref] = grad

                        return [v2g[rv.ref()] for rv in vars]
                    else:
                        for rv, rg in zip(rv, rg):
                            v2g[rv] = rg

                        if self.scale_local_gradients and len(self._local_vars):
                            # Scale local gradients by a size factor. See pull/3695 and discussions/3705 for context.
                            for v in v2g:
                                if v in self._local_vars and v2g[v]:
                                    grad = v2g[v]
                                    if isinstance(grad, tf.IndexedSlices):
                                        grad = tf.IndexedSlices(grad.values / horovod_size, grad.indices, grad.dense_shape)
                                    else:
                                        grad /= horovod_size
                                    v2g[v] = grad

                        return [v2g[rv] for rv in vars]

                avg_grads = _filtered_reduce_grads(grads, vars)
            return list(zip(avg_grads, vars))

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
            """Calls this same method on the underlying optimizer."""
            if self._agg_helper:
                return self._agg_helper.apply_gradients(
                    lambda: self._optimizer.apply_gradients(
                        grads_and_vars, global_step=global_step, name=name),
                    self._optimizer,
                    grads_and_vars,
                    global_step=global_step,
                    name=name,
                )
            return self._optimizer.apply_gradients(
                grads_and_vars, global_step=global_step, name=name)

        def get_slot(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot(*args, **kwargs)

        def get_slot_names(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.get_slot_names(*args, **kwargs)

        def variables(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.variables(*args, **kwargs)

    class _DistributedAdasumOptimizer(_LegacyOptimizer):
        """An optimizer that wraps another tf.Optimizer, using an allreduce to
        combine model deltas after applying gradients to model weights."""

        def __init__(self, optimizer, name=None, use_locking=False, device_dense='',
                    device_sparse='', compression=Compression.none, backward_passes_per_step=1):
            if name is None:
                name = "DistributedDelta{}".format(type(optimizer).__name__)
            super(_DistributedAdasumOptimizer, self).__init__(name=name, use_locking=use_locking)

            self._optimizer = optimizer
            self._name = name
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._backward_passes_per_step = backward_passes_per_step

        def _prepare(self):
            self._step_count = tf.get_variable(
                name="step_count", shape=[], dtype=tf.int64, trainable=False,
                initializer=tf.zeros_initializer)
            self._is_first_step = tf.cast(tf.math.equal(self._step_count, 0), dtype=tf.bool)
            self._is_comm_step  = tf.cast(tf.math.equal(self._step_count % self._backward_passes_per_step, self._backward_passes_per_step - 1), dtype=tf.bool)

        def _apply_shared(self, var, get_update_op):
            start_slot = self._get_or_make_slot(var, "delta_start")

            # initialize start on the first step
            assign_op = tf.cond(self._is_first_step,
                lambda: start_slot.assign(var, use_locking=self.use_locking).op,
                tf.no_op)

            with tf.control_dependencies([assign_op]):
                update_op = get_update_op()
                with tf.control_dependencies([update_op]):
                    def update():
                        # delta = var - start
                        local_delta = var.assign_sub(start_slot, use_locking=self.use_locking) # reuse var's memory
                        # delta = allreduce (delta)
                        global_delta = allreduce(local_delta,
                                                 device_dense=self._device_dense,
                                                 device_sparse=self._device_sparse,
                                                 compression=self._compression,
                                                 op=Adasum)
                        # start = start + delta
                        new_start = start_slot.assign_add(global_delta, use_locking=self.use_locking)
                        # var = start
                        return var.assign(new_start, use_locking=self.use_locking).op

                    # if its a communication step, then apply logic above
                    # if its not a communication step then just have the underlying
                    # optimizer update the model parameters according to its logic
                    return tf.cond(self._is_comm_step, update, tf.no_op)

        def _apply_dense(self, grad, var):
            return self._apply_shared(var, lambda: self._optimizer._apply_dense(grad, var))

        def _resource_apply_dense(self, grad, handle):
            return self._apply_shared(handle, lambda: self._optimizer._resource_apply_dense(grad, handle))

        def _apply_sparse(self, grad, var):
            return self._apply_shared(var, lambda: self._optimizer._apply_sparse(grad, var))

        def _resource_apply_sparse(self, grad, handle, indices):
            return self._apply_shared(handle, lambda: self._optimizer._resource_apply_sparse(grad, handle, indices))

        def _finish(self, update_ops, name_scope):
            with tf.control_dependencies(update_ops):
                return tf.assign_add(self._step_count, 1)

        def compute_gradients(self, *args, **kwargs):
            """Compute gradients of all trainable variables.
            See Optimizer.compute_gradients() for more info.
            """
            return self._optimizer.compute_gradients(*args, **kwargs)

        def apply_gradients(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.apply_gradients(*args, **kwargs)

        def get_slot(self, var, name):
            """Calls this same method on the underlying optimizer."""
            tmp = super(_DistributedAdasumOptimizer, self).get_slot(var, name)
            if tmp is not None:
                return tmp
            return self._optimizer.get_slot(var, name)

        def get_slot_names(self):
            """Appends local slot names to those of the underlying optimizer."""
            return super(_DistributedAdasumOptimizer, self).get_slot_names() +\
                self._optimizer.get_slot_names()

        def variables(self, *args, **kwargs):
            """Calls this same method on the underlying optimizer."""
            return self._optimizer.variables(*args, **kwargs)


def DistributedOptimizer(optimizer, name=None, use_locking=False, device_dense='',
                         device_sparse='', compression=Compression.none,
                         sparse_as_dense=False, backward_passes_per_step=1,
                         op=Average, gradient_predivide_factor=1.0,
                         average_aggregated_gradients=False,
                         num_groups=0, groups=None,
                         process_set=global_process_set, scale_local_gradients=True):
    """Construct a new DistributedOptimizer, which uses another optimizer
    under the hood for computing single-process gradient values and
    applying gradient updates after the gradient values have been combined
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
        if Horovod was built with HOROVOD_GPU_OPERATIONS.
      device_sparse:
        Device to be used for sparse tensors. Uses GPU by default
        if Horovod was built with HOROVOD_GPU_OPERATIONS.
      compression:
        Compression algorithm used during allreduce to reduce the amount
        of data sent during each parameter update step.  Defaults to
        not using compression.
      sparse_as_dense:
        Treat all sparse gradients as dense tensors.  This can help improve
        performance and memory utilization if the original sparse gradient
        has high density.  Defaults to false.
      backward_passes_per_step:
        Number of backward passes to perform before calling hvd.allreduce.
        This allows accumulating updates over multiple mini-batches before
        reducing and applying them.
      op:
        The reduction operation to use when combining gradients across
        different ranks.
      gradient_predivide_factor:
        If op == Average, gradient_predivide_factor splits the averaging
        before and after the sum. Gradients are scaled by
        1.0 / gradient_predivide_factor before the sum and
        gradient_predivide_factor / size after the sum.
      average_aggregated_gradients:
        Whether to average the aggregated gradients that have been accumulated
        over multiple mini-batches. If true divides gradients updates by
        backward_passes_per_step. Only applicable for backward_passes_per_step > 1.
      num_groups:
        Number of groups to assign gradient allreduce ops to for explicit
        grouping. Defaults to no explicit groups.
      groups:
        The parameter to group the gradient allreduce ops. Accept values is a
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
    if gradient_predivide_factor != 1.0:
        if rocm_built():
            raise ValueError('gradient_predivide_factor not supported yet with ROCm')
        if op != Average:
            raise ValueError('gradient_predivide_factor not supported with op != Average')

    if op == Adasum and average_aggregated_gradients:
        raise ValueError('Adasum does not support average_aggregated_gradients == True')

    if num_groups != 0:
        warnings.warn('Parameter `num_groups` has been replaced by `groups` '
                      'and will be removed in v0.23.0.', DeprecationWarning)
        if groups is None:
            groups = num_groups

    if groups is not None:
        if not (isinstance(groups, list) or groups > 0):
            raise ValueError('groups should be a non-negative integer or '
                            'a list of list of tf.Variable.')

    if isinstance(optimizer, _LegacyOptimizer):
        if op == Adasum:
            if process_set.process_set_id != 0:
                raise NotImplementedError("Adasum does not support process sets yet")
            return _DistributedAdasumOptimizer(optimizer, name, use_locking, device_dense,
                                            device_sparse, compression, backward_passes_per_step)

        return _DistributedOptimizer(
            optimizer=optimizer,
            name=name,
            use_locking=use_locking,
            device_dense=device_dense,
            device_sparse=device_sparse,
            compression=compression,
            sparse_as_dense=sparse_as_dense,
            op=op,
            gradient_predivide_factor=gradient_predivide_factor,
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=average_aggregated_gradients,
            groups=groups,
            process_set=process_set,
            scale_local_gradients=scale_local_gradients
        )
    elif (isinstance(optimizer, tf.keras.optimizers.Optimizer) or
          (not support_non_legacy_keras_optimizers(tf.keras) and
           isinstance(optimizer, tf.keras.optimizers.legacy.Optimizer))):
        if op == Adasum:
            raise ValueError('op == Adasum is not supported yet with Keras')

        import horovod.tensorflow.keras as hvd_k
        return hvd_k.DistributedOptimizer(
            optimizer=optimizer,
            name=name,
            device_dense=device_dense,
            device_sparse=device_sparse,
            compression=compression,
            sparse_as_dense=sparse_as_dense,
            gradient_predivide_factor=gradient_predivide_factor,
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=average_aggregated_gradients,
            process_set=process_set,
            scale_local_gradients=scale_local_gradients
        )
    else:
        raise ValueError('Provided optimizer doesn\'t inherit from either legacy '
                         'TensorFlow or Keras optimizer: %s' % optimizer)


if hasattr(tf, 'GradientTape'):
    class _DistributedGradientTape(tf.GradientTape):
        def __init__(self, tape, device_dense, device_sparse, compression, sparse_as_dense, op,
                     gradient_predivide_factor, groups, persistent=False,
                     watch_accessed_variables=True, process_set=global_process_set, scale_local_gradients=True):
            if hasattr(tape, '_watch_accessed_variables'):
                super(self.__class__, self).__init__(persistent, watch_accessed_variables)
            else:
                super(self.__class__, self).__init__(persistent)

            self._tape = tape
            self._allreduce_grads = _make_allreduce_grads_fn(
                'DistributedGradientTape', device_dense, device_sparse, compression,
                sparse_as_dense, op, gradient_predivide_factor, groups, process_set)

            self.process_set = process_set
            self.scale_local_gradients = scale_local_gradients
            self._local_sources = set()


        def register_local_source(self, source):
            """Registers a source/variable as worker local. Horovod will not perform any global
            operations on gradients corresponding to these sources and will instead return the local
            gradient."""
            if _IS_TF2:
                self._local_sources.add(source.ref())
            else:
                self._local_sources.add(source)

        def gradient(self, target, sources, output_gradients=None, use_generic_names=False):
            gradients = super(self.__class__, self).gradient(target, sources, output_gradients)

            # Collect source/grad pairs requiring reduction (i.e. not from a registered local source)
            rs = []
            rg = []
            if _IS_TF2:
                s2g = {s.ref() : g for s,g in zip(sources, gradients)}
                for s,g in zip(sources, gradients):
                    if s.ref() not in self._local_sources:
                        rs.append(s)
                        rg.append(g)
            else:
                s2g = {s : g for s,g in zip(sources, gradients)}
                for s,g in zip(sources, gradients):
                    if s not in self._local_sources:
                        rs.append(s)
                        rg.append(g)

            # Reduce grads
            rg = self._allreduce_grads(rg, rs, use_generic_names)
            horovod_size = size_op(process_set_id=self.process_set.process_set_id) if int(os.environ.get("HOROVOD_ELASTIC", 0)) else self.process_set.size()
            # Replace dict entries with reduced grads
            if _IS_TF2:
                for rs, rg in zip(rs, rg):
                    s2g[rs.ref()] = rg

                if self.scale_local_gradients and len(self._local_sources):
                    # Scale local gradients by a size factor. See pull/3695 and discussions/3705 for context.
                    for s_ref in s2g:
                        if s_ref in self._local_sources and s2g[s_ref] is not None:
                            grad = s2g[s_ref]
                            if isinstance(grad, tf.IndexedSlices):
                                grad = tf.IndexedSlices(grad.values / horovod_size, grad.indices, grad.dense_shape)
                            else:
                                grad /= horovod_size
                            s2g[s_ref] = grad

                return [s2g[s.ref()] for s in sources]
            else:
                for rs, rg in zip(rs, rg):
                    s2g[rs] = rg

                if self.scale_local_gradients and len(self._local_sources):
                    # Scale local gradients by a size factor. See pull/3695 and discussions/3705 for context.
                    for s in s2g:
                        if s in self._local_sources and s2g[s] is not None:
                            grad = s2g[s]
                            if isinstance(grad, tf.IndexedSlices):
                                grad = tf.IndexedSlices(grad.values / horovod_size, grad.indices, grad.dense_shape)
                            else:
                                grad /= horovod_size
                            s2g[s] = grad

                return [s2g[s] for s in sources]

    def DistributedGradientTape(gradtape, device_dense='', device_sparse='',
                                compression=Compression.none, sparse_as_dense=False,
                                op=Average, gradient_predivide_factor=1.0,
                                num_groups=0, groups=None, process_set=global_process_set, scale_local_gradients=True):
        """A tape that wraps another tf.GradientTape, using an allreduce to
        combine gradient values before applying gradients to model weights.

        Args:
          gradtape:
            GradientTape to use for computing gradients and applying updates.
          device_dense:
            Device to be used for dense tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_OPERATIONS.
          device_sparse:
            Device to be used for sparse tensors. Uses GPU by default
            if Horovod was built with HOROVOD_GPU_OPERATIONS.
          compression:
            Compression algorithm used during allreduce to reduce the amount
            of data sent during each parameter update step.  Defaults to
            not using compression.
          sparse_as_dense:
            Treat all sparse gradients as dense tensors.  This can help improve
            performance and memory utilization if the original sparse gradient
            has high density.  Defaults to false.
          op:
            The reduction operation to use when combining gradients across
            different ranks.
          gradient_predivide_factor:
            If op == Average, gradient_predivide_factor splits the averaging
            before and after the sum. Gradients are scaled by
            1.0 / gradient_predivide_factor before the sum and
            gradient_predivide_factor / size after the sum.
          num_groups:
            Number of groups to assign gradient allreduce ops to for explicit
            grouping. Defaults to no explicit groups.
          groups:
            The parameter to group the gradient allreduce ops. Accept values is a
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
        if gradient_predivide_factor != 1.0:
            if rocm_built():
                raise ValueError('gradient_predivide_factor not supported yet with ROCm')
            if op != Average:
                raise ValueError('gradient_predivide_factor not supported with op != Average')

        if num_groups != 0:
            warnings.warn('Parameter `num_groups` has been replaced by `groups` '
                          'and will be removed in v0.23.0.', DeprecationWarning)
            if groups is None:
                groups = num_groups

        if groups is not None:
            if not (isinstance(groups, list) or groups > 0):
                raise ValueError('groups should be a non-negative integer or '
                                'a list of list of tf.Variable.')

        cls = type(gradtape.__class__.__name__, (gradtape.__class__,),
                   dict(_DistributedGradientTape.__dict__))

        if hasattr(gradtape, '_watch_accessed_variables'):
            return cls(gradtape._tape, device_dense, device_sparse, compression,
                       sparse_as_dense, op, gradient_predivide_factor, groups,
                       gradtape._persistent, gradtape._watch_accessed_variables,
                       process_set=process_set, scale_local_gradients=scale_local_gradients)
        else:
            return cls(gradtape._tape, device_dense, device_sparse, compression,
                       sparse_as_dense, op, gradient_predivide_factor, groups,
                       gradtape._persistent, process_set=process_set, scale_local_gradients=scale_local_gradients)


    def PartialDistributedGradientTape(gradtape, device_dense='', device_sparse='',
                                       compression=Compression.none, sparse_as_dense=False,
                                       op=Average, gradient_predivide_factor=1.0,
                                       num_groups=0, groups=None, process_set=global_process_set,
                                       local_layers=None, scale_local_gradients=True):
        """A tape that wraps another tf.GradientTape, using an allreduce to
        combine gradient values before applying gradients to model weights similar to
        DistributedGradientTape execpt it skips allreducing gradients of the local layers
        passed in local_layers parameter.

        Args:
          gradtape:
            GradientTape to use for computing gradients and applying updates.
          local_layers:
            A collection of type tf.keras.layers.Layer local layers that their gradients need not
            to be synced accross ranks and is kept and applied locally.
            If not provided, the functionality of PartialDistributedGradientTape is
            identical to DistributedGradientTape.

        The rest of the arguments are similar to those of DistributedGradientTape.
        """
        if local_layers is None:
            local_layers = []
        elif isinstance(local_layers, tf.keras.layers.Layer):
            local_layers = [local_layers]
        elif not all(isinstance(layer, tf.keras.layers.Layer) for layer in local_layers):
            raise ValueError("All local layers must be of tf.keras.layers.Layer type.")

        local_vars = [var for layer in local_layers for var in layer.trainable_weights]

        _tape = DistributedGradientTape(gradtape, device_dense, device_sparse,
                                        compression, sparse_as_dense,
                                        op, gradient_predivide_factor,
                                        num_groups, groups, process_set, scale_local_gradients)
        for var in local_vars:
            _tape.register_local_source(var)
        return _tape
