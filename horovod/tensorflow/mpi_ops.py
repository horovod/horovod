# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
# Modifications copyright Microsoft
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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
# =============================================================================
"""Inter-process communication using MPI."""

import re
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from horovod.common.util import get_ext_suffix, get_average_backwards_compatibility_fun, gpu_available, \
    num_rank_is_power_2
from horovod.common.basics import HorovodBasics as _HorovodBasics
from horovod.tensorflow.util import _executing_eagerly


def _load_library(name):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.

    Raises:
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    return library


MPI_LIB = _load_library('mpi_lib' + get_ext_suffix())

_basics = _HorovodBasics(__file__, 'mpi_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
is_initialized = _basics.is_initialized
start_timeline = _basics.start_timeline
stop_timeline = _basics.stop_timeline
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
mpi_threads_supported = _basics.mpi_threads_supported
mpi_enabled = _basics.mpi_enabled
mpi_built = _basics.mpi_built
gloo_enabled = _basics.gloo_enabled
gloo_built = _basics.gloo_built
nccl_built = _basics.nccl_built
ddl_built = _basics.ddl_built
ccl_built = _basics.ccl_built
cuda_built = _basics.cuda_built
rocm_built = _basics.rocm_built

# import reduction op values
Average = _basics.Average
Sum = _basics.Sum
Adasum = _basics.Adasum

is_homogeneous = _basics.is_homogeneous

handle_average_backwards_compatibility = get_average_backwards_compatibility_fun(_basics)

check_num_rank_power_of_2 = num_rank_is_power_2


# This function will create a default device map which includes all visible devices.
# Please run this function in a subprocess
def _check_has_gpu():
    import tensorflow as tf
    return tf.test.is_gpu_available()


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)


def _allreduce(tensor, name=None, op=Sum, prescale_factor=1.0, postscale_factor=1.0,
               ignore_name_scope=False):
    """An op which reduces an input tensor over all the Horovod processes. The
    default reduction is a sum.

    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None and not _executing_eagerly():
        name = 'HorovodAllreduce_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_allreduce(tensor, name=name, reduce_op=op,
                                     prescale_factor=prescale_factor,
                                     postscale_factor=postscale_factor,
                                     ignore_name_scope=ignore_name_scope)


@ops.RegisterGradient('HorovodAllreduce')
def _allreduce_grad(op, grad):
    """Gradient for allreduce op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    reduce_op = op.get_attr('reduce_op')
    prescale_factor = op.get_attr('prescale_factor')
    postscale_factor = op.get_attr('postscale_factor')
    ignore_name_scope = op.get_attr('ignore_name_scope')
    return _allreduce(grad, op=reduce_op, prescale_factor=prescale_factor,
                      postscale_factor=postscale_factor,
                      ignore_name_scope=ignore_name_scope)


def _grouped_allreduce(tensors, name=None, op=Sum, prescale_factor=1.0, postscale_factor=1.0,
                       ignore_name_scope=False):
    """An op which reduces input tensors over all the Horovod processes. The
    default reduction is a sum.

    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.

    The reduction operations are keyed by the name of the op. Reductions are
    performed across tensors in the same list position. The tensor type and
    shape must be the same on all Horovod processes for tensors sharing
    positions in the input tensor list. The reduction will not start until all
    processes are ready to send and receive the tensors.

    Returns:
      A list of tensors of the same shape and type as those in `tensors`,
      summed across all processes.
    """
    if name is None and not _executing_eagerly():
        name = _normalize_name('HorovodGroupedAllreduce_%s_%s' % (tensors[0].name, tensors[-1].name))
    return MPI_LIB.horovod_grouped_allreduce(tensors, name=name, reduce_op=op,
                                             prescale_factor=prescale_factor,
                                             postscale_factor=postscale_factor,
                                             ignore_name_scope=ignore_name_scope)


@ops.RegisterGradient('HorovodGroupedAllreduce')
def _grouped_allreduce_grad(op, *grads):
    """Gradient for the grouped allreduce op.

    Args:
      op: An operation.
      grads: List of `Tensor` gradients with respect to the outputs of the op.

    Returns:
      The gradients with respect to the inputs of the op.
    """
    reduce_op = op.get_attr('reduce_op')
    prescale_factor = op.get_attr('prescale_factor')
    postscale_factor = op.get_attr('postscale_factor')
    ignore_name_scope = op.get_attr('ignore_name_scope')
    # TODO(joshr): should this be done as separate allreduce ops?
    return _grouped_allreduce(list(grads), op=reduce_op, prescale_factor=prescale_factor,
                      postscale_factor=postscale_factor,
                      ignore_name_scope=ignore_name_scope)


def allgather(tensor, name=None, ignore_name_scope=False):
    """An op which concatenates the input tensor with the same input tensor on
    all other Horovod processes.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Returns:
      A tensor of the same type as `tensor`, concatenated on dimension zero
      across all processes. The shape is identical to the input shape, except for
      the first dimension, which may be greater and is the sum of all first
      dimensions of the tensors in different Horovod processes.
    """
    if name is None and not _executing_eagerly():
        name = 'HorovodAllgather_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_allgather(tensor, name=name,
                                     ignore_name_scope=ignore_name_scope)


@ops.RegisterGradient('HorovodAllgather')
def _allgather_grad(op, grad):
    """Gradient for allgather op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    ignore_name_scope = op.get_attr('ignore_name_scope')
    grad = _allreduce(grad, ignore_name_scope=ignore_name_scope)

    with tf.device('/cpu:0'):
        # Keep the tensor of split sizes on CPU.
        x = op.inputs[0]
        d = tf.shape(x)
        d = tf.reshape(d[0], [1])

        s = size()
        d = tf.reshape(allgather(d, ignore_name_scope=ignore_name_scope), [s])

    splits = tf.split(grad, num_or_size_splits=d, axis=0)
    return splits[rank()]


def broadcast(tensor, root_rank, name=None, ignore_name_scope=False):
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other Horovod processes.

    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    if name is None and not _executing_eagerly():
        name = 'HorovodBroadcast_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_broadcast(tensor, name=name, root_rank=root_rank,
                                     ignore_name_scope=ignore_name_scope)


@ops.RegisterGradient('HorovodBroadcast')
def _broadcast_grad(op, grad):
    """Gradient for broadcast op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    root_rank = op.get_attr('root_rank')
    ignore_name_scope = op.get_attr('ignore_name_scope')
    grad_reduced = _allreduce(grad, ignore_name_scope=ignore_name_scope)
    if rank() != root_rank:
        return grad_reduced * 0
    return grad_reduced


def alltoall(tensor, splits=None, name=None, ignore_name_scope=False):
    """An op that scatters slices of the input tensor to all other Horovod processes
    and returns a tensor of gathered slices from all other Horovod processes.

    The slicing is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    Arguments:
        tensor: A tensor to distribute with alltoall.
        splits: A tensor of integers in rank order describing how many
                elements in `tensor` to send to each worker.  Splitting is
                applied along the first dimension of `tensor`. If `splits` is
                not provided, the first dimension is split equally by the
                number of Horovod processes.
        name: A name of the alltoall operation.
        ignore_name_scope: If True, ignores any outer name scope applied by
                           TensorFlow in the name used by the Horovod operation.

    Returns:
      A tensor of the same type as `tensor`, concatenated on dimension zero
      across all processes. The shape is identical to the input shape, except for
      the first dimension, which may be greater and is the sum of all first
      dimensions of the gathered tensor slices from different Horovod processes.
    """
    # If splits not provided, create empty tensor as placeholder
    splits_ = tf.convert_to_tensor(splits) if splits is not None else tf.constant([], dtype=tf.int32)

    if name is None and not _executing_eagerly():
        name = 'HorovodAlltoall_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_alltoall(tensor, splits=splits_, name=name,
                                    ignore_name_scope=ignore_name_scope)

@ops.RegisterGradient('HorovodAlltoall')
def _alltoall_grad(op, grad):
    """Gradient for alltoall op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    tensor = op.inputs[0]
    splits = op.inputs[1]
    ignore_name_scope = op.get_attr('ignore_name_scope')

    splits = tf.cond(tf.equal(tf.size(splits), 0),
                     lambda : tf.ones([size()], dtype=tf.int32) * (tf.shape(tensor)[0] // size()),
                     lambda : splits)
    recvsplits = alltoall(splits, splits=[1 for _ in range(size())],
                          ignore_name_scope=ignore_name_scope)
    return [alltoall(grad, splits=recvsplits, ignore_name_scope=ignore_name_scope), None]

def join():
    return MPI_LIB.horovod_join()


def size_op(name=None):
    """An op that returns the number of Horovod processes.

    This operation determines the return value at the graph execution time,
    rather than at the graph construction time, and so allows for a graph to be
    constructed in a different environment than where it will be executed.

    Returns:
      An integer scalar containing the number of Horovod processes.
    """
    return MPI_LIB.horovod_size(name=name)


ops.NotDifferentiable('HorovodSize')


def local_size_op(name=None):
    """An op that returns the number of Horovod processes within the
    node the current process is running on.

    This operation determines the return value at the graph execution time,
    rather than at the graph construction time, and so allows for a graph to be
    constructed in a different environment than where it will be executed.

    Returns:
      An integer scalar containing the number of local Horovod processes.
    """
    return MPI_LIB.horovod_local_size(name=name)


ops.NotDifferentiable('HorovodLocalSize')


def rank_op(name=None):
    """An op that returns the Horovod rank of the calling process.

    This operation determines the return value at the graph execution time,
    rather than at the graph construction time, and so allows for a graph to be
    constructed in a different environment than where it will be executed.

    Returns:
      An integer scalar with the Horovod rank of the calling process.
    """
    return MPI_LIB.horovod_rank(name=name)


ops.NotDifferentiable('HorovodRank')


def local_rank_op(name=None):
    """An op that returns the local Horovod rank of the calling process, within the
    node that it is running on. For example, if there are seven processes running
    on a node, their local ranks will be zero through six, inclusive.

    This operation determines the return value at the graph execution time,
    rather than at the graph construction time, and so allows for a graph to be
    constructed in a different environment than where it will be executed.

    Returns:
      An integer scalar with the local Horovod rank of the calling process.
    """
    return MPI_LIB.horovod_rank(name=name)


ops.NotDifferentiable('HorovodLocalRank')
