# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 Uber Technologies, Inc.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from horovod.common import get_ext_suffix
from horovod.common import HorovodBasics as _HorovodBasics


def _load_library(name, op_list=None):
    """Loads a .so file containing the specified operators.

    Args:
      name: The name of the .so file to load.
      op_list: A list of names of operators that the library should have. If None
          then the .so file's contents will not be verified.

    Raises:
      NameError if one of the required ops is missing.
      NotFoundError if were not able to load .so file.
    """
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    for expected_op in (op_list or []):
        for lib_op in library.OP_LIST.op:
            if lib_op.name == expected_op:
                break
        else:
            raise NameError(
                'Could not find operator %s in dynamic library %s' %
                (expected_op, name))
    return library


MPI_LIB = _load_library('mpi_lib' + get_ext_suffix(),
                        ['HorovodAllgather', 'HorovodAllreduce'])

_basics = _HorovodBasics(__file__, 'mpi_lib')

# import basic methods
init = _basics.init
shutdown = _basics.shutdown
size = _basics.size
local_size = _basics.local_size
rank = _basics.rank
local_rank = _basics.local_rank
mpi_threads_supported = _basics.mpi_threads_supported


# Schema: handle -> input, output
# We keep input in order to make sure it does not get garbage collected
# before the operation is finished.
_handle_map = {}


def _normalize_name(name):
    """Normalizes operation name to TensorFlow rules."""
    return re.sub('[^a-zA-Z0-9_]', '_', name)


def _allreduce(tensor, name=None):
    """An op which sums an input tensor over all the Horovod processes.

    The reduction operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The reduction
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, summed across all
      processes.
    """
    if name is None:
        name = 'HorovodAllreduce_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_allreduce(tensor, name=name)


@ops.RegisterGradient('HorovodAllreduce')
def _allreduce_grad(op, grad):
    """Gradient for allreduce op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    return _allreduce(grad)


def _allreduce_list(tensors):
    """An op which sums a list of input tensors over all the Horovod processes.

    This function is intended to be used in eager execution mode, when all ops
    are normally executed synchronously. By batching all tensors together into
    a single call, we can perform tensor fusion, resulting in fewer network calls
    and quicker results.

    Args:
        tensors: A list of tensors to sum independently across workers.

    Returns:
      A list of summed tensors.
    """
    return MPI_LIB.horovod_allreduce_list(tensors)


def allgather(tensor, name=None):
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
    if name is None:
        name = 'HorovodAllgather_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_allgather(tensor, name=name)


@ops.RegisterGradient('HorovodAllgather')
def _allgather_grad(op, grad):
    """Gradient for allgather op.

    Args:
      op: An operation.
      grad: `Tensor` gradient with respect to the output of the op.

    Returns:
      The gradient with respect to the input of the op.
    """
    grad = _allreduce(grad)

    x = op.inputs[0]
    d0 = x.get_shape().as_list()[0]
    d = tf.convert_to_tensor([d0], dtype=tf.int32)

    s = size()
    d = tf.reshape(allgather(d), [s])

    splits = tf.split(grad, num_or_size_splits=d, axis=0)
    return splits[rank()]


def allgather_list(tensors):
    """An op which concatenates the list of input tensors with the same input tensors on
    all other Horovod processes.

    The concatenation is done on the first dimension, so the input tensors on the
    different processes must have the same rank and shape, except for the first
    dimension, which is allowed to be different.

    This function is intended to be used in eager execution mode, when all ops
    are normally executed synchronously. By batching all tensors together into
    a single call, we can perform tensor fusion, resulting in fewer network calls
    and quicker results.

    Args:
        tensors: A list of tensors to gather independently across workers.

    Returns:
      A list of tensosr of the same types as the input `tensors`, concatenated on dimension
      zero across all processes. For every tensor in the list, the shape is identical to the
      shape of the input tensor at the same index, except for the first dimension, which may
      be greater and is the sum of all first dimensions of the tensors in different Horovod
      processes.
    """
    return MPI_LIB.horovod_allgather_list(tensors)


def broadcast(tensor, root_rank, name=None):
    """An op which broadcasts the input tensor on root rank to the same input tensor
    on all other Horovod processes.

    The broadcast operation is keyed by the name of the op. The tensor type and
    shape must be the same on all Horovod processes for a given name. The broadcast
    will not start until all processes are ready to send and receive the tensor.

    Returns:
      A tensor of the same shape and type as `tensor`, with the value broadcasted
      from root rank.
    """
    if name is None:
        name = 'HorovodBroadcast_%s' % _normalize_name(tensor.name)
    return MPI_LIB.horovod_broadcast(tensor, name=name, root_rank=root_rank)


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
    grad_reduced = _allreduce(grad)
    if rank() != root_rank:
        return grad_reduced * 0
    return grad_reduced


def broadcast_list(tensors, root_rank):
    """An op which broadcasts the input tensors on root rank to the same input tensors
    on all other Horovod processes.

    The broadcast operation is keyed by the name of the op. The tensor types, shapes,
    and order in the list must be the same on all Horovod processes. The broadcast
    will not start until all processes are ready to send and receive the tensor.

    This function is intended to be used in eager execution mode, when all ops
    are normally executed synchronously. By batching all tensors together into
    a single call, we can perform tensor fusion, resulting in fewer network calls
    and quicker results.

    Args:
        tensors: A list of tensors to broadcast from the root rank to all other workers.
        root_rank: Rank that will send data, other ranks will receive data.

    Returns:
      A list of tensors of the same shapes, types, and order as `tensors`, with the
      values broadcasted from root rank.
    """
    return MPI_LIB.horovod_broadcast_list(tensors, root_rank=root_rank)
