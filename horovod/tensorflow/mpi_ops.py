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
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

from horovod.common import get_ext_suffix


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


ops.NotDifferentiable('HorovodAllreduce')


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


ops.NotDifferentiable('HorovodAllgather')


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


ops.NotDifferentiable('HorovodBroadcast')
