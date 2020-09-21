# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

# Load all the necessary MXNet C types.
import ctypes
import os

import mxnet as mx
from mxnet.base import c_str, check_call, string_types

from horovod.common.util import get_ext_suffix
from horovod.common.basics import HorovodBasics as _HorovodBasics
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

dll_path = os.path.join(os.path.dirname(__file__),
                        'mpi_lib' + get_ext_suffix())
MPI_MXNET_LIB_CTYPES = ctypes.CDLL(dll_path, ctypes.RTLD_GLOBAL)


def allreduce(tensor, average=True, name=None, priority=0, prescale_factor=1.0,
              postscale_factor=1.0):
    """
    A function that performs averaging or summation of the input tensor over
    all the Horovod processes. The input tensor is not modified.

    The reduction operation is keyed by the name. If name is not provided, an
    incremented auto-generated name is used. The tensor type and shape must be
    the same on all Horovod processes for a given name. The reduction will not
    start until all processes are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
        priority: The priority of this operation. Higher priority operations
                  are likely to be executed before other operations.
        prescale_factor: Multiplicative factor to scale tensor before allreduce
        postscale_factor: Multiplicative factor to scale tensor after allreduce

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed
        across all processes.
    """
    output = mx.nd.zeros(shape=tensor.shape, ctx=tensor.context,
                         dtype=tensor.dtype)
    c_in = tensor.handle
    c_out = output.handle
    if isinstance(name, string_types):
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_allreduce_async(
            c_in, c_out, c_str(name), ctypes.c_bool(average),
            ctypes.c_int(priority),
            ctypes.c_double(prescale_factor),
            ctypes.c_double(postscale_factor)))
    else:
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_allreduce_async(
            c_in, c_out, name, ctypes.c_bool(average),
            ctypes.c_int(priority),
            ctypes.c_double(prescale_factor),
            ctypes.c_double(postscale_factor)))

    return output


def allreduce_(tensor, average=True, name=None, priority=0, prescale_factor=1.0,
              postscale_factor=1.0):
    """
    A function that performs in-place averaging or summation of the input
    tensor over all the Horovod processes.

    The reduction operation is keyed by the name. If name is not provided, an
    incremented auto-generated name is used. The tensor type and shape must be
    the same on all Horovod processes for a given name. The reduction will not
    start until all processes are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to average and sum.
        average: A flag indicating whether to compute average or summation,
                 defaults to average.
        name: A name of the reduction operation.
        priority: The priority of this operation. Higher priority operations
                  are likely to be executed before other operations.
        prescale_factor: Multiplicative factor to scale tensor before allreduce
        postscale_factor: Multiplicative factor to scale tensor after allreduce

    Returns:
        A tensor of the same shape and type as `tensor`, averaged or summed
        across all processes.
    """
    c_in = tensor.handle
    c_out = tensor.handle
    if isinstance(name, string_types):
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_allreduce_async(
            c_in, c_out, c_str(name), ctypes.c_bool(average),
            ctypes.c_int(priority),
            ctypes.c_double(prescale_factor),
            ctypes.c_double(postscale_factor)))
    else:
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_allreduce_async(
            c_in, c_out, name, ctypes.c_bool(average),
            ctypes.c_int(priority),
            ctypes.c_double(prescale_factor),
            ctypes.c_double(postscale_factor)))
    return tensor


def allgather(tensor, name=None, priority=0):
    """
    A function that concatenates the input tensor with the same input tensor on
    all other Horovod processes. The input tensor is not modified.

    The concatenation is done on the first dimension, so the input tensors on
    the different processes must have the same rank and shape, except for the
    first dimension, which is allowed to be different.

    Arguments:
        tensor: A tensor to allgather.
        name: A name of the allgather operation.
        priority: The priority of this operation. Higher priority operations
                  are likely to be executed before other operations.

    Returns:
        A tensor of the same type as `tensor`, concatenated on dimension zero
        across all processes. The shape is identical to the input shape, except
        for the first dimension, which may be greater and is the sum of all
        first dimensions of the tensors in different Horovod processes.
    """
    assert(isinstance(tensor, mx.nd.NDArray))
    # Size of output is unknown, create output array that
    # will be resized during Horovod operation
    output = mx.nd.empty(shape=[1], ctx=tensor.context,
                         dtype=tensor.dtype)
    c_in = tensor.handle
    c_out = output.handle
    if isinstance(name, string_types):
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_allgather_async(
            c_in, c_out, c_str(name), ctypes.c_int(priority)))
    else:
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_allgather_async(
            c_in, c_out, name, ctypes.c_int(priority)))

    # Need to block here so changes to output tensor are visible
    output.wait_to_read()
    return output


def broadcast(tensor, root_rank, name=None, priority=0):
    """
    A function that broadcasts the input tensor on root rank to the same input
    tensor on all other Horovod processes. The input tensor is not modified.

    The broadcast operation is keyed by the name. If name is not provided, an
    incremented auto-generated name is used. The tensor type and shape must be
    the same on all Horovod processes for a given name. The broadcast will not
    start until all processes are ready to send and receive the tensor.

    This acts as a thin wrapper around an autograd function.  If your input
    tensor requires gradients, then callings this function will allow gradients
    to be computed and backpropagated.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.
        priority: The priority of this operation. Higher priority operations
                  are likely to be executed before other operations.

    Returns:
        A tensor of the same shape and type as `tensor`, with the value
        broadcasted from root rank.
    """
    if rank() == root_rank:
        output = tensor.copy()
    else:
        output = mx.nd.zeros(shape=tensor.shape, ctx=tensor.context,
                             dtype=tensor.dtype)
    c_in = tensor.handle
    c_out = output.handle
    if isinstance(name, string_types):
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_broadcast_async(
            c_in, c_out, c_str(name), ctypes.c_int(root_rank),
            ctypes.c_int(priority)))
    else:
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_broadcast_async(
            c_in, c_out, name, ctypes.c_int(root_rank),
            ctypes.c_int(priority)))
    return output


def broadcast_(tensor, root_rank, name=None, priority=0):
    """
    A function that broadcasts the input tensor on root rank to the same input
    tensor on all other Horovod processes. The operation is performed in-place.

    The broadcast operation is keyed by the name. If name is not provided, an
    incremented auto-generated name is used. The tensor type and shape must be
    the same on all Horovod processes for a given name. The broadcast will not
    start until all processes are ready to send and receive the tensor.

    Arguments:
        tensor: A tensor to broadcast.
        root_rank: The rank to broadcast the value from.
        name: A name of the broadcast operation.
        priority: The priority of this operation. Higher priority operations
                  are likely to be executed before other operations.

    Returns:
        A tensor of the same shape and type as `tensor`, with the value
        broadcasted from root rank.
    """
    c_in = tensor.handle
    c_out = tensor.handle
    if isinstance(name, string_types):
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_broadcast_async(
            c_in, c_out, c_str(name), ctypes.c_int(root_rank),
            ctypes.c_int(priority)))
    else:
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_broadcast_async(
            c_in, c_out, name, ctypes.c_int(root_rank),
            ctypes.c_int(priority)))
    return tensor

def alltoall(tensor, splits=None, name=None, priority=0):
    """
    A function that scatters slices of the input tensor to all other Horovod processes
    and returns a tensor of gathered slices from all other Horovod processes. The input
    tensor is not modified.

    The slicing is done on the first dimension, so the input tensors on
    the different processes must have the same rank and shape, except for the
    first dimension, which is allowed to be different.

    Arguments:
        tensor: A tensor to distribute with alltoall.
        splits: A tensor of integers in rank order describing how many
                elements in `tensor` to send to each worker.  Splitting is
                applied along the first dimension of `tensor`. If `splits` is
                not provided, the first dimension is split equally by the
                number of Horovod processes.
        name: A name of the alltoall operation.
        priority: The priority of this operation. Higher priority operations
                  are likely to be executed before other operations.

    Returns:
        A tensor containing the gathered tensor data from all workers.
    """
    assert(isinstance(tensor, mx.nd.NDArray))

    if splits is None:
        # If splits not provided, create empty tensor as placeholder
        splits = mx.nd.array([], ctx=mx.cpu(), dtype='int32')
    elif not isinstance(splits, mx.nd.NDArray):
        splits = mx.nd.array(splits, ctx=mx.cpu(), dtype='int32')

    # Size of output is unknown, create output array that
    # will be resized during Horovod operation
    output = mx.nd.empty(shape=[1], ctx=tensor.context,
                         dtype=tensor.dtype)
    c_in = tensor.handle
    c_out = output.handle
    c_splits = splits.handle
    if isinstance(name, string_types):
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_alltoall_async(
            c_in, c_out, c_str(name), c_splits, ctypes.c_int(priority)))
    else:
        check_call(MPI_MXNET_LIB_CTYPES.horovod_mxnet_alltoall_async(
            c_in, c_out, name, c_splits, ctypes.c_int(priority)))

    # Need to block here so changes to output tensor are visible
    output.wait_to_read()
    return output
