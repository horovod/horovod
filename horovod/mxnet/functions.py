
# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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

import io

import cloudpickle
import mxnet as mx

from horovod.mxnet.mpi_ops import allgather, broadcast_
from horovod.mxnet.mpi_ops import rank, size


def broadcast_object(obj, root_rank=0, name=None):
    """
    Serializes and broadcasts an object from root rank to all other processes.

    Arguments:
        obj: An object capable of being serialized without losing any context.
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
        name: Optional name to use during broadcast, will default to the class
              type.
    Returns:
        The object that was broadcast from the `root_rank`.
    """
    if name is None:
        name = type(obj).__name__

    if rank() == root_rank:
        b = io.BytesIO()
        cloudpickle.dump(obj, b)
        t = mx.nd.array(bytearray(b.getvalue()), dtype='byte')
        sz = mx.nd.array([t.size], dtype='int')

        broadcast_(sz, root_rank, name + '.sz')
    else:
        sz = mx.nd.empty(shape=[1], dtype='int')
        broadcast_(sz, root_rank, name + '.sz')
        t = mx.nd.empty(shape=[sz.asscalar()], dtype='byte')

    broadcast_(t, root_rank, name + '.t')

    if rank() != root_rank:
        buf = io.BytesIO(t.asnumpy().tobytes())
        obj = cloudpickle.load(buf)

    return obj


def allgather_object(obj, name=None):
    """
    Serializes and allgathers an object from all other processes.

    Arguments:
        obj: An object capable of being serialized without losing any context.
        name: Optional name to use during allgather, will default to the class
              type.

    Returns:
        The list of objects that were allgathered across all ranks.
    """
    if name is None:
        name = type(obj).__name__

    def load(byte_array):
        buf = io.BytesIO(byte_array.tobytes())
        return cloudpickle.load(buf)

    b = io.BytesIO()
    cloudpickle.dump(obj, b)

    t = mx.nd.array(bytearray(b.getvalue()), dtype='byte')
    sz = mx.nd.array([t.size], dtype='int')

    sizes = allgather(sz, name=name + '.sz').asnumpy()
    gathered = allgather(t, name=name + '.t').asnumpy()

    def select(i):
        start = sum(sizes[:i])
        end = start + sizes[i]
        return gathered[start:end]

    return [load(select(i)) for i in range(size())]
