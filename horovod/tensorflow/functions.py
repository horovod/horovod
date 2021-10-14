# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from horovod.tensorflow.mpi_ops import allgather, broadcast, broadcast_
from horovod.tensorflow.mpi_ops import rank, size
from horovod.tensorflow.util import _cache, _executing_eagerly, _make_subgraph
from horovod.common.process_sets import ProcessSet, global_process_set


@_cache
def _make_broadcast_group_fn():
    if _executing_eagerly():
        # Eager mode will parallelize independent control flow
        def broadcast_group(variables, root_rank, process_set: ProcessSet):
            for var in variables:
                var.assign(broadcast(var, root_rank, process_set=process_set))

        return _make_subgraph(broadcast_group)
    else:
        # Graph mode requires an Op
        def broadcast_group(variables, root_rank, process_set: ProcessSet):
            return tf.group(*[var.assign(broadcast(var, root_rank, process_set=process_set))
                              for var in variables])

        return broadcast_group


@_cache
def _make_inplace_broadcast_group_fn():
    if _executing_eagerly():
        # These are just a few calls of broadcast_, no need to aggregate them in a tf.function
        def broadcast_group(variable_lists, root_rank, process_set: ProcessSet):
            for variables in variable_lists:
                broadcast_(variables, root_rank, process_set=process_set)

        return broadcast_group
    else:
        # Graph mode requires an Op
        def broadcast_group(variable_lists, root_rank, process_set: ProcessSet):
            return tf.group(*[broadcast_(variables, root_rank, process_set=process_set)
                              for variables in variable_lists])

        return broadcast_group


def broadcast_variables(variables, root_rank, process_set=global_process_set, inplace=False):
    """
    Broadcasts variables from root rank to all other processes
    in a process set (defaults to all Horovod processes).

    Optionally, the broadcast may be performed in-place, which avoids
    temporary memory allocations and fragmentation. This is only
    supported with TensorFlow 2.6 or later. Reference variables
    (legacy support in TF 2) must all be of the same data type. There
    is no such restriction for resource variables (default in TF 2).

    Arguments:
        variables: variables for broadcast
        root_rank: rank of the process from which global variables will be broadcasted
                   to all other processes.
        process_set: Process set object to limit this operation to a subset of
                     Horovod processes. Default is the global process set.
        inplace: whether to perform in-place broadcasts
    """
    if inplace:
        vars_by_device = {}
        for var in variables:
            vars_by_device.setdefault(var.device, []).append(var)

        inplace_broadcast_group = _make_inplace_broadcast_group_fn()
        return inplace_broadcast_group(vars_by_device.values(), root_rank, process_set)
    else:
        broadcast_group = _make_broadcast_group_fn()
        return broadcast_group(variables, root_rank, process_set)


def broadcast_object(obj, root_rank=0, session=None, name=None, process_set=global_process_set):
    """
    Serializes and broadcasts an object from root rank to all other processes
    in a process set (defaults to all Horovod processes).

    Arguments:
        obj: An object capable of being serialized without losing any context.
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
        session: Session for TensorFlow v1 compatibility.
        name: Optional name to use during broadcast, will default to the class
              type.
        process_set: Process set object to limit this operation to a subset of
                     Horovod processes. Default is the global process set.
    Returns:
        The object that was broadcast from the `root_rank`.
    """
    if name is None:
        name = type(obj).__name__

    def to_numpy(v):
        if not _executing_eagerly():
            sess = session or ops.get_default_session()
            return sess.run(v)
        else:
            return v.numpy()

    if rank() == root_rank:
        b = io.BytesIO()
        cloudpickle.dump(obj, b)
        t = tf.convert_to_tensor(bytearray(b.getvalue()), dtype=tf.uint8)
        sz = tf.convert_to_tensor([t.shape[0]], dtype=tf.int32)
        to_numpy(broadcast(sz, root_rank, name + '.sz', process_set=process_set))
    else:
        sz = tf.convert_to_tensor([0], dtype=tf.int32)
        sz = to_numpy(broadcast(sz, root_rank, name + '.sz', process_set=process_set))
        t = tf.zeros(sz.tolist()[0], dtype=tf.uint8)

    t = to_numpy(broadcast(t, root_rank, name + '.t', process_set=process_set))

    if rank() != root_rank:
        buf = io.BytesIO(t.tobytes())
        obj = cloudpickle.load(buf)

    return obj


def broadcast_object_fn(root_rank=0, session=None, name=None, process_set=global_process_set):
    name = name or 'broadcast_object_fn'

    sz = tf.placeholder(tf.int32, [1], name='bcast_object_size')
    bcast_size = broadcast(sz, root_rank, name + '.sz', process_set=process_set)

    t = tf.placeholder(tf.uint8, [None], name='bcast_object_data')
    bcast_data = broadcast(t, root_rank, name + '.t', process_set=process_set)

    session = session or ops.get_default_session()

    def _bcast(obj):
        if rank() == root_rank:
            b = io.BytesIO()
            cloudpickle.dump(obj, b)
            t_ = bytearray(b.getvalue())
            sz_ = [len(t_)]
            session.run(bcast_size, feed_dict={sz: sz_})
        else:
            sz_ = [0]
            sz_ = session.run(bcast_size, feed_dict={sz: sz_})
            t_ = np.zeros(sz_, dtype=np.uint8)

        t_ = session.run(bcast_data, feed_dict={t: t_})

        if rank() != root_rank:
            buf = io.BytesIO(t_.tobytes())
            obj = cloudpickle.load(buf)

        return obj
    return _bcast


def allgather_object(obj, session=None, name=None, process_set=global_process_set):
    """
    Serializes and allgathers an object from all other processes.

    Arguments:
        obj: An object capable of being serialized without losing any context.
        session: Session for TensorFlow v1 compatibility.
        name: Optional name to use during allgather, will default to the class
              type.
        process_set: Process set object to limit this operation to a subset of
                     Horovod processes. Default is the global process set.

    Returns:
        The list of objects that were allgathered across all ranks.
    """
    if name is None:
        name = type(obj).__name__

    def load(byte_array):
        buf = io.BytesIO(byte_array.tobytes())
        return cloudpickle.load(buf)

    def to_numpy(v):
        if not _executing_eagerly():
            sess = session or ops.get_default_session()
            return sess.run(v)
        else:
            return v.numpy()

    b = io.BytesIO()
    cloudpickle.dump(obj, b)

    t = tf.convert_to_tensor(bytearray(b.getvalue()), dtype=tf.uint8)
    sz = tf.convert_to_tensor([t.shape[0]], dtype=tf.int32)

    sizes = to_numpy(allgather(sz, name=name + '.sz', process_set=process_set))
    gathered = to_numpy(allgather(t, name=name + '.t', process_set=process_set))

    def select(i):
        start = sum(sizes[:i])
        end = start + sizes[i]
        return gathered[start:end]

    return [load(select(i)) for i in range(process_set.size())]
