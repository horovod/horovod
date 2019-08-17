# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import horovod.tensorflow as hvd

import copy
import os
import tensorflow as tf
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import values
from tensorflow.python.eager import tape
from tensorflow.python.util import nest


class HorovodDistributionStrategy(distribute_lib.Strategy):
    """
    HorovodDistributionStrategy integrates TensorFlow DistributionStrategy
    with Horovod ecosystem. It is currently optimized for use with TF-Keras APIs.
    """
    def __init__(self):
        extended = HorovodDistributionStrategyExtended(self)
        super(HorovodDistributionStrategy, self).__init__(extended)


class HorovodDistributionStrategyExtended(distribute_lib.StrategyExtendedV1):
    def __init__(self, container_strategy):
        super(HorovodDistributionStrategyExtended, self).__init__(container_strategy)
        self._gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        if self._gpu_devices:
            tf.config.experimental.set_visible_devices(self._gpu_devices[hvd.local_rank()], 'GPU')
        self._host_device = device_util.canonicalize('/device:CPU:0')
        self._compute_device = device_util.canonicalize('/device:GPU:0') if self._gpu_devices else self._host_device

    def _create_variable(self, next_creator, *args, **kwargs):
        with tape.stop_recording():
            v = next_creator(*args, **kwargs)
            v.assign(hvd.broadcast(v, root_rank=0))
            return v

    def _local_results(self, val):
        return (val,)

    def value_container(self, value):
        return value

    def _call_for_each_replica(self, fn, args, kwargs):
        with tf.distribute.ReplicaContext(self._container_strategy(),
                                          replica_id_in_sync_group=hvd.rank()):
            return fn(*args, **kwargs)

    def _update(self, var, fn, args, kwargs, group):
        # The implementations of _update() and _update_non_slot() are identical
        # except _update() passes `var` as the first argument to `fn()`.
        return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        del colocate_with
        result = fn(*args, **kwargs)
        if group:
            return result
        else:
            return nest.map_structure(self._local_results, result)

    def _make_input_workers(self):
        return input_lib.InputWorkers(
            values.SingleDeviceMap(self._compute_device),
            [(self._host_device, [self._compute_device])])

    def _make_input_context(self):
        return distribute_lib.InputContext(
            num_input_pipelines=hvd.size(),
            input_pipeline_id=hvd.rank(),
            num_replicas_in_sync=hvd.size())

    def _experimental_distribute_dataset(self, dataset):
        return input_lib.get_distributed_dataset(
            dataset.repeat(), # TODO: hacky hack?
            self._make_input_workers(),
            self._container_strategy(),
            split_batch_by=hvd.size(),
            input_context=self._make_input_context())

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        return numpy_dataset.one_host_numpy_dataset(
            numpy_input, numpy_dataset.SingleDevice(self._host_device), session)

    def _reduce_to(self, reduce_op, value, destinations):
        return hvd.allreduce(value, average=(tf.VariableAggregation.MEAN == reduce_op))

    @property
    def experimental_between_graph(self):
        return True

    @property
    def experimental_should_init(self):
        return True

    @property
    def should_checkpoint(self):
        return hvd.rank() == 0

    @property
    def should_save_summary(self):
        return hvd.rank() == 0

    @property
    def _num_replicas_in_sync(self):
        return hvd.size()

    @property
    def _global_batch_size(self):
        return True
