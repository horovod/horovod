# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
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

import horovod.tensorflow as hvd
from horovod.tensorflow import init
from horovod.tensorflow import shutdown
from horovod.tensorflow import size
from horovod.tensorflow import local_size
from horovod.tensorflow import rank
from horovod.tensorflow import local_rank
from horovod.tensorflow import mpi_threads_supported
from horovod.keras import callbacks

import tensorflow as tf


class DistributedOptimizerImpl:
    def __init__(self, name, device_dense, device_sparse, **kwargs):
        if name is None:
            name = "Distributed%s" % self.__class__.__base__.__name__
        self._name = name
        self._device_dense = device_dense
        self._device_sparse = device_sparse

    def get_gradients(self, loss, params):
        """
        Compute gradients of all trainable variables.

        See Optimizer.get_gradients() for more info.

        In DistributedOptimizer, get_gradients() is overriden to also
        allreduce the gradients before returning them.
        """
        gradients = super(self.__class__, self).get_gradients(loss, params)
        if hvd.size() > 1:
            averaged_gradients = []
            with tf.name_scope(self._name + "_Allreduce"):
                for grad in gradients:
                    if grad is not None:
                        avg_grad = hvd.allreduce(grad, device_dense=self._device_dense,
                                                 device_sparse=self._device_sparse)
                        averaged_gradients.append(avg_grad)
                    else:
                        averaged_gradients.append(None)
                return averaged_gradients
        else:
            return gradients


def broadcast_global_variables(backend, root_rank):
    bcast_op = hvd.broadcast_global_variables(root_rank)
    return backend.get_session().run(bcast_op)


def allreduce(backend, value, name=None, average=True):
    allreduce_op = hvd.allreduce(tf.constant(value, name=name), average=average)
    return backend.get_session().run(allreduce_op)


def allgather(backend, value, name=None):
    allgather_op = hvd.allgather(tf.constant(value, name=name))
    return backend.get_session().run(allgather_op)


def broadcast(backend, value, root_rank, name=None):
    bcast_op = hvd.broadcast(tf.constant(value, name=name), root_rank)
    return backend.get_session().run(bcast_op)


def load_model(keras, wrap_optimizer, filepath, custom_optimizers=None, custom_objects=None):
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ == 'keras.optimizers'
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
