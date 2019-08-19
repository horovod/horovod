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
import tensorflow as tf


def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse,
                                 compression, sparse_as_dense):
    class _DistributedOptimizerWithApplyGradients(keras.optimizers.Optimizer):
        def __init__(self, name, device_dense, device_sparse, compression, sparse_as_dense,
                     config):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._allreduce_grads = hvd._make_allreduce_grads_fn(
                name, device_dense, device_sparse, compression, sparse_as_dense)
            super(self.__class__, self).__init__(**config)

        def apply_gradients(self, grads_and_vars, *args, **kwargs):
            """Apply gradients to provided variables.

            See Optimizer.apply_gradients() for more info.

            In DistributedOptimizer, apply_gradients() is overriden to also
            allreduce the gradients before applying them.
            """
            if hvd.size() > 1:
                grads, vars = zip(*grads_and_vars)
                avg_grads = self._allreduce_grads(grads)
                grads_and_vars = list(zip(avg_grads, vars))
            return super(self.__class__, self).apply_gradients(grads_and_vars, *args, **kwargs)

    class _DistributedOptimizerWithGetGradients(keras.optimizers.Optimizer):
        def __init__(self, name, device_dense, device_sparse, compression, sparse_as_dense,
                     config):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._allreduce_grads = hvd._make_allreduce_grads_fn(
                name, device_dense, device_sparse, compression, sparse_as_dense)
            super(self.__class__, self).__init__(**config)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = super(self.__class__, self).get_gradients(loss, params)
            if hvd.size() > 1:
                return self._allreduce_grads(gradients)
            else:
                return gradients

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    if hasattr(optimizer, 'apply_gradients'):
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                dict(_DistributedOptimizerWithApplyGradients.__dict__))
    else:
        cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                dict(_DistributedOptimizerWithGetGradients.__dict__))
    return cls(name, device_dense, device_sparse, compression, sparse_as_dense,
               optimizer.get_config())


def _eval(backend, op_or_result):
    if hvd._executing_eagerly():
        return op_or_result
    else:
        return backend.get_session().run(op_or_result)


if hasattr(hvd, 'broadcast_global_variables'):
    def broadcast_global_variables(backend, root_rank):
        return _eval(backend, hvd.broadcast_global_variables(root_rank))


def allreduce(backend, value, name, average):
    return _eval(backend, hvd.allreduce(tf.constant(value, name=name), average=average))


def allgather(backend, value, name):
    return _eval(backend, hvd.allgather(tf.constant(value, name=name)))


def broadcast(backend, value, root_rank, name):
    return _eval(backend, hvd.broadcast(tf.constant(value, name=name), root_rank))


def load_model(keras, wrap_optimizer, filepath, custom_optimizers, custom_objects):
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ == keras.optimizers.Optimizer.__module__
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
