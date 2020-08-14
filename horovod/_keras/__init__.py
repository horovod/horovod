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

from distutils.version import LooseVersion

import horovod.tensorflow as hvd
import tensorflow as tf


_PRE_TF_2_4_0 = LooseVersion(tf.__version__) < LooseVersion('2.4.0')


def create_distributed_optimizer(keras, optimizer, name, device_dense, device_sparse,
                                 compression, sparse_as_dense, gradient_predivide_factor):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        _HAS_AGGREGATE_GRAD = True

        def __init__(self, **kwargs):
            self._name = name or "Distributed%s" % self.__class__.__base__.__name__
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense
            self._aggregated_gradients = False
            self._gradient_predivide_factor = gradient_predivide_factor
            super(self.__class__, self).__init__(**kwargs)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            gradients = super(self.__class__, self).get_gradients(loss, params)
            return self._allreduce(gradients)

        def _aggregate_gradients(self, grads_and_vars):
            grads, vars = list(zip(*grads_and_vars))
            aggregated_grads = self._allreduce(grads)
            if _PRE_TF_2_4_0:
                # Prior to TF 2.4.0, this function was expected to return only a list of
                # grads, not a list of (grad, var) tuples.
                return aggregated_grads
            return list(zip(aggregated_grads, vars))

        def _allreduce(self, gradients):
            self._aggregated_gradients = True
            if hvd.size() > 1:
                if self._gradient_predivide_factor != 1.0:
                    # Perform averaging via pre/postscaling factors.
                    # Split average operation across pre/postscale factors
                    prescale_factor = 1.0 / gradient_predivide_factor
                    postscale_factor = gradient_predivide_factor / hvd.size()
                    do_average = False
                else:
                    prescale_factor = 1.0
                    postscale_factor = 1.0
                    do_average = True

                averaged_gradients = []
                with tf.name_scope(self._name + "_Allreduce"):
                    for grad in gradients:
                        if grad is not None:
                            if self._sparse_as_dense and \
                                    isinstance(grad, tf.IndexedSlices):
                                grad = tf.convert_to_tensor(grad)
                            avg_grad = hvd.allreduce(grad,
                                                     average=do_average,
                                                     device_dense=self._device_dense,
                                                     device_sparse=self._device_sparse,
                                                     compression=self._compression,
                                                     prescale_factor=prescale_factor,
                                                     postscale_factor=postscale_factor)
                            averaged_gradients.append(avg_grad)
                        else:
                            averaged_gradients.append(None)
                    return averaged_gradients
            else:
                return gradients

        def apply_gradients(self, *args, **kwargs):
            results = super(self.__class__, self).apply_gradients(*args, **kwargs)
            if not self._aggregated_gradients:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()` or `_aggregate_gradients`. If you\'re '
                                'using TensorFlow 2.0, please specify '
                                '`experimental_run_tf_function=False` in `compile()`.')
            return results

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls.from_config(optimizer.get_config())


def _eval(backend, op_or_result):
    if hvd._executing_eagerly():
        return op_or_result
    else:
        return backend.get_session().run(op_or_result)


if hasattr(hvd, 'broadcast_global_variables'):
    def broadcast_global_variables(backend, root_rank):
        return _eval(backend, hvd.broadcast_global_variables(root_rank))


def allreduce(backend, value, name, average, prescale_factor, postscale_factor):
    return _eval(backend, hvd.allreduce(tf.constant(value, name=name), average=average,
                                        prescale_factor=prescale_factor,
                                        postscale_factor=postscale_factor))


def allgather(backend, value, name):
    return _eval(backend, hvd.allgather(tf.constant(value, name=name)))


def broadcast(backend, value, root_rank, name):
    return _eval(backend, hvd.broadcast(tf.constant(value, name=name), root_rank))


def load_model(keras, wrap_optimizer, optimizer_modules, filepath, custom_optimizers, custom_objects):
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in keras.optimizers.Optimizer.__subclasses__()
        if subclass.__module__ in optimizer_modules
    }

    if custom_optimizers is not None:
        horovod_objects.update({
            cls.__name__: wrap_optimizer(cls)
            for cls in custom_optimizers
        })

    if custom_objects is not None:
        horovod_objects.update(custom_objects)

    return keras.models.load_model(filepath, custom_objects=horovod_objects)
