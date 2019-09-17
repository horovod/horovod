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
                                 compression, sparse_as_dense, aggregation_frequency):
    class _DistributedOptimizer(keras.optimizers.Optimizer):
        def __init__(self, name, device_dense, device_sparse, compression, sparse_as_dense,
                     config, aggregation_frequency):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._name = name
            self._device_dense = device_dense
            self._device_sparse = device_sparse
            self._compression = compression
            self._sparse_as_dense = sparse_as_dense
            self._get_gradients_used = False

            # How often are parameters synchronized
            self._aggregation_frequency = aggregation_frequency
            assert self._aggregation_frequency > 0

            # This is going to be N data structure holding the per-GPU aggregated gradient updates
            # for parameter updates. N is the number of parameters.
            self.gpu_shadow_vars = []

            # Used to know when it can begin reading aggregated values.
            self.counter = None

            super(self.__class__, self).__init__(**config)

        def get_gradients(self, loss, params):
            """
            Compute gradients of all trainable variables.

            See Optimizer.get_gradients() for more info.

            In DistributedOptimizer, get_gradients() is overriden to also
            allreduce the gradients before returning them.
            """
            def init_aggregation_vars():
                v = tf.get_collection('aggregation_variables')
                vars_init_op = tf.variables_initializer(v)
                sess = tf.keras.backend.get_session(op_input_list=())

                with tf.variable_scope("aggregation_variables"):
                    self.counter = tf.get_variable(
                        "aggregation_counter", shape=(), dtype=tf.int32,
                        trainable=False, initializer=tf.zeros_initializer())
                    if self._aggregation_frequency > 1:
                        for idx, grad in enumerate(self.grads):
                            if self._sparse_as_dense and \
                                    isinstance(grad, tf.IndexedSlices):
                                grad = tf.convert_to_tensor(grad)
                            grad_aggregation_variable_name = str(idx)
                            grad_aggregation_variable = tf.get_variable(
                                grad_aggregation_variable_name, shape=grad.get_shape().as_list(),
                                trainable=False, initializer=tf.zeros_initializer(),
                                collections=[tf.GraphKeys.LOCAL_VARIABLES, "aggregating_collection"])
                            self.gpu_shadow_vars.append(
                                grad_aggregation_variable)
                        assert len(self.gpu_shadow_vars) == len(self.grads)
                    vars_init_op = tf.variables_initializer(
                        [self.counter, *self.gpu_shadow_vars])
                    sess.run(vars_init_op)

            def clear_grads():
                clear_ops_list = []
                for idx, grad in enumerate(self.gpu_shadow_vars):
                    grad_aggregation_variable_name = str(idx)
                    grad_aggregator = tf.get_variable(
                        grad_aggregation_variable_name)
                    clear_op = grad_aggregator.assign(
                        grad_aggregator.initial_value)
                    clear_ops_list.append(clear_op)
                return tf.group(*clear_ops_list)

            def aggregate_grads():
                aggregation_ops_list = []
                if self._aggregation_frequency > 1:
                    for idx, grad in enumerate(self.grads):
                        if self._sparse_as_dense and \
                                isinstance(grad, tf.IndexedSlices):
                            grad = tf.convert_to_tensor(grad)
                        grad_aggregation_variable_name = str(idx)
                        grad_aggregator = tf.get_variable(
                            grad_aggregation_variable_name)
                        update_op = grad_aggregator.assign_add(grad)
                        aggregation_ops_list.append(update_op)
                return aggregation_ops_list

            def allreduce_grads():
                if self._aggregation_frequency > 1:
                    # Read in latest variables values.
                    aggregated_grads = []
                    aggregation_read_ops_list = []
                    with tf.variable_scope("aggregation_variables", reuse=True):
                        for idx, grad in enumerate(self.gpu_shadow_vars):
                            grad_aggregation_variable_name = str(idx)
                            grad_aggregator = tf.get_variable(
                                grad_aggregation_variable_name)
                            aggregated_grads.append(
                                grad_aggregator.read_value())
                            aggregation_read_ops_list.append(
                                aggregated_grads[idx])
                    aggregation_read_ops = tf.group(
                        *aggregation_read_ops_list)
                else:
                    aggregated_grads = self.grads
                    aggregation_read_ops = tf.no_op()

                with tf.control_dependencies([aggregation_read_ops]):
                    averaged_gradients = []
                    for idx, grad in enumerate(aggregated_grads):
                        if grad is not None:
                            avg_grad = hvd.allreduce(grad,
                                                     device_dense=self._device_dense,
                                                     device_sparse=self._device_sparse,
                                                     compression=self._compression)
                            averaged_gradients.append(avg_grad)
                        else:
                            averaged_gradients.append(None)
                    with tf.control_dependencies([g.op for g in averaged_gradients]):
                        reset_op = self.counter.assign(
                            tf.constant(0), use_locking=True)
                    with tf.control_dependencies([reset_op]):
                        return [tf.identity(g) for g in averaged_gradients]

            self._get_gradients_used = True
            self.grads = super(
                self.__class__, self).get_gradients(loss, params)
            init_aggregation_vars()
            if hvd.size() > 1:
                if self._aggregation_frequency > 1:
                    with tf.variable_scope("aggregation_variables", reuse=True):
                        clear_op = tf.cond(
                            tf.equal(self.counter, 0), clear_grads, tf.no_op)
                        with tf.control_dependencies([clear_op]):
                            aggregation_ops_list = aggregate_grads()

                    aggregation_ops = tf.group(*aggregation_ops_list)
                    with tf.control_dependencies([aggregation_ops]):
                        update_ops = [self.counter.assign_add(tf.constant(1))]
                else:
                    update_ops = []
                with tf.control_dependencies(update_ops):
                    return tf.cond(
                        tf.logical_or(tf.equal(self._aggregation_frequency, 1), tf.equal(self.counter, self._aggregation_frequency)),
                        allreduce_grads,
                        lambda: self.grads,
                    )
            else:
                return self.grads

        def apply_gradients(self, *args, **kwargs):
            if not self._get_gradients_used:
                raise Exception('`apply_gradients()` was called without a call to '
                                '`get_gradients()`. If you\'re using TensorFlow 2.0, '
                                'please specify `experimental_run_tf_function=False` in '
                                '`compile()`.')
            flattended_args0 = [item for tup in args[0] for item in tup]
            with tf.control_dependencies(flattended_args0):
                return tf.cond(tf.equal(self.counter, 0), lambda: super(self.__class__, self).apply_gradients(*args, **kwargs), tf.no_op)

    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override get_gradients() method with an allreduce implementation.
    # This class will have the same name as the optimizer it's wrapping, so that the saved
    # model could be easily restored without Horovod.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(_DistributedOptimizer.__dict__))
    return cls(name, device_dense, device_sparse, compression, sparse_as_dense,
               optimizer.get_config(), aggregation_frequency)


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
