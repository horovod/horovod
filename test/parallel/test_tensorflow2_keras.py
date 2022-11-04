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

"""Tests for horovod.tensorflow.keras in TensorFlow 2."""

import tensorflow as tf
import numpy as np
import warnings

from packaging import version
from parameterized import parameterized

import pytest

from tensorflow import keras

from horovod.common.util import is_version_greater_equal_than

if is_version_greater_equal_than(tf.__version__, "2.6.0"):
    if version.parse(keras.__version__) < version.parse("2.9.0"):
        from keras.optimizer_v2 import optimizer_v2
    else:
        from keras.optimizers.optimizer_v2 import optimizer_v2
else:
    from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import horovod.keras as hvd_keras
import horovod.tensorflow.keras as hvd


_PRE_TF_2_2_0 = version.parse(tf.__version__) < version.parse("2.2.0")


@pytest.mark.skipif(version.parse(tf.__version__) <
                    version.parse('2.0.0'), reason='TensorFlow v2 tests')
class Tf2KerasTests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.keras.
    """

    def __init__(self, *args, **kwargs):
        super(Tf2KerasTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        hvd.init()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[hvd.local_rank()], 'GPU')

    def test_train_model_lr_schedule(self):
        initial_lr = 0.1 * hvd.size()
        if version.parse(tf.keras.__version__) < version.parse("2.11"):
            opt = tf.keras.optimizers.Adam()
        else:
            opt = tf.keras.optimizers.legacy.Adam()
        opt = hvd.DistributedOptimizer(opt)

        def linear_multiplier(epoch):
            return epoch

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2, input_shape=(3,)))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.ThresholdedReLU(0.5))
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=opt,
                      metrics=[keras.metrics.categorical_accuracy],
                      experimental_run_tf_function=False)
        x = np.random.random((10, 3))
        y = np.random.random((10, 3, 2))

        class StoreLearningRateCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                # test learning rate warmup
                lr = self.model.optimizer.lr.numpy()
                if epoch >= 0 and epoch < 5:
                    assert lr <= initial_lr or np.isclose(lr, initial_lr)

                # # test learning rate schedule callback
                if epoch > 5 and epoch < 10:
                    assert lr <= initial_lr * \
                        1e-1 or np.isclose(lr, initial_lr * 1e-1)
                if epoch > 10 and epoch < 15:
                    assert lr < initial_lr * \
                        1e-2 or np.isclose(lr, initial_lr * 1e-2)
                if epoch >= 15 and epoch < 20:
                    assert np.isclose(
                        lr, initial_lr * linear_multiplier(epoch))

        # No assertions needed for BroadcastGlobalVariableCallbacks
        # We just need to verify that it doesn't hang or error
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            hvd.callbacks.LearningRateWarmupCallback(
                initial_lr=initial_lr,
                warmup_epochs=5),
            hvd.callbacks.LearningRateScheduleCallback(
                initial_lr=initial_lr,
                multiplier=1e-1,
                start_epoch=5,
                end_epoch=10),
            hvd.callbacks.LearningRateScheduleCallback(
                initial_lr=initial_lr,
                multiplier=1e-2,
                start_epoch=10,
                end_epoch=15),
            hvd.callbacks.LearningRateScheduleCallback(
                initial_lr=initial_lr,
                multiplier=linear_multiplier,
                start_epoch=15,
                end_epoch=20),
            StoreLearningRateCallback()]
        train_history = model.fit(x,
                                  y,
                                  steps_per_epoch=5,
                                  callbacks=callbacks,
                                  epochs=20)

        # test that the metrics average is being respected
        loss_metrics = train_history.history["loss"]
        loss_metrics_tensor = tf.convert_to_tensor(
            loss_metrics, dtype=tf.float32)
        expected_loss_metrics_tensor = hvd.broadcast(
            loss_metrics_tensor, root_rank=0)
        self.assertAllClose(expected_loss_metrics_tensor, loss_metrics_tensor)

    def test_sparse_as_dense(self):
        opt = keras.optimizers.RMSprop(lr=0.0001)
        opt = hvd.DistributedOptimizer(opt, sparse_as_dense=True)

        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(1000, 64, input_length=10))
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=opt,
                      experimental_run_tf_function=False)

        x = np.random.randint(1000, size=(32, 10))
        y = np.random.random((32, 10, 64))
        # No assertions, we just need to verify that it doesn't hang
        model.train_on_batch(x, y)

    def test_sparse_as_dense_with_grad_aggregation(self):
        backward_passes_per_step = 2
        if version.parse(keras.__version__) < version.parse("2.11"):
            opt = keras.optimizers.RMSprop(lr=0.0001)
        else:
            opt = keras.optimizers.legacy.RMSprop(lr=0.0001)
        opt = hvd.DistributedOptimizer(
            opt,
            sparse_as_dense=True,
            backward_passes_per_step=backward_passes_per_step
        )

        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(1000, 64, input_length=10))
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=opt,
                      experimental_run_tf_function=False)

        x = np.random.randint(1000, size=(32, 10))
        y = np.random.random((32, 10, 64))

        training_steps = 3
        for _ in range(training_steps):
            model.train_on_batch(x, y)

        aggregation_counter = opt._agg_helper.counter.numpy()
        assert aggregation_counter == training_steps % backward_passes_per_step

    def test_grad_aggregation_with_inf_grad(self):
        backward_passes_per_step = 2
        step_count = tf.Variable(0, trainable=False, dtype=tf.int32)
        if version.parse(tf.keras.__version__) < version.parse("2.11"):
            opt = tf.keras.optimizers.SGD()
        else:
            opt = tf.keras.optimizers.legacy.SGD()
        opt = hvd.DistributedOptimizer(
            opt,
            backward_passes_per_step=backward_passes_per_step,
            sparse_as_dense=True
        )
        x = tf.Variable(0.)
        var = [x]

        def loss():
            step_count.assign_add(1)
            return tf.cond(
                pred=tf.greater(step_count, 1),
                true_fn=lambda: x,
                false_fn=lambda: x * float('inf')
            )
        for _ in range(2 * backward_passes_per_step):
            # in the first aggregation cycle the gradient is infinite,
            # and it should be cleaned up to zero after apply_gradients
            # and doesn't affect the 2nd aggregation cycle
            grads_and_vars = opt._compute_gradients(loss=loss, var_list=var)
            opt.apply_gradients(grads_and_vars)
        assert tf.math.is_finite(grads_and_vars[0][0])

    def test_from_config(self):
        if version.parse(keras.__version__) < version.parse("2.11"):
            opt = keras.optimizers.Adam()
        else:
            opt = keras.optimizers.legacy.Adam()
        hopt = hvd.DistributedOptimizer(opt)
        cfg = hopt.get_config()

        hopt_copy1 = hopt.from_config(cfg)
        self.assertEqual(cfg, hopt_copy1.get_config())

        hopt_copy2 = hopt.__class__.from_config(cfg)
        self.assertEqual(cfg, hopt_copy2.get_config())

    def test_elastic_state(self):
        v = 1.0 if hvd.rank() == 0 else 2.0
        model1 = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model1.build((2, 2))
        model1.set_weights(
            [np.array([[v, v], [v, v]], dtype=np.float32),
             np.array([v, v], dtype=np.float32)])

        model2 = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model2.build((2, 2))
        model2.set_weights(
            [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
             np.array([0.0, 0.0], dtype=np.float32)])

        optimizer = tf.optimizers.Adam(0.001 * hvd.size())

        state = hvd.elastic.KerasState(
            model1,
            optimizer,
            batch=20 +
            hvd.rank(),
            epoch=10 +
            hvd.rank())
        state.sync()

        model1_weights = model1.get_weights()
        model2_weights = model2.get_weights()

        # After sync, all values should match the root rank
        for w in state.model.get_weights():
            self.assertAllClose(w, np.ones_like(w))
        assert state.batch == 20
        assert state.epoch == 10

        # Partially modify then restore
        model1.set_weights(model2_weights)
        state.batch = 21
        state.epoch = 11

        state.restore()

        for w1, w2 in zip(model1.get_weights(), model1_weights):
            self.assertAllClose(w1, w2)
        assert state.batch == 20
        assert state.epoch == 10

        # Partially modify then commit
        model1.set_weights(model2_weights)
        state.batch = 21
        state.epoch = 11

        state.commit()
        state.restore()

        for w1, w2 in zip(model1.get_weights(), model2_weights):
            self.assertAllClose(w1, w2)
        assert state.batch == 21
        assert state.epoch == 11

    @parameterized.expand([
        [True],
        [False]
    ])
    def test_gradient_aggregation(self, average_aggregated_gradients):
        class TestingOptimizer(optimizer_v2.OptimizerV2):
            """
            Custom optimizer we use for testing gradient aggregation.
            """

            def get_config(self):
                config = super(TestingOptimizer, self).get_config()
                return config

            def _create_slots(self, var_list):
                # Only needed for TF < 2.2.
                pass

            def _resource_apply_dense(self, grad, var, apply_state=None):
                return var.assign_add(grad)

        backward_passes_per_step = 4
        hvd_optimizer = hvd.DistributedOptimizer(
            optimizer=TestingOptimizer("test"),
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=average_aggregated_gradients,
            sparse_as_dense=True,
        )
        _ = hvd_optimizer.iterations

        x_0 = 0.0
        y_0, y_1 = 1.0, 2.0
        x = tf.Variable([x_0])
        y = tf.Variable([y_0, y_1])
        variables = [x, y]

        def loss():
            """
            loss = x - y * e1 where e1 = [1.0, 0.0]
            """
            # Gather the first row of y. It is equivalent to y * e1.
            # Use tf.gather to produce tf.IndexedSlices gradient to improve test coverage.
            gathered_y_1 = tf.gather(y, [0])
            return x - gathered_y_1

        def compute_expected_value(batch_id):
            """
            Given the loss function above, the gradient of x and y can be derived.
              dloss/dx = 1.0
              dloss/dy = [-1.0, 0.0]
            Therefore, for each step, the value of x increases by 1 and
            the value of y increases by [-1.0, 0.0].
            """
            num_of_steps = (batch_id + 1) // backward_passes_per_step

            gradient_of_x = num_of_steps * 1.0
            gradient_of_y_0 = num_of_steps * -1.0

            if not average_aggregated_gradients:
                gradient_of_x *= backward_passes_per_step
                gradient_of_y_0 *= backward_passes_per_step

            # Add gradient with its initial values because the TestingOptimizer optimize
            # variable by assign_add.
            expected_x = x_0 + gradient_of_x
            expected_y_0 = y_0 + gradient_of_y_0
            # It should remain constant as gradient is always 0.
            expected_y_1 = y_1

            return np.array(expected_x), np.array([expected_y_0, expected_y_1])

        @tf.function
        def compute_and_apply_gradients_in_tf_function(var_list, **kwargs):
            # Compute and apply gradient updates in tf.function to reproduce
            # how it is done inside `model.fit()`.
            grads_and_vars = hvd_optimizer._compute_gradients(
                loss, var_list=var_list)
            hvd_optimizer.apply_gradients(grads_and_vars, **kwargs)

        total_num_of_steps = 10
        for idx in range(total_num_of_steps):
            if _PRE_TF_2_2_0:
                compute_and_apply_gradients_in_tf_function(var_list=variables)
            else:
                # In 2.2 and 2.3 the horovod optimizer sets `_HAS_AGGREGATE_GRAD = True`.
                # This configures tf.keras to call `_aggregate_gradients()` outside of
                # `apply_gradients()` and to set `experimental_aggregate_gradients` to
                # False when calling `apply_gradients()` to prevent it from calling
                # `_aggregate_gradients()` again.
                compute_and_apply_gradients_in_tf_function(
                    var_list=variables,
                    experimental_aggregate_gradients=False)

            expected_x, expected_y = compute_expected_value(idx)
            updated_x = variables[0].numpy()
            updated_y = variables[1].numpy()
            assert np.isclose(updated_x, expected_x)
            assert np.isclose(updated_y, expected_y).all()
            assert idx + 1 == hvd_optimizer.iterations.numpy()

        aggregation_counter = hvd_optimizer._agg_helper.counter.numpy()
        assert aggregation_counter == total_num_of_steps % backward_passes_per_step

    @parameterized.expand([
        [True],
        [False]
    ])
    def test_gradient_aggregation_with_local_vars(self, average_aggregated_gradients):
        class TestingOptimizer(optimizer_v2.OptimizerV2):
            """
            Custom optimizer we use for testing gradient aggregation.
            """

            def get_config(self):
                config = super(TestingOptimizer, self).get_config()
                return config

            def _create_slots(self, var_list):
                # Only needed for TF < 2.2.
                pass

            def _resource_apply_dense(self, grad, var, apply_state=None):
                return var.assign_add(grad)

        backward_passes_per_step = 4
        hvd_optimizer = hvd.DistributedOptimizer(
            optimizer=TestingOptimizer("test"),
            backward_passes_per_step=backward_passes_per_step,
            average_aggregated_gradients=average_aggregated_gradients,
            sparse_as_dense=True,
        )

        _ = hvd_optimizer.iterations

        total_num_variables = 8
        num_local_vars = 4
        X_0 = [0.0]*total_num_variables
        Y_0, Y_1 = [1.0]*total_num_variables, [2.0]*total_num_variables
        X = [tf.Variable([x_0]) for x_0 in X_0]
        Y = [tf.Variable([y_0, y_1]) for y_0, y_1 in zip(Y_0, Y_1)]
        variables = [X, Y]

        for i in range(num_local_vars):
            x_var = X[i]
            y_var = Y[i]
            hvd_optimizer.register_local_var(x_var)
            hvd_optimizer.register_local_var(y_var)

        def loss():
            """
            loss = x - y * er where er = [float(hvd.rank()+1), 0.0]
            """
            # Gather the first row of y. It is equivalent to y * er.
            # Use tf.gather to produce tf.IndexedSlices gradient to improve test coverage.
            gathered_y_1 = tf.gather(Y, [0], axis=1)
            return X - (float(hvd.rank()+1) * gathered_y_1)

        def compute_expected_value(batch_id):
            """
            Given the loss function above, the gradient of x and y can be derived.
              dloss/dx = 1.0
              dloss/dy = [-float(hvd.rank()+1), 0.0]
            Therefore, for each step, the value of x increases by 1.0 and
            the value of y increases by [-float(hvd.rank()), 0.0].
            """
            num_of_steps = (batch_id + 1) // backward_passes_per_step

            gradient_of_x = num_of_steps * 1.0

            # At each rank, the gradient of y_0 evaluates to (hvd.rank()+1).
            # For non-local variables we need to average the value accross the ranks.
            # The average value is (1+2+...hvd.size())/hvd.size() wich is
            # equivalent to the following expression:
            gradient_of_y_multiplier = float(hvd.size()+1)/2
            gradient_of_y_0 = num_of_steps * gradient_of_y_multiplier * -1.0

            if not average_aggregated_gradients:
                gradient_of_x *= backward_passes_per_step
                gradient_of_y_0 *= backward_passes_per_step

            expected_x = np.array(X_0)

            for i in range(len(X_0)):
                if i < num_local_vars:
                    expected_x[i] += gradient_of_x/hvd.size()
                else:
                    expected_x[i] += gradient_of_x

            expected_y_0 = np.array(Y_0)
            for i in range(len(Y_0)):
                if i < num_local_vars:
                    # recover the gradient of local vars
                    expected_y_0[i] += ((gradient_of_y_0/gradient_of_y_multiplier)*float(hvd.rank()+1))/hvd.size()
                else:
                    expected_y_0[i] += gradient_of_y_0

            # It should remain constant as gradient is always 0.
            expected_y_1 = np.array(Y_1)
            expected_y = [[ey0, ey1] for ey0,ey1 in zip(expected_y_0, expected_y_1)]

            return np.reshape(expected_x, (-1, 1)), np.array(expected_y)

        @tf.function
        def compute_and_apply_gradients_in_tf_function(var_list, **kwargs):
            # Compute and apply gradient updates in tf.function to reproduce
            # how it is done inside `model.fit()`.
            grads_and_vars = hvd_optimizer._compute_gradients(
                loss, var_list=var_list)
            hvd_optimizer.apply_gradients(grads_and_vars, **kwargs)

        total_num_of_steps = 10
        for idx in range(total_num_of_steps):
            if _PRE_TF_2_2_0:
                compute_and_apply_gradients_in_tf_function(var_list=variables)
            else:
                # In 2.2 and 2.3 the horovod optimizer sets `_HAS_AGGREGATE_GRAD = True`.
                # This configures tf.keras to call `_aggregate_gradients()` outside of
                # `apply_gradients()` and to set `experimental_aggregate_gradients` to
                # False when calling `apply_gradients()` to prevent it from calling
                # `_aggregate_gradients()` again.
                compute_and_apply_gradients_in_tf_function(
                    var_list=variables,
                    experimental_aggregate_gradients=False)

            expected_x, expected_y = compute_expected_value(idx)
            updated_x = np.array(variables[0])
            updated_y = np.array(variables[1])
            assert np.isclose(updated_x, expected_x).all()
            assert np.isclose(updated_y, expected_y).all()
            assert idx + 1 == hvd_optimizer.iterations.numpy()

        aggregation_counter = hvd_optimizer._agg_helper.counter.numpy()
        assert aggregation_counter == total_num_of_steps % backward_passes_per_step

    def test_partial_distributed_optimizer(self):
        """ Note: test makes most sense with more than 1 nodes. """
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        # the keras model has 3 layers, we test cases with 0, 1, and 2 local layers.
        for num_local_layers in range(3):
            model = tf.keras.models.Sequential()
            initializer = tf.keras.initializers.Constant(hvd.rank())
            model.add(tf.keras.layers.Dense(2, input_shape=(3,), kernel_initializer=initializer, bias_initializer=initializer))
            model.add(tf.keras.layers.RepeatVector(3))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, kernel_initializer=initializer, bias_initializer=initializer)))
            if version.parse(tf.keras.__version__) < version.parse("2.11"):
                opt = tf.keras.optimizers.Adam()
            else:
                opt = tf.keras.optimizers.legacy.Adam()
            model.compile(loss=tf.keras.losses.MSE,
                            metrics=[tf.keras.metrics.categorical_accuracy])

            X = np.random.random((1, 3))
            Y = np.random.random((1, 3, 3))

            try:
                init = tf.global_variables_initializer()
            except AttributeError:
                init = tf.compat.v1.global_variables_initializer()
            self.evaluate(init)

            with tf.GradientTape(persistent=True) as tape:
                p = model(X, training=True)
                l = model.loss(Y, p)

            gradients_tape = tape.gradient(l, model.trainable_weights)

            # deem local layers
            local_layers = model.layers[:num_local_layers]

            opt = hvd_keras.PartialDistributedOptimizer(opt, sparse_as_dense=True, local_layers=local_layers)
            gradients_vars_opt = opt._compute_gradients(l, model.trainable_weights, tape=tape)

            var_grad_tape = {var.ref():grad for var,grad in zip(model.trainable_weights, gradients_tape)}
            var_grad_opt = {var.ref():grad for grad,var in gradients_vars_opt}
            local_vars = [var.ref() for layer in local_layers for var in layer.trainable_weights]

            for var in model.trainable_weights:
                if var.ref() in local_vars:
                    # local gradients should not change.
                    self.assertAllClose(var_grad_tape[var.ref()], hvd.size()*var_grad_opt[var.ref()])
                else:
                    # non-local gradients shouldn't be equal given that the initial weights are set to ranks
                    self.assertNotAllClose(var_grad_tape[var.ref()], var_grad_opt[var.ref()])

    def test_broadcast_global_variables_callback_with_local_vars(self):
        """ Note: test makes most sense with more than 1 nodes. """
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        root_rank = hvd.size() - 1

        X = np.random.random((1, 3))
        Y = np.random.random((1, 3, 3))

        # all of the model's layers are set to be non-trainable
        # so that we can assert their inital variable values are
        # as expected after the call to model.fit() with the call back
        model = tf.keras.models.Sequential()
        # the inital value for all variables is set ot each rank's value
        initializer = tf.keras.initializers.Constant(hvd.rank())
        model.add(tf.keras.layers.Dense(3, input_shape=(3,), kernel_initializer=initializer,
                                        bias_initializer=initializer, trainable=False))
        model.add(tf.keras.layers.Dense(3, kernel_initializer=initializer, bias_initializer=initializer, trainable=False))
        model.add(tf.keras.layers.Dense(3, kernel_initializer=initializer, bias_initializer=initializer, trainable=False))
        model.add(tf.keras.layers.Dense(3, kernel_initializer=initializer, bias_initializer=initializer, trainable=False))
        model.add(tf.keras.layers.RepeatVector(3))
        model.add(tf.keras.layers.Dense(3, kernel_initializer=initializer, bias_initializer=initializer, trainable=False))
        model.compile(loss=tf.keras.losses.MSE,
                        metrics=[tf.keras.metrics.categorical_accuracy])

        # deem first layer's variables as local
        local_variables = [var for var in model.layers[0].variables]
        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=root_rank, local_variables=local_variables)]
        model.fit(X, Y, callbacks=callbacks)
        local_variables_ref = [var.ref() for var in local_variables]
        for var in model.variables:
            if var.ref() in local_variables_ref:
                # the value for local variables should not change and remain as rank's value
                self.assertAllClose(var, tf.constant(float(hvd.rank()), shape=var.shape))
            else:
                # the value for non-local variables should equal the root rank's value
                self.assertAllClose(var, tf.constant(float(root_rank), shape=var.shape))

