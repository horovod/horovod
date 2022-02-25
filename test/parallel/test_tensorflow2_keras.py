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

import math
import tensorflow as tf
import numpy as np
import os
import warnings

from distutils.version import LooseVersion

import pytest

from tensorflow import keras

from horovod.common.util  import is_version_greater_equal_than

if is_version_greater_equal_than(tf.__version__, "2.6.0"):
    if LooseVersion(keras.__version__) < LooseVersion("2.9.0"):
        from keras.optimizer_v2 import optimizer_v2
    else:
        from keras.optimizers.optimizer_v2 import optimizer_v2
else:
    from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import horovod.tensorflow.keras as hvd


_PRE_TF_2_2_0 = LooseVersion(tf.__version__) < LooseVersion("2.2.0")

# Set environment variable to enable adding/removing process sets after
# initializing Horovod.
os.environ["HOROVOD_DYNAMIC_PROCESS_SETS"] = "1"


@pytest.mark.skipif(LooseVersion(tf.__version__) <
                    LooseVersion('2.0.0'), reason='TensorFlow v2 tests')
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
        opt = tf.keras.optimizers.Adam()
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

    def test_from_config(self):
        opt = keras.optimizers.Adam()
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

    def test_gradient_aggregation(self):
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
            average_aggregated_gradients=True,
        )
        _ = hvd_optimizer.iterations

        def compute_expected_value(batch_id):
            sum_per_aggregation = 0.0
            for _ in range(backward_passes_per_step):
                grads_for_batch = 0.0

                # Apply `average_aggregated_gradients`.
                grads_for_batch /= float(backward_passes_per_step)

                # Averages across workers.
                sum_per_aggregation += grads_for_batch / float(hvd.size())

            aggregations_completed = math.floor(
                (batch_id + 1) / backward_passes_per_step)
            return aggregations_completed * sum_per_aggregation

        @tf.function
        def apply_gradients_in_tf_function(grads_and_vars, **kwargs):
            # Apply gradient updates in tf.function to reproduce how it is
            # done inside `model.fit()`.
            hvd_optimizer.apply_gradients(grads_and_vars, **kwargs)

        var = tf.Variable([0.0])
        variables = [var]
        def loss():
            return (var - var)
        for idx in range(10):
            if _PRE_TF_2_2_0:
                grads_and_vars = hvd_optimizer._compute_gradients(
                    loss, var_list=variables)
                apply_gradients_in_tf_function(grads_and_vars, variables)
            else:
                # In 2.2 and 2.3 the horovod optimizer sets `_HAS_AGGREGATE_GRAD = True`.
                # This configures tf.keras to call `_aggregate_gradients()` outside of
                # `apply_gradients()` and to set `experimental_aggregate_gradients` to
                # False when calling `apply_gradients()` to prevent it from calling
                # `_aggregate_gradients()` again.

                grads_and_vars = hvd_optimizer._compute_gradients(
                    loss, var_list=variables)
                apply_gradients_in_tf_function(
                    grads_and_vars, experimental_aggregate_gradients=False)

            updated_variable_value = variables[0][0].numpy()
            assert updated_variable_value == compute_expected_value(idx)
            assert idx + 1 == hvd_optimizer.iterations.numpy()

    def test_process_set_optimizer(self):
        """ Note that this test makes the most sense when running with > 2 processes. """
        size = hvd.size()

        if size == 1:
            self.skipTest("Only one worker available")

        subset = hvd.add_process_set(range(0, size, 2))

        class TestOptimizer(keras.optimizers.Optimizer):
            def __init__(self, name, **kwargs):
                super(TestOptimizer, self).__init__(name, **kwargs)

            def get_gradients(self, loss, params):
                assert len(params) == 1
                return [tf.constant([float(hvd.rank())])]

            def _create_slots(self, var_list):
                pass

            def _resource_apply_dense(self, grad, var, apply_state):
                return var.assign_add(grad)

            def get_config(self):
                config = super(TestOptimizer, self).get_config()
                return config

        opt = TestOptimizer(name="TestOpti")
        opt = hvd.DistributedOptimizer(opt, process_set=subset)

        variable = tf.Variable([0.0])
        gradient, = opt.get_gradients(None, [variable])
        opt.apply_gradients([(gradient, variable)])
        computed_value = variable.numpy()

        if subset.included():
            self.assertAlmostEqual(
                computed_value, sum(
                    range(
                        0, size, 2)) / subset.size())
        else:
            self.assertAlmostEqual(computed_value, float(hvd.rank()))

        hvd.remove_process_set(subset)
