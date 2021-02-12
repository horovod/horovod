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
import warnings

from distutils.version import LooseVersion

import pytest

from tensorflow import keras
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import horovod.tensorflow.keras as hvd


_PRE_TF_2_4_0 = LooseVersion(tf.__version__) < LooseVersion("2.4.0")
_PRE_TF_2_2_0 = LooseVersion(tf.__version__) < LooseVersion("2.2.0")


@pytest.mark.skipif(LooseVersion(tf.__version__) < LooseVersion('2.0.0'), reason='TensorFlow v2 tests')
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
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    def test_train_model_lr_schedule(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            0.001 * hvd.size(),
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        opt = tf.keras.optimizers.Adam(lr_schedule)
        opt = hvd.DistributedOptimizer(opt)

        model = keras.models.Sequential()
        model.add(keras.layers.Dense(2, input_shape=(3,)))
        model.add(keras.layers.RepeatVector(3))
        model.add(keras.layers.ThresholdedReLU(0.5))
        model.compile(loss=keras.losses.mean_squared_error,
                      optimizer=opt,
                      metrics=[keras.metrics.categorical_accuracy],
                      experimental_run_tf_function=False)

        x = np.random.random((1, 3))
        y = np.random.random((1, 3, 2))

        # No assertions, we just need to verify that it doesn't hang or error
        callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
        model.fit(x,
                  y,
                  steps_per_epoch=10,
                  callbacks=callbacks,
                  epochs=1)

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
            [np.array([[v,  v], [v, v]], dtype=np.float32),
             np.array([v, v], dtype=np.float32)])

        model2 = tf.keras.Sequential([
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model2.build((2, 2))
        model2.set_weights(
            [np.array([[1.0,  2.0], [3.0, 4.0]], dtype=np.float32),
             np.array([0.0, 0.0], dtype=np.float32)])

        optimizer = tf.optimizers.Adam(0.001 * hvd.size())

        state = hvd.elastic.KerasState(model1, optimizer, batch=20 + hvd.rank(), epoch=10 + hvd.rank())
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

    @pytest.mark.skipif(LooseVersion(tf.__version__) >= LooseVersion('2.4.0'),
                        reason='TensorFlow 2.4.0+ does not support this path')
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
                for rank in range(hvd.size()):
                    grads_for_batch += rank

                # Apply `average_aggregated_gradients`.
                grads_for_batch /= float(backward_passes_per_step)

                # Averages across workers.
                sum_per_aggregation += grads_for_batch / float(hvd.size())

            aggregations_completed = math.floor((batch_id + 1) / backward_passes_per_step)
            return aggregations_completed * sum_per_aggregation

        @tf.function
        def apply_gradients_in_tf_function(gradient_updates, model_variables, **kwargs):
            # Apply gradient updates in tf.function to reproduce how it is
            # done inside `model.fit()`.
            hvd_optimizer.apply_gradients(zip(gradient_updates, model_variables), **kwargs)

        gradients = [tf.constant([float(hvd.rank())])]
        variables = [tf.Variable([0.0])]
        for idx in range(10):
            if _PRE_TF_2_2_0:
                updated_gradients = hvd_optimizer._allreduce(gradients, variables)
                apply_gradients_in_tf_function(updated_gradients, variables)
            elif _PRE_TF_2_4_0:
                # In 2.2 and 2.3 the horovod optimizer sets `_HAS_AGGREGATE_GRAD = True`.
                # This configures tf.keras to call `_aggregate_gradients()` outside of
                # `apply_gradients()` and to set `experimental_aggregate_gradients` to
                # False when calling `apply_gradients()` to prevent it from calling
                # `_aggregate_gradients()` again.
                updated_gradients = hvd_optimizer._aggregate_gradients(
                    zip(gradients, variables))
                apply_gradients_in_tf_function(
                    updated_gradients, variables,
                    experimental_aggregate_gradients=False
                )
            else:
                raise RuntimeError("This test should be skipped ...")

            updated_variable_value = variables[0][0].numpy()
            assert updated_variable_value == compute_expected_value(idx)
            assert idx + 1 == hvd_optimizer.iterations.numpy()
