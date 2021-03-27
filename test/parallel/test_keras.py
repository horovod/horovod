# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

"""Tests for horovod.keras."""

from distutils.version import LooseVersion
import os
import sys

try:
    import keras
    from keras import backend as K
    import horovod.keras as hvd
    _HAS_KERAS = True
except ImportError:
    _HAS_KERAS = False

import numpy as np
import pytest
import tensorflow as tf
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import temppath


@pytest.mark.skipif(not _HAS_KERAS, reason='Keras not available')
@pytest.mark.skipif(LooseVersion(tf.__version__) >= LooseVersion('2.0.0'), reason='TensorFlow v1 tests')
class KerasTests(tf.test.TestCase):
    """
    Tests for ops in horovod.keras.
    """

    def __init__(self, *args, **kwargs):
        super(KerasTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        hvd.init()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.visible_device_list = str(hvd.local_rank())

    def test_sparse_as_dense(self):
        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            opt = keras.optimizers.RMSprop(lr=0.0001)
            opt = hvd.DistributedOptimizer(opt, sparse_as_dense=True)

            model = keras.models.Sequential()
            model.add(keras.layers.Embedding(1000, 64, input_length=10))
            model.compile(loss=keras.losses.MSE,
                          optimizer=opt)

            x = np.random.randint(1000, size=(32, 10))
            y = np.random.random((32, 10, 64))
            # No assertions, we just need to verify that it doesn't hang
            model.train_on_batch(x, y)

    def test_load_model(self):
        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            opt = keras.optimizers.RMSprop(lr=0.0001)
            opt = hvd.DistributedOptimizer(opt)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(2, input_shape=(3,)))
            model.add(keras.layers.RepeatVector(3))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
            model.compile(loss=keras.losses.MSE,
                          optimizer=opt,
                          metrics=[keras.metrics.categorical_accuracy],
                          sample_weight_mode='temporal')

            x = np.random.random((1, 3))
            y = np.random.random((1, 3, 3))
            model.train_on_batch(x, y)

            with temppath() as fname:
                model.save(fname)

                new_model = hvd.load_model(fname)
                new_opt = new_model.optimizer

            self.assertEqual(type(new_opt).__module__, 'horovod._keras')
            self.assertEqual(type(new_opt).__name__, 'RMSprop')
            self.assertEqual(K.get_value(opt.lr), K.get_value(new_opt.lr))
            self._check_optimizer_weights(opt, new_opt)

    def test_load_model_custom_optimizers(self):
        class TestOptimizer(keras.optimizers.RMSprop):
            def __init__(self, **kwargs):
                super(TestOptimizer, self).__init__(**kwargs)

        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            opt = TestOptimizer(lr=0.0001)
            opt = hvd.DistributedOptimizer(opt)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(2, input_shape=(3,)))
            model.add(keras.layers.RepeatVector(3))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
            model.compile(loss=keras.losses.MSE,
                          optimizer=opt,
                          metrics=[keras.metrics.categorical_accuracy],
                          sample_weight_mode='temporal')

            x = np.random.random((1, 3))
            y = np.random.random((1, 3, 3))
            model.train_on_batch(x, y)

            with temppath() as fname:
                model.save(fname)

                custom_optimizers = [TestOptimizer]
                new_model = hvd.load_model(fname, custom_optimizers=custom_optimizers)
                new_opt = new_model.optimizer

            self.assertEqual(type(new_opt).__module__, 'horovod._keras')
            self.assertEqual(type(new_opt).__name__, 'TestOptimizer')
            self._check_optimizer_weights(opt, new_opt)

    def test_load_model_custom_objects(self):
        class TestOptimizer(keras.optimizers.RMSprop):
            def __init__(self, **kwargs):
                super(TestOptimizer, self).__init__(**kwargs)

        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            opt = TestOptimizer(lr=0.0001)
            opt = hvd.DistributedOptimizer(opt)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(2, input_shape=(3,)))
            model.add(keras.layers.RepeatVector(3))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
            model.compile(loss=keras.losses.MSE,
                          optimizer=opt,
                          metrics=[keras.metrics.categorical_accuracy],
                          sample_weight_mode='temporal')

            x = np.random.random((1, 3))
            y = np.random.random((1, 3, 3))
            model.train_on_batch(x, y)

            with temppath() as fname:
                model.save(fname)

                custom_objects = {
                    'TestOptimizer': lambda **kwargs: hvd.DistributedOptimizer(
                        TestOptimizer(**kwargs))
                }
                new_model = hvd.load_model(fname, custom_objects=custom_objects)
                new_opt = new_model.optimizer

            self.assertEqual(type(new_opt).__module__, 'horovod._keras')
            self.assertEqual(type(new_opt).__name__, 'TestOptimizer')
            self.assertEqual(K.get_value(opt.lr), K.get_value(new_opt.lr))
            self._check_optimizer_weights(opt, new_opt)

    def test_load_model_broadcast(self):
        def create_model():
            opt = keras.optimizers.SGD(lr=0.01 * hvd.size(), momentum=0.9)
            opt = hvd.DistributedOptimizer(opt)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(2, input_shape=(3,)))
            model.add(keras.layers.RepeatVector(3))
            model.add(keras.layers.TimeDistributed(keras.layers.Dense(3)))
            model.compile(loss=keras.losses.MSE,
                          optimizer=opt,
                          metrics=[keras.metrics.categorical_accuracy],
                          sample_weight_mode='temporal')

            return model

        with temppath() as fname:
            with self.test_session(config=self.config) as sess:
                K.set_session(sess)

                model = create_model()

                x = np.random.random((1, 3))
                y = np.random.random((1, 3, 3))
                model.train_on_batch(x, y)

                if hvd.rank() == 0:
                    model.save(fname)

            K.clear_session()
            with self.test_session(config=self.config) as sess:
                K.set_session(sess)

                if hvd.rank() == 0:
                    model = hvd.load_model(fname)
                else:
                    model = create_model()

                def generator():
                    while 1:
                        yield (x, y)

                if hvd.rank() == 0:
                    self.assertEqual(len(model.optimizer.weights), 5)
                else:
                    self.assertEqual(len(model.optimizer.weights), 0)

                # No assertions, we just need to verify that it doesn't hang
                callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
                model.fit_generator(generator(),
                                    steps_per_epoch=10,
                                    callbacks=callbacks,
                                    epochs=0,
                                    verbose=0,
                                    workers=4,
                                    initial_epoch=1)

                self.assertEqual(len(model.optimizer.weights), 5)

    def _check_optimizer_weights(self, opt, new_opt):
        self.assertEqual(len(opt.get_weights()), len(new_opt.get_weights()))
        for weights, new_weights in zip(opt.get_weights(),
                                        new_opt.get_weights()):
            if np.isscalar(weights):
                self.assertEqual(weights, new_weights)
            else:
                self.assertListEqual(weights.tolist(), new_weights.tolist())

    def test_from_config(self):
        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            opt = keras.optimizers.Adam()
            hopt = hvd.DistributedOptimizer(opt)
            cfg = hopt.get_config()

            hopt_copy1 = hopt.from_config(cfg)
            self.assertEqual(cfg, hopt_copy1.get_config())

            hopt_copy2 = hopt.__class__.from_config(cfg)
            self.assertEqual(cfg, hopt_copy2.get_config())

    def test_elastic_state(self):
        with self.test_session(config=self.config) as sess:
            K.set_session(sess)

            v = 1.0 if hvd.rank() == 0 else 2.0
            model1 = keras.models.Sequential([
                keras.layers.Dense(2, activation='softmax')
            ])
            model1.build((2, 2))
            model1.set_weights(
                [np.array([[v,  v], [v, v]], dtype=np.float32),
                 np.array([v, v], dtype=np.float32)])

            model2 = keras.models.Sequential([
                keras.layers.Dense(2, activation='softmax')
            ])
            model2.build((2, 2))
            model2.set_weights(
                [np.array([[1.0,  2.0], [3.0, 4.0]], dtype=np.float32),
                 np.array([0.0, 0.0], dtype=np.float32)])

            optimizer = keras.optimizers.Adam(0.001 * hvd.size())

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
