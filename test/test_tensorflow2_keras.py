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

"""Tests for horovod.tensorflow.keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import warnings

from tensorflow import keras

import horovod.tensorflow.keras as hvd


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
