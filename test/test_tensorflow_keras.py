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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from distutils.version import LooseVersion
if LooseVersion(tf.__version__) >= LooseVersion("1.4.0"):
    from tensorflow import keras
    from tensorflow.python.keras import backend as K
else:
    from tensorflow.contrib import keras
    from tensorflow.contrib.keras import backend as K

import horovod.tensorflow.keras as hvd


class TfKerasTests(tf.test.TestCase):
    """
    Tests for ops in horovod.keras.
    """

    def __init__(self, *args, **kwargs):
        super(TfKerasTests, self).__init__(*args, **kwargs)

    def test_train_model(self):
        hvd.init()

        with self.test_session() as sess:
            K.set_session(sess)

            opt = keras.optimizers.RMSprop(lr=0.0001)
            opt = hvd.DistributedOptimizer(opt)

            model = keras.models.Sequential()
            model.add(keras.layers.Dense(2, input_shape=(3,)))
            model.add(keras.layers.RepeatVector(3))
            model.add(keras.layers.ThresholdedReLU(0.5))
            model.compile(loss=keras.losses.mean_squared_error,
                          optimizer=opt,
                          metrics=[keras.metrics.categorical_accuracy],
                          sample_weight_mode='temporal')

            x = np.random.random((1, 3))
            y = np.random.random((1, 3, 3))

            def generator():
                while 1:
                    yield (x, y)

            # No assertions, we just need to verify that it doesn't hang
            callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
            model.fit_generator(generator(),
                                steps_per_epoch=10,
                                callbacks=callbacks,
                                epochs=0,
                                verbose=0,
                                workers=4,
                                initial_epoch=1)
