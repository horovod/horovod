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

from tensorflow import keras
from tensorflow.python.keras import backend as K

import numpy as np
import tensorflow as tf

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

            num_classes = 10

            x = np.random.random((5, 10, 10))
            y = np.random.random((5, 10))

            x = x.astype(np.float32) / 255
            x = np.expand_dims(x, -1)
            y = tf.one_hot(y, num_classes)

            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            dataset = dataset.repeat()
            dataset = dataset.shuffle(1)
            dataset = dataset.batch(1)
            iterator = dataset.make_one_shot_iterator()

            inputs, targets = iterator.get_next()
            model_input = keras.layers.Input(tensor=inputs)

            x = keras.layers.Conv2D(32, (3, 3),
                                    activation='relu', padding='valid')(model_input)
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(512, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)
            model_output = keras.layers.Dense(num_classes,
                                              activation='softmax',
                                              name='x_train_out')(x)

            train_model = keras.models.Model(inputs=model_input, outputs=model_output)

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

            train_model.compile(optimizer=opt,
                                loss='categorical_crossentropy',
                                metrics=['accuracy'],
                                target_tensors=[targets])

            # No assertions, we just need to verify that it doesn't raise an Exception
            callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
            train_model.fit(epochs=1,
                            steps_per_epoch=10,
                            callbacks=callbacks)
