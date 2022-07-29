# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
# Modifications copyright (C) 2019 Intel Corporation
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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
# =============================================================================

""" Testing code shared between test_tensorflow.py and test_tensorflow_process_sets.py
"""

import os
import tensorflow as tf
from horovod.tensorflow.util import _executing_eagerly
from tensorflow.python.framework import ops
import warnings

if hasattr(tf, 'ConfigProto'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

if hasattr(tf, 'config') and hasattr(tf.config, 'experimental') \
        and hasattr(tf.config.experimental, 'set_memory_growth'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # Specifies the config to use with eager execution. Does not preclude
    # tests from running in the graph mode.
    tf.enable_eager_execution(config=config)

ccl_supported_types = set([tf.uint8, tf.int8, tf.uint16, tf.int16,
                           tf.int32, tf.int64, tf.float32, tf.float64])

class BaseTensorFlowTests(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super(BaseTensorFlowTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        if hasattr(tf, 'contrib') and hasattr(tf.contrib, 'eager'):
            self.tfe = tf.contrib.eager
        else:
            self.tfe = tf

    def evaluate(self, tensors):
        if _executing_eagerly():
            return self._eval_helper(tensors)
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)

    def assign(self, variables, values):
        if _executing_eagerly():
            for var, val in zip(variables, values):
                var.assign(val)
        else:
            sess = ops.get_default_session()
            if sess is None:
                with self.test_session(config=config) as sess:
                    for var, val in zip(variables, values):
                        var.load(val, sess)
            else:
                for var, val in zip(variables, values):
                    var.load(val, sess)

    def random_uniform(self, *args, **kwargs):
        if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(1234)
            return tf.random.uniform(*args, **kwargs)
        else:
            tf.set_random_seed(1234)
            return tf.random_uniform(*args, **kwargs)

    def filter_supported_types(self, types):
        if 'CCL_ROOT' in os.environ:
           types = [t for t in types if t in ccl_supported_types]
        return types
