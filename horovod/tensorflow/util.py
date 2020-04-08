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
from distutils.version import LooseVersion

import tensorflow as tf


if LooseVersion(tf.__version__) >= LooseVersion('1.7.0'):  # Eager Mode has been introduced in TF 1.7.0
    from tensorflow.python.eager import context
    _has_eager = True
else:
    _has_eager = False


def _executing_eagerly():
    """Returns true if eager execution is supported and enabled."""
    return _has_eager and context.executing_eagerly()


def _make_subgraph(f):
    if hasattr(tf, 'function'):
        # TensorFlow 1.14.0+
        return tf.function(f)
    return tf.contrib.eager.defun(f)


def _cache(f):
    cache = dict()

    def wrapper(*args):
        key = (args, _executing_eagerly())

        if key in cache:
            return cache[key]
        else:
            retval = f(*args)
            cache[key] = retval
            return retval

    return wrapper
