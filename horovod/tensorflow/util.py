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

import tensorflow as tf
from packaging import version

from tensorflow.python.eager import context

_POST_TF_2_16_1 = version.parse(tf.__version__) >= version.parse('2.16.1')


def _executing_eagerly():
    """Returns true if eager execution is supported and enabled."""
    return context.executing_eagerly()


def _make_subgraph(f):
    return tf.function(f)


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


def _get_var_ref(var):
    """Get a reference to a variable that can be used as a dict key or in a set.

    In TensorFlow < 2.16.1, variables have a .ref() method that returns a reference.
    In TensorFlow >= 2.16.1, variables are directly hashable.
    """
    if _POST_TF_2_16_1:
        return var
    else:
        return var.ref()


def _deref_var(ref):
    """Convert a variable reference back to a variable.

    In TensorFlow < 2.16.1, references have a .deref() method.
    In TensorFlow >= 2.16.1, references are the variables themselves.
    """
    if _POST_TF_2_16_1:
        return ref
    else:
        return ref.deref()


def vars_to_refs(vars):
    if isinstance(vars, list):
        return tuple(vars_to_refs(v) for v in vars)
    return _get_var_ref(vars)


def refs_to_vars(refs):
    if isinstance(refs, tuple):
        return [refs_to_vars(r) for r in refs]
    return _deref_var(refs)
