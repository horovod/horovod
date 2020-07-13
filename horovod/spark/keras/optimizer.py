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

import io

import h5py

from horovod.runner.common.util import codec


def serialize_bare_keras_optimizer(x):
    import keras
    from horovod.spark.keras.bare import save_bare_keras_optimizer
    return _serialize_keras_optimizer(x,
                                      optimizer_class=keras.optimizers.Optimizer,
                                      save_optimizer_fn=save_bare_keras_optimizer)


def deserialize_bare_keras_optimizer(x):
    from horovod.spark.keras.bare import load_bare_keras_optimizer
    return _deserialize_keras_optimizer(x,
                                        load_keras_optimizer_fn=load_bare_keras_optimizer)


def serialize_tf_keras_optimizer(x):
    import tensorflow as tf
    from horovod.spark.keras.tensorflow import save_tf_keras_optimizer

    return _serialize_keras_optimizer(x,
                                      optimizer_class=tf.keras.optimizers.Optimizer,
                                      save_optimizer_fn=save_tf_keras_optimizer)


def deserialize_tf_keras_optimizer(x):
    from horovod.spark.keras.tensorflow import load_tf_keras_optimizer

    return _deserialize_keras_optimizer(x,
                                        load_keras_optimizer_fn=load_tf_keras_optimizer)


def _serialize_keras_optimizer(opt, optimizer_class, save_optimizer_fn):
    if isinstance(opt, str):
        return opt
    elif isinstance(opt, optimizer_class):
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as f:
            save_optimizer_fn(opt, f)
        return codec.dumps_base64(bio.getvalue())
    else:
        raise \
            ValueError('Keras optimizer has to be an instance of str or keras.optimizers.Optimizer')


def is_string(obj):
    return isinstance(obj, str)


def _deserialize_keras_optimizer(serialized_opt, load_keras_optimizer_fn):
    if is_string(serialized_opt):
        return serialized_opt
    bio = io.BytesIO(serialized_opt)
    with h5py.File(bio, 'r') as f:
        return load_keras_optimizer_fn(f)
