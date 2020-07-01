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

from distutils.version import LooseVersion

import h5py
import tensorflow as tf
import tensorflow.keras as keras

from horovod.run.common.util import codec

from horovod.spark.common import params
from horovod.spark.keras import optimizer, remote


def serialize_optimizer(*args, **kwargs):
    return optimizer.serialize_tf_keras_optimizer(*args, **kwargs)


def deserialize_optimizer(*args, **kwargs):
    return optimizer.deserialize_tf_keras_optimizer(*args, **kwargs)


def serialize_model(*args, **kwargs):
    def serialize_keras_model(x):
        return _serialize_keras_model(x, keras.models.save_model)

    return serialize_keras_model(*args, **kwargs)


def deserialize_model(*args, **kwargs):
    return _deserialize_keras_model(*args, **kwargs)


def serialize_param_value(*args, **kwargs):
    def _serialize_param(x, y):
        return _serialize_param_value(x, y,
                                      serialize_model_fn=serialize_model,
                                      serialize_opt_fn=serialize_optimizer)

    return _serialize_param(*args, **kwargs)


def _serialize_keras_model(model, save_model_fn):
    """Serialize model into byte array encoded into base 64."""
    bio = io.BytesIO()
    with h5py.File(bio, 'w') as f:
        save_model_fn(model, f)
    return codec.dumps_base64(bio.getvalue())


def _deserialize_keras_model(model_bytes, load_model_fn):
    deserialize_keras_model = remote._deserialize_keras_model_fn()
    return deserialize_keras_model(model_bytes, load_model_fn)


def _deserialize_keras_model_fn():
    def deserialize_keras_model(model_bytes, load_model_fn):
        """Deserialize model from byte array encoded in base 64."""
        model_bytes = codec.loads_base64(model_bytes)
        bio = io.BytesIO(model_bytes)
        with h5py.File(bio, 'r') as f:
            return load_model_fn(f)
    return deserialize_keras_model


def _serialize_param_value(param_name, param_val, serialize_model_fn, serialize_opt_fn):
    if param_val is None:
        return param_val

    if param_name in [params.EstimatorParams.backend.name, params.EstimatorParams.store.name]:
        # We do not serialize backend and store. These params have to be regenerated for each
        # run of the pipeline
        return None
    elif param_name == params.EstimatorParams.model.name:
        return serialize_model_fn(param_val)
    if param_name == params.EstimatorParams.optimizer.name:
        return serialize_opt_fn(param_val)
    else:
        return codec.dumps_base64(param_val)


def custom_sparse_to_dense_fn():
    # TODO(fardin): ask petastorm team about codecs for sparse and dense vectors and see if that is
    # a better solution
    def custom_sparse_to_dense(custom_sparse_vec, dense_shape):
        # original sparse vector:   v = {1:2.0, 3:.4.5, 5:7.1}
        # custom sparse vector:     v = [3, 1, 3, 5, 2.0, 4.5, 7.1]
        # dense vector:             v = [0, 2.0, 0, 4.5, 0, 7.1]

        # Get the first element from custom_sparse_vec. This element is the size of
        # non-zero elements in the original sparse vector.
        sparse_vector_size = tf.cast(tf.gather(custom_sparse_vec, 0, axis=0), tf.int32)
        sparse_vector_size = tf.reshape(sparse_vector_size, [1])

        # get the first sparse_vector_size elements of the custom_sparse_vec which are the
        # indices
        indices_1d = tf.cast(
            tf.slice(custom_sparse_vec, begin=tf.constant([1]), size=sparse_vector_size),
            tf.int64)
        indices_reshaped = tf.reshape(indices_1d,
                                      tf.concat([sparse_vector_size, tf.constant([1])], 0))
        # have to pad the indices to match the expected format by the SparseTensor
        indices = tf.pad(indices_reshaped, [[0, 0], [1, 0]], "CONSTANT")

        # get the second sparse_vector_size elements of the custom_sparse_vec which are
        # the values
        begin_index = sparse_vector_size + tf.constant(1)
        values = tf.slice(custom_sparse_vec, begin=begin_index, size=sparse_vector_size)

        # construct a sparse vector with the indices and values
        dense_shape = [1, dense_shape]
        sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values,
                                               dense_shape=dense_shape)
        # convert the sparse vector into a dense vector
        return tf.sparse.to_dense(sparse_tensor)

    return custom_sparse_to_dense
