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

import json

import tensorflow as tf

from horovod.common.util  import is_version_greater_equal_than

if is_version_greater_equal_than(tf.__version__, "2.6.0"):
    from keras import backend as K
    from keras import optimizers
else:
    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras import optimizers

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import serialization


def save_tf_keras_optimizer(optimizer, h5py_file):
    if isinstance(optimizer, optimizers.TFOptimizer):
        logging.warning(
            'TensorFlow optimizers do not '
            'make it possible to access '
            'optimizer attributes or optimizer state '
            'after instantiation. '
            'As a result, we cannot save the optimizer '
            'as part of the model save file.'
            'You will have to compile your model again after loading it. '
            'Prefer using a Keras optimizer instead '
            '(see keras.io/optimizers).')
    else:
        h5py_file.attrs['training_config'] = json.dumps(
            {
                'optimizer_config': {
                    'class_name': optimizer.__class__.__name__,
                    'config': optimizer.get_config()
                }
            },
            default=serialization.get_json_type).encode('utf8')

        # Save optimizer weights.
        symbolic_weights = getattr(optimizer, 'weights')
        if symbolic_weights:
            optimizer_weights_group = h5py_file.create_group('optimizer_weights')
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for w, val in zip(symbolic_weights, weight_values):
                name = str(w.name)
                weight_names.append(name.encode('utf8'))
            optimizer_weights_group.attrs['weight_names'] = weight_names
            for name, val in zip(weight_names, weight_values):
                param_dset = optimizer_weights_group.create_dataset(
                    name, val.shape, dtype=val.dtype)
                if not val.shape:
                    # scalar
                    param_dset[()] = val
                else:
                    param_dset[:] = val
    h5py_file.flush()


def load_tf_keras_optimizer(h5py_file, custom_objects=None):
    if not custom_objects:
        custom_objects = {}

    def convert_custom_objects(obj):
        """Handles custom object lookup.

        Arguments:
            obj: object, dict, or list.

        Returns:
            The same structure, where occurrences
                of a custom object name have been replaced
                with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    optimizer, optimizer_weight_values = None, None

    # instantiate optimizer
    training_config = h5py_file.attrs.get('training_config')
    training_config = json.loads(training_config.decode('utf-8'))
    optimizer_config = training_config['optimizer_config']
    optimizer = optimizers.deserialize(optimizer_config, custom_objects=custom_objects)

    if 'optimizer_weights' in h5py_file:
        optimizer_weights_group = h5py_file['optimizer_weights']
        optimizer_weight_names = [
            n.decode('utf8')
            for n in optimizer_weights_group.attrs['weight_names']
        ]
        optimizer_weight_values = [optimizer_weights_group[n].value for n in
                                   optimizer_weight_names]
    if optimizer_weight_values:
        optimizer.set_weights(optimizer_weight_values)
    return optimizer
