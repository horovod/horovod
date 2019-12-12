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

from __future__ import absolute_import

import horovod.spark.common._namedtuple_fix

import numbers
import time

import numpy as np
import tensorflow as tf

from pyspark import keyword_only
from pyspark.ml import Estimator, Model
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.ml.param.shared import Param, Params

from horovod.run.common.util import codec

from horovod.spark.common import util
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.params import EstimatorParams, ModelParams
from horovod.spark.common.serialization import \
    HorovodParamsWriter, HorovodParamsReader
from horovod.spark.keras import remote
from horovod.spark.keras.util import \
    BARE_KERAS, TF_KERAS, \
    BareKerasUtil, TFKerasUtil, \
    is_instance_of_bare_keras_model, is_instance_of_bare_keras_optimizer


class KerasEstimatorParamsWriter(HorovodParamsWriter):
    def saveImpl(self, path):
        keras_utils = self.instance._get_keras_utils()
        # Write the parameters
        HorovodParamsWriter.saveMetadata(self.instance, path, self.sc,
                                         param_serializer_fn=keras_utils.serialize_param_value)


class KerasEstimatorParamsWritable(MLWritable):
    def write(self):
        return KerasEstimatorParamsWriter(self)


class KerasEstimatorParamsReader(HorovodParamsReader):
    def _deserialize_dict(self, dict):
        def _param_deserializer_fn(name, param_val, keras_utils, custom_objects):
            if param_val is None:
                return param_val

            if name == EstimatorParams.model.name:
                def load_model_fn(x):
                    with keras_utils.keras().utils.custom_object_scope(custom_objects):
                        return keras_utils.keras().models.load_model(x, compile=True)

                return keras_utils.deserialize_model(param_val,
                                                     load_model_fn=load_model_fn)
            elif name == KerasEstimator.optimizer.name:
                opt_base64_encoded = codec.loads_base64(param_val)
                return keras_utils.deserialize_optimizer(opt_base64_encoded)
            else:
                return codec.loads_base64(param_val)

        # In order to deserialize the model, we need to deserialize the custom_objects param
        # first.
        keras_utils = None
        if KerasEstimator._keras_pkg_type.name in dict:
            keras_pkg_type = _param_deserializer_fn(KerasEstimator._keras_pkg_type.name,
                                                    dict[KerasEstimator._keras_pkg_type.name],
                                                    None, None)
            if keras_pkg_type == BARE_KERAS:
                keras_utils = BareKerasUtil
            elif keras_pkg_type == TF_KERAS:
                keras_utils = TFKerasUtil

        custom_objects = {}
        if KerasEstimator.custom_objects.name in dict:
            custom_objects = _param_deserializer_fn(KerasEstimator.custom_objects.name,
                                                    dict[KerasEstimator.custom_objects.name],
                                                    None, None)

        for key, val in dict.items():
            dict[key] = _param_deserializer_fn(key, val, keras_utils, custom_objects)
        return dict


class KerasEstimatorParamsReadable(MLReadable):
    @classmethod
    def read(cls):
        """Returns a KerasEstimatorParamsReader instance for this class."""
        return KerasEstimatorParamsReader(cls)


class KerasEstimator(Estimator, EstimatorParams, KerasEstimatorParamsReadable,
                     KerasEstimatorParamsWritable):
    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')
    _keras_pkg_type = Param(Params._dummy(), '_keras_pkg_type', 'keras package type')

    @keyword_only
    def __init__(self,
                 num_proc=None,
                 model=None,
                 backend=None,
                 store=None,
                 custom_objects=None,
                 optimizer=None,
                 loss=None,
                 loss_weights=None,
                 sample_weight_col=None,
                 compression=None,
                 metrics=None,
                 feature_cols=None,
                 label_cols=None,
                 validation=None,
                 callbacks=None,
                 batch_size=None,
                 epochs=None,
                 verbose=None,
                 shuffle_buffer_size=None,
                 partitions_per_process=None,
                 run_id=None,
                 train_steps_per_epoch=None,
                 validation_steps_per_epoch=None):

        super(KerasEstimator, self).__init__()

        self._setDefault(optimizer=None,
                         custom_objects={},
                         _keras_pkg_type=None)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def _get_keras_utils(self):
        # This function determines the keras package type of the Estimator based on the passed
        # optimizer and model and updates _keras_pkg_type parameter.

        model_type = None
        model = self.getModel()
        if model:
            if isinstance(model, tf.keras.Model):
                model_type = TF_KERAS
            elif is_instance_of_bare_keras_model(model):
                model_type = BARE_KERAS
            else:
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model or keras.Model")

        optimizer_type = None
        optimizer = self.getOptimizer()
        if optimizer:
            if isinstance(optimizer, str):
                optimizer_type = None
            elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
                optimizer_type = TF_KERAS
            elif is_instance_of_bare_keras_optimizer(optimizer):
                optimizer_type = BARE_KERAS
            else:
                raise ValueError("invalid optimizer type")

        types = set([model_type, optimizer_type])
        types.discard(None)

        if len(types) > 1:
            raise ValueError('mixed keras and tf.keras values for optimizers and model')
        elif len(types) == 1:
            pkg_type = types.pop()
            super(KerasEstimator, self)._set(_keras_pkg_type=pkg_type)

            if pkg_type == TF_KERAS:
                return TFKerasUtil
            elif pkg_type == BARE_KERAS:
                return BareKerasUtil
            else:
                raise ValueError("invalid keras type")

    def setCustomObjects(self, value):
        return self._set(custom_objects=value)

    def getCustomObjects(self):
        return self.getOrDefault(self.custom_objects)

    def _check_metadata_compatibility(self, metadata):
        input_shapes, output_shapes = self.get_model_shapes()
        util.check_shape_compatibility(metadata,
                                       self.getFeatureCols(),
                                       self.getLabelCols(),
                                       input_shapes=input_shapes,
                                       output_shapes=output_shapes)

    def get_model_shapes(self):
        model = self.getModel()
        input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                        for input in model.inputs]
        output_shapes = [[dim if dim else -1 for dim in output.shape.as_list()]
                         for output in model.outputs]
        return input_shapes, output_shapes

    def _get_or_create_backend(self):
        backend = self.getBackend()
        if backend is None:
            backend = SparkBackend(self.getNumProc())
        elif self.getNumProc() is not None:
            raise ValueError('At most one of parameters "backend" and "num_proc" may be specified')
        return backend

    def fit_on_parquet(self, params=None):
        if params:
            return self.copy(params)._fit_on_parquet()
        else:
            return self._fit_on_parquet()

    def _fit_on_parquet(self):
        backend = self._get_or_create_backend()
        store = self.getStore()
        label_columns = self.getLabelCols()
        feature_columns = self.getFeatureCols()
        sample_weight_col = self.getSampleWeightCol()

        train_rows, val_rows, metadata, avg_row_size = \
            util.get_simple_meta_from_parquet(store,
                                              label_columns=label_columns,
                                              feature_columns=feature_columns,
                                              sample_weight_col=sample_weight_col)

        return self._fit_on_prepared_data(backend, train_rows, val_rows, metadata, avg_row_size)

    def _fit(self, df):
        backend = self._get_or_create_backend()
        store = self.getStore()
        label_columns = self.getLabelCols()
        feature_columns = self.getFeatureCols()
        validation = self.getValidation()
        sample_weight_col = self.getSampleWeightCol()
        partitions_per_process = self.getPartitionsPerProcess()

        with util.prepare_data(backend.num_processes(),
                               store,
                               df,
                               label_columns=label_columns,
                               feature_columns=feature_columns,
                               validation=validation,
                               sample_weight_col=sample_weight_col,
                               partitions_per_process=partitions_per_process,
                               verbose=self.getVerbose()) as dataset_idx:
            train_rows, val_rows, metadata, avg_row_size = util.get_dataset_properties(dataset_idx)
            self._check_metadata_compatibility(metadata)
            return self._fit_on_prepared_data(
                backend, train_rows, val_rows, metadata, avg_row_size, dataset_idx)

    def _fit_on_prepared_data(self, backend, train_rows, val_rows, metadata, avg_row_size, dataset_idx=None):
        self._check_params(metadata)
        keras_utils = self._get_keras_utils()

        run_id = self.getRunId()
        if run_id is None:
            run_id = 'keras_' + str(int(time.time()))

        if self._has_checkpoint(run_id):
            serialized_model = self._load_model_from_checkpoint(run_id)
        else:
            serialized_model = self._compile_model(keras_utils)

        # Workaround:
        # https://stackoverflow.com/questions/50583056/is-there-any-way-to-set-java-opts-for-tensorflow-process/50615570
        env = {'LIBHDFS_OPTS': '-Xms2048m -Xmx2048m'}

        trainer = remote.RemoteTrainer(self, metadata, keras_utils, run_id, dataset_idx)
        handle = backend.run(trainer,
                             args=(serialized_model, train_rows, val_rows, avg_row_size),
                             env=env)
        return self._create_model(handle, run_id, metadata)

    def _has_checkpoint(self, run_id):
        store = self.getStore()
        last_ckpt_path = store.get_checkpoint_path(run_id)
        return last_ckpt_path is not None and store.exists(last_ckpt_path)

    def _load_model_from_checkpoint(self, run_id):
        store = self.getStore()
        last_ckpt_path = store.get_checkpoint_path(run_id)

        if self.getVerbose():
            print('Resuming training from last checkpoint: {}'.format(last_ckpt_path))

        model_bytes = store.read(last_ckpt_path)
        return codec.dumps_base64(model_bytes)

    def _compile_model(self, keras_utils):
        # Compile the model with all the parameters
        model = self.getModel()

        loss = self.getLoss()
        loss_weights = self.getLossWeights()

        if not loss:
            raise ValueError('Loss parameter is required for the model to compile')

        optimizer = self.getOptimizer()
        if not optimizer:
            optimizer = model.optimizer

        if not optimizer:
            raise ValueError('Optimizer must be provided either as a parameter or as part of a '
                             'compiled model')

        metrics = self.getMetrics()
        compression = self.getCompression()
        optimizer_weight_values = optimizer.get_weights()

        dist_optimizer_args = dict(optimizer=optimizer)
        if compression:
            dist_optimizer_args['compression'] = compression

        # Horovod: wrap optimizer with DistributedOptimizer.
        dist_optimizer = keras_utils.get_horovod().DistributedOptimizer(**dist_optimizer_args)
        model.compile(optimizer=dist_optimizer,
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=metrics)

        if optimizer_weight_values:
            model.optimizer.set_weights(optimizer_weight_values)

        return keras_utils.serialize_model(model)

    def _create_model(self, run_results, run_id, metadata):
        keras_utils = self._get_keras_utils()
        keras_module = keras_utils.keras()
        floatx = keras_module.backend.floatx()

        custom_objects = self.getCustomObjects()

        history, serialized_model, hvd_size = run_results[0]

        def load_model_fn(x):
            with keras_module.utils.custom_object_scope(custom_objects):
                return keras_module.models.load_model(x)

        model = keras_utils.deserialize_model(serialized_model, load_model_fn=load_model_fn)

        # Here, learning rate is scaled down with the number of horovod workers.
        # This is important the retraining of the model. User may retrain the model with
        # different number of workers and we need the raw learning rate to adjust with the
        # new number of workers.
        scaled_lr = keras_module.backend.get_value(model.optimizer.lr)
        keras_module.backend.set_value(model.optimizer.lr, scaled_lr / hvd_size)

        return self.get_model_class()(**self._get_model_kwargs(
            model, history, run_id, metadata, floatx))

    def get_model_class(self):
        return KerasModel

    def _get_model_kwargs(self, model, history, run_id, metadata, floatx):
        return dict(history=history,
                    model=model,
                    feature_columns=self.getFeatureCols(),
                    label_columns=self.getLabelCols(),
                    custom_objects=self.getCustomObjects(),
                    run_id=run_id,
                    _metadata=metadata,
                    _floatx=floatx)


class KerasModel(Model, ModelParams, KerasEstimatorParamsReadable,
                 KerasEstimatorParamsWritable):
    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')

    # Setting _keras_pkg_type parameter helps us determine the type of keras package during
    # deserializing the transformer
    _keras_pkg_type = Param(Params._dummy(), '_keras_pkg_type', 'keras package type')

    _floatx = Param(Params._dummy(), '_floatx', 'keras default float type')

    @keyword_only
    def __init__(self,
                 history=None,
                 model=None,
                 feature_columns=None,
                 label_columns=None,
                 custom_objects=None,
                 run_id=None,
                 _metadata=None,
                 _floatx=None):

        super(KerasModel, self).__init__()

        if label_columns:
            self.setOutputCols([col + '__output' for col in label_columns])

        self._setDefault(custom_objects={})

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setCustomObjects(self, value):
        return self._set(custom_objects=value)

    def getCustomObjects(self):
        return self.getOrDefault(self.custom_objects)

    def _get_keras_utils(self, model=None):
        # infer keras package from model
        model = self.getModel()
        if model:
            if isinstance(model, tf.keras.Model):
                pkg_type = TF_KERAS
            elif is_instance_of_bare_keras_model(model):
                pkg_type = BARE_KERAS
            else:
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model or keras.Model")

            super(KerasModel, self)._set(_keras_pkg_type=pkg_type)

            if pkg_type == TF_KERAS:
                return TFKerasUtil
            elif pkg_type == BARE_KERAS:
                return BareKerasUtil
            else:
                raise ValueError("invalid keras type")

        raise ValueError("model is not set")

    def _get_floatx(self):
        return self.getOrDefault(self._floatx)

    # To run locally on OS X, need export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    def _transform(self, df):
        keras_utils = self._get_keras_utils()
        floatx = self._get_floatx()
        serialized_model = keras_utils.serialize_model(self.getModel())

        label_cols = self.getLabelColumns()
        output_cols = self.getOutputCols()
        feature_cols = self.getFeatureColumns()
        custom_objects = self.getCustomObjects()
        metadata = self._get_metadata()

        pin_cpu = remote._pin_cpu_fn()

        def predict(rows):
            import tensorflow as tf
            from pyspark import Row
            from pyspark.ml.linalg import DenseVector, SparseVector

            k = keras_utils.keras()
            k.backend.set_floatx(floatx)

            # Do not use GPUs for prediction, use single CPU core per task.
            pin_cpu(tf, k)

            def load_model_fn(x):
                with k.utils.custom_object_scope(custom_objects):
                    return k.models.load_model(x)

            model = keras_utils.deserialize_model(serialized_model,
                                                  load_model_fn=load_model_fn)

            input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                            for input in model.inputs]

            def to_array(item):
                if type(item) in [DenseVector or SparseVector]:
                    return item.toArray()
                else:
                    return np.array(item)

            def to_numpy(item):
                # Some versions of TensorFlow will return an EagerTensor
                return item.numpy() if hasattr(item, 'numpy') else item

            # Perform predictions.
            for row in rows:
                fields = row.asDict().copy()
                preds = model.predict_on_batch(
                    [to_array(row[feature_cols[i]]).reshape(input_shapes[i])
                     for i in range(len(feature_cols))])
                preds = [to_numpy(item) for item in preds]

                for label_col, output_col, pred, in zip(label_cols, output_cols, preds):
                    meta = metadata[label_col]
                    col_type = meta['spark_data_type']
                    # dtype for dense and spark tensor is always np.float64
                    if col_type == DenseVector:
                        shape = np.prod(pred.shape())
                        flattened_pred = pred.reshape(shape, )
                        field = DenseVector(flattened_pred)
                    elif col_type == SparseVector:
                        shape = meta['shape']
                        flattened_pred = pred.reshape(shape, )
                        nonzero_indices = flattened_pred.nonzero()[0]
                        field = SparseVector(shape, nonzero_indices,
                                             flattened_pred[nonzero_indices])
                    else:
                        # If the column is scalar type, int, float, etc.
                        value = pred[0]
                        python_type = util.spark_scalar_to_python_type(col_type)
                        if issubclass(python_type, numbers.Integral):
                            value = round(value)
                        field = python_type(value)

                    fields[output_col] = field

                yield Row(**fields)

        return df.rdd.mapPartitions(predict).toDF()
