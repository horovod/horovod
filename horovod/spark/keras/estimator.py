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
import numbers
import time
import os
import numpy as np
import tensorflow as tf

from pyspark import keyword_only
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.sql import SparkSession

from horovod.runner.common.util import codec

from horovod.spark.common import util
from horovod.spark.common.estimator import HorovodEstimator, HorovodModel
from horovod.spark.common.params import EstimatorParams
from horovod.spark.common.serialization import HorovodParamsWriter, HorovodParamsReader
from horovod.spark.keras import remote
from horovod.spark.keras.util import TFKerasUtil


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
        custom_objects = {}
        if KerasEstimator.custom_objects.name in dict:
            custom_objects = _param_deserializer_fn(KerasEstimator.custom_objects.name,
                                                    dict[KerasEstimator.custom_objects.name],
                                                    None, None)

        for key, val in dict.items():
            dict[key] = _param_deserializer_fn(key, val, TFKerasUtil, custom_objects)
        return dict


class KerasEstimatorParamsReadable(MLReadable):
    @classmethod
    def read(cls):
        """Returns a KerasEstimatorParamsReader instance for this class."""
        return KerasEstimatorParamsReader(cls)


class KerasEstimator(HorovodEstimator, KerasEstimatorParamsReadable,
                     KerasEstimatorParamsWritable):
    """Spark Estimator for fitting Keras models to a DataFrame.

    Supports standalone `keras` and `tf.keras`, and TensorFlow 1.X and 2.X.

    Args:
        num_proc: Number of Horovod processes.  Defaults to `spark.default.parallelism`.
        model: Keras model to train.
        backend: Optional Backend object for running distributed training function. Defaults to SparkBackend with
                 `num_proc` worker processes. Cannot be specified if `num_proc` is also provided.
        store: Store object that abstracts reading and writing of intermediate data and run results.
        custom_objects: Optional dictionary mapping names (strings) to custom classes or functions to be considered
                        during serialization/deserialization.
        optimizer: Keras optimizer to be converted into a `hvd.DistributedOptimizer` for training.
        loss: Keras loss or list of losses.
        loss_weights: Optional list of float weight values to assign each loss.
        sample_weight_col: Optional column indicating the weight of each sample.
        gradient_compression: Gradient compression used by `hvd.DistributedOptimizer`.
        metrics: Optional metrics to record.
        feature_cols: Column names used as feature inputs to the model. Must be a list with each feature
                      mapping to a sequential argument in the model's forward() function.
        label_cols: Column names used as labels.  Must be a list with one label for each output of the model.
        validation: Optional validation column name (string) where every row in the column is either 1/True or 0/False,
                    or validation split (float) giving percent of data to be randomly selected for validation.
        callbacks: Keras callbacks.
        batch_size: Number of rows from the DataFrame per batch.
        val_batch_size: Number of rows from the DataFrame per batch for validation, if not set, will use batch_size.
        epochs: Number of epochs to train.
        verbose: Verbosity level [0, 2] (default: 1).
        random_seed: Optional random seed to use for Tensorflow. Default: None.
        shuffle_buffer_size: Optional size of in-memory shuffle buffer in rows (on training data).
                             Allocating a larger buffer size increases randomness of shuffling at
                             the cost of more host memory. Defaults to estimating with an assumption
                             of 4GB of memory per host. Set shuffle_buffer_size=0 would turn off shuffle.
        partitions_per_process: Number of Parquet partitions to assign per worker process from `num_proc` (default: 10).
        run_id: Optional unique ID for this run for organization in the Store. Will be automatically assigned if not
                provided.
        train_steps_per_epoch: Number of steps to train each epoch. Useful for testing that model trains successfully.
                               Defaults to training the entire dataset each epoch.
        validation_steps_per_epoch: Number of validation steps to perform each epoch.
        transformation_fn: Optional function that takes a row as its parameter
                           and returns a modified row that is then fed into the
                           train or validation step. This transformation is
                           applied after batching. See Petastorm [TransformSpec](https://github.com/uber/petastorm/blob/master/petastorm/transform.py)
                           for more details. Note that this fucntion constructs
                           another function which should perform the
                           transformation.
        train_reader_num_workers: This parameter specifies the number of parallel processes that
                               read the training data from data store and apply data
                               transformations to it. Increasing this number
                               will generally increase the reading rate but will also
                               increase the memory footprint. More processes are
                               particularly useful if the bandwidth to the data store is not
                               high enough, or users need to apply transformation such as
                               decompression or data augmentation on raw data.
        val_reader_num_workers: Similar to the train_reader_num_workers.
        reader_pool_type: Type of worker pool used to parallelize reading data from the dataset.
                          Should be one of ['thread', 'process']. Defaults to 'process'.
        inmemory_cache_all: boolean value. Cache the data in memory for training and validation. Default: False.
        backend_env: dict to add to the environment of the backend.  Defaults to setting the java heap size to
                     2G min and max for libhdfs through petastorm
        use_gpu: Whether to use the GPU for training. Defaults to True.
        mp_start_method: The method to use to start multiprocessing. Defaults to None.
    """

    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')
    checkpoint_callback = Param(Params._dummy(), 'checkpoint_callback',
                                'model checkpointing callback')
    backend_env = Param(Params._dummy(), "backend_env",
                        "dict to add to the environment of the command run on the environment")

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
                 gradient_compression=None,
                 metrics=None,
                 feature_cols=None,
                 label_cols=None,
                 validation=None,
                 callbacks=None,
                 batch_size=None,
                 val_batch_size=None,
                 epochs=None,
                 verbose=None,
                 random_seed=None,
                 shuffle_buffer_size=None,
                 partitions_per_process=None,
                 run_id=None,
                 train_steps_per_epoch=None,
                 validation_steps_per_epoch=None,
                 transformation_fn=None,
                 train_reader_num_workers=None,
                 val_reader_num_workers=None,
                 reader_pool_type=None,
                 label_shapes=None,
                 checkpoint_callback=None,
                 inmemory_cache_all=False,
                 backend_env=None,
                 use_gpu=True,
                 mp_start_method=None):

        super(KerasEstimator, self).__init__()

        self._setDefault(optimizer=None,
                         custom_objects={},
                         checkpoint_callback=None,
                         backend_env={'LIBHDFS_OPTS': '-Xms2048m -Xmx2048m'})

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def _get_keras_utils(self):
        # This function checks the keras package type is tensorflow.keras

        model = self.getModel()
        if model:
            if not isinstance(model, tf.keras.Model):
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model")

        optimizer = self.getOptimizer()
        if optimizer:
            if isinstance(optimizer, str):
                pass
            elif not isinstance(optimizer, tf.keras.optimizers.Optimizer):
                raise ValueError("optimizer has to be an instance of tensorflow.keras.optimizers.Optimizer")

        return TFKerasUtil

    def setCustomObjects(self, value):
        return self._set(custom_objects=value)

    def getCustomObjects(self):
        return self.getOrDefault(self.custom_objects)

    def setCheckpointCallback(self, value):
        return self._set(checkpoint_callback=value)

    def getCheckpointCallback(self):
        return self.getOrDefault(self.checkpoint_callback)

    def setBackendEnv(self, value):
        self._set(backend_env=value)

    def getBackendEnv(self):
        return self.getOrDefault(self.backend_env)

    def _check_metadata_compatibility(self, metadata):
        input_shapes, output_shapes = self.get_model_shapes()
        util.check_shape_compatibility(metadata,
                                       self.getFeatureCols(),
                                       self.getLabelCols(),
                                       input_shapes=input_shapes,
                                       output_shapes=output_shapes,
                                       label_shapes=self.getLabelShapes())

    def get_model_shapes(self):
        model = self.getModel()
        input_shapes = [[dim if dim else -1 for dim in input.shape.as_list()]
                        for input in model.inputs]
        output_shapes = [[dim if dim else -1 for dim in output.shape.as_list()]
                         for output in model.outputs]
        return input_shapes, output_shapes

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

        trainer = remote.RemoteTrainer(self, metadata, keras_utils, run_id, dataset_idx)
        handle = backend.run(trainer,
                             args=(serialized_model, train_rows, val_rows, avg_row_size),
                             env=self.getBackendEnv())
        return self._create_model(handle, run_id, metadata)

    def _load_model_from_checkpoint(self, run_id):
        store = self.getStore()
        last_ckpt_path = os.path.join(store.get_checkpoint_path(run_id), store.get_checkpoint_filename())

        if not store.fs.exists(last_ckpt_path):
            return None

        if self.getVerbose():
            print('Resuming training from last checkpoint: {}'.format(last_ckpt_path))

        return store.read_serialized_keras_model(
            last_ckpt_path, self.getModel(), self.getCustomObjects())

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
        gradient_compression = self.getGradientCompression()
        optimizer_weight_values = optimizer.get_weights()

        dist_optimizer_args = dict(optimizer=optimizer)
        if gradient_compression:
            dist_optimizer_args['compression'] = gradient_compression

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


class KerasModel(HorovodModel, KerasEstimatorParamsReadable,
                 KerasEstimatorParamsWritable):
    """Spark Transformer wrapping a Keras model, used for making predictions on a DataFrame.

    Retrieve the underlying Keras model by calling `keras_model.getModel()`.

    Args:
        history: List of metrics, one entry per epoch during training.
        model: Trained Keras model.
        feature_columns: List of feature column names.
        label_columns: List of label column names.
        custom_objects: Keras custom objects.
        run_id: ID of the run used to train the model.
    """

    custom_objects = Param(Params._dummy(), 'custom_objects', 'custom objects')
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
            if not isinstance(model, tf.keras.Model):
                raise ValueError(
                    "model has to be an instance of tensorflow.keras.Model")

            return TFKerasUtil

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

        final_output_schema = util.get_spark_df_output_schema(df.schema, label_cols, output_cols, metadata)
        final_output_cols = [field.name for field in final_output_schema.fields]

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
                    # dtype for DenseVector and SparseVector is always np.float64
                    if col_type == DenseVector:
                        shape = np.prod(pred.shape)
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

                values = [fields[col] for col in final_output_cols]

                yield Row(*values)

        spark0 = SparkSession._instantiatedSession

        pred_rdd = df.rdd.mapPartitions(predict)

        # Use the schema from previous section to construct the final DF with prediction
        return spark0.createDataFrame(pred_rdd, schema=final_output_schema)
