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

import horovod.spark.common._namedtuple_fix

import copy
import io
import numbers
import time
import os

from pyspark import keyword_only
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.util import MLWritable, MLReadable
from pyspark.sql import SparkSession

from horovod.runner.common.util import codec
from horovod.spark.common import util
from horovod.spark.common.estimator import HorovodEstimator, HorovodModel
from horovod.spark.common.params import EstimatorParams
from horovod.spark.common.serialization import \
    HorovodParamsWriter, HorovodParamsReader
from horovod.spark.torch import remote
from horovod.spark.torch.util import deserialize_fn, serialize_fn, \
    save_into_bio

import numpy as np
import torch
import torch.utils.data


def _torch_param_serialize(param_name, param_val):
    if param_val is None:
        return None

    if param_name in [EstimatorParams.backend.name, EstimatorParams.store.name]:
        # We do not serialize backend and store. These params have to be regenerated for each
        # run of the pipeline
        return None
    elif param_name == EstimatorParams.model.name:
        serialize = serialize_fn()
        return serialize(param_val)

    return codec.dumps_base64(param_val)


class TorchEstimatorParamsWriter(HorovodParamsWriter):
    def saveImpl(self, path):
        # Write the parameters
        HorovodParamsWriter.saveMetadata(self.instance, path, self.sc,
                                         param_serializer_fn=_torch_param_serialize)


class TorchEstimatorParamsWritable(MLWritable):
    def write(self):
        return TorchEstimatorParamsWriter(self)


class TorchEstimatorParamsReader(HorovodParamsReader):
    def _deserialize_dict(self, dict_values):
        deserialized_dict = dict()
        for key, val in dict_values.items():
            if val is None:
                deserialized_dict[key] = None
            elif key == EstimatorParams.model.name:
                deserialize = deserialize_fn()
                deserialized_dict[key] = deserialize(val)
            else:
                deserialized_dict[key] = codec.loads_base64(val)
        return deserialized_dict


class TorchEstimatorParamsReadable(MLReadable):
    @classmethod
    def read(cls):
        """Returns a DefaultParamsReader instance for this class."""
        return TorchEstimatorParamsReader(cls)


class TorchEstimator(HorovodEstimator, TorchEstimatorParamsWritable,
                     TorchEstimatorParamsReadable):
    """Spark Estimator for fitting PyTorch models to a DataFrame.

    Args:
        num_proc: Number of Horovod processes.  Defaults to `spark.default.parallelism`.
        model: PyTorch model to train.
        backend: Optional Backend object for running distributed training function. Defaults to SparkBackend with
                 `num_proc` worker processes. Cannot be specified if `num_proc` is also provided.
        store: Store object that abstracts reading and writing of intermediate data and run results.
        optimizer: PyTorch optimizer to be converted into a `hvd.DistributedOptimizer` for training.
        loss: PyTorch loss or list of losses.
        loss_constructors: Optional functions that generate losses.
        metrics: Optional metrics to record.
        loss_weights: Optional list of float weight values to assign each loss.
        sample_weight_col: Optional column indicating the weight of each sample.
        gradient_compression: Gradient compression used by `hvd.DistributedOptimizer`.
        feature_cols: Column names used as feature inputs to the model. Must be a list with each feature
                      mapping to a sequential argument in the model's forward() function.
        input_shapes: List of shapes for each input tensor to the model.
        validation: Optional validation column name (string) where every row in the column is either 1/True or 0/False,
                    or validation split (float) giving percent of data to be randomly selected for validation.
        label_cols: Column names used as labels.  Must be a list with one label for each output of the model.
        batch_size: Number of rows from the DataFrame per batch.
        val_batch_size: Number of rows from the DataFrame per batch for validation, if not set, will use batch_size.
        epochs: Number of epochs to train.
        verbose: Verbosity level [0, 2] (default: 1).
        random_seed: Optional random seed to use for Torch. Default: None.
        shuffle_buffer_size: Optional size of in-memory shuffle buffer in rows (on training data).
                             Allocating a larger buffer size increases randomness of shuffling at
                             the cost of more host memory. Defaults to estimating with an assumption
                             of 4GB of memory per host. Set shuffle_buffer_size=0 would turn off shuffle.
        partitions_per_process: Number of Parquet partitions to assign per worker process from `num_proc` (default: 10).
        run_id: Optional unique ID for this run for organization in the Store. Will be automatically assigned if not
                provided.
        train_minibatch_fn: Optional custom function to execute within the training loop. Defaults to standard
                            gradient descent process.
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
        inmemory_cache_all: (Optional) Cache the data in memory for training and validation.
        use_gpu: Whether to use the GPU for training. Defaults to True.
        mp_start_method: The method to use to start multiprocessing. Defaults to None.
    """

    input_shapes = Param(Params._dummy(), 'input_shapes', 'input layer shapes')
    loss_constructors = Param(Params._dummy(), 'loss_constructors',
                              'functions that construct the loss')
    train_minibatch_fn = Param(Params._dummy(), 'train_minibatch_fn',
                               'functions that construct the minibatch train function for torch')

    @keyword_only
    def __init__(self,
                 num_proc=None,
                 model=None,
                 backend=None,
                 store=None,
                 optimizer=None,
                 loss=None,
                 loss_constructors=None,
                 metrics=None,
                 loss_weights=None,
                 sample_weight_col=None,
                 gradient_compression=None,
                 feature_cols=None,
                 input_shapes=None,
                 validation=None,
                 label_cols=None,
                 callbacks=None,
                 batch_size=None,
                 val_batch_size=None,
                 epochs=None,
                 verbose=1,
                 random_seed=None,
                 shuffle_buffer_size=None,
                 partitions_per_process=None,
                 run_id=None,
                 train_minibatch_fn=None,
                 train_steps_per_epoch=None,
                 validation_steps_per_epoch=None,
                 transformation_fn=None,
                 train_reader_num_workers=None,
                 val_reader_num_workers=None,
                 reader_pool_type=None,
                 label_shapes=None,
                 inmemory_cache_all=False,
                 use_gpu=True,
                 mp_start_method=None):

        super(TorchEstimator, self).__init__()
        self._setDefault(loss_constructors=None,
                         input_shapes=None,
                         train_minibatch_fn=None,
                         transformation_fn=None)

        kwargs = self._input_kwargs

        if EstimatorParams.loss.name in kwargs and TorchEstimator.loss_constructors.name in kwargs:
            raise ValueError("only one of loss_constructors and loss parameters can be specified.")

        self.setParams(**kwargs)

    def setTrainMinibatchFn(self, value):
        return self._set(train_minibatch_fn=value)

    def getTrainMinibatchFn(self):
        return self.getOrDefault(self.train_minibatch_fn)

    def setInputShapes(self, value):
        return self._set(input_shapes=value)

    def getInputShapes(self):
        return self.getOrDefault(self.input_shapes)

    def setLossConstructors(self, value):
        return self._set(loss_constructors=value)

    def getLossConstructors(self):
        return self.getOrDefault(self.loss_constructors)

    def _get_optimizer(self):
        return self.getOrDefault(self.optimizer)

    # Overwrites Model's getOptimizer method
    def getOptimizer(self):
        model = self.getModel()
        if model:
            optimizer = self._get_optimizer()
            optimizer_cls = optimizer.__class__
            optimizer_state = optimizer.state_dict()
            optimzer = optimizer_cls(model.parameters(), lr=1)
            optimzer.load_state_dict(optimizer_state)
            return optimzer
        else:
            return self._get_optimizer()

    def _check_metadata_compatibility(self, metadata):
        util.check_shape_compatibility(metadata,
                                       self.getFeatureCols(),
                                       self.getLabelCols(),
                                       input_shapes=self.getInputShapes(),
                                       label_shapes=self.getLabelShapes())

    def _fit_on_prepared_data(self, backend, train_rows, val_rows, metadata, avg_row_size, dataset_idx=None):
        self._check_params(metadata)

        run_id = self.getRunId()
        if run_id is None:
            run_id = 'pytorch_' + str(int(time.time()))

        last_checkpoint_state = None
        if self._has_checkpoint(run_id):
            last_checkpoint_state = self._load_checkpoint(run_id)

        # Model parameters
        model_pre_train = self.getModel()
        model_state = model_pre_train.state_dict()
        serialized_model = serialize_fn()(model_pre_train)

        # Optimizer parameters
        optimizer = self._get_optimizer()
        optimizer_cls = optimizer.__class__
        optimizer_state = optimizer.state_dict()

        # Combine model and optimizer state
        model_opt_state = {'model': model_state, 'optimizer': optimizer_state} \
            if last_checkpoint_state is None else last_checkpoint_state
        model_opt_state_serialized = save_into_bio(model_opt_state, torch.save)

        trainer = remote.RemoteTrainer(self, metadata, last_checkpoint_state, run_id, dataset_idx)
        handle = backend.run(trainer,
                             args=(serialized_model, optimizer_cls, model_opt_state_serialized,
                                   train_rows, val_rows, avg_row_size),
                             env={})
        return self._create_model(handle, run_id, metadata)

    def _load_checkpoint(self, run_id):
        store = self.getStore()
        last_ckpt_path = os.path.join(store.get_checkpoint_path(run_id),store.get_checkpoint_filename())
        
        if not store.fs.exists(last_ckpt_path):
            return None

        if self.getVerbose():
            print('Resuming training from last checkpoint: {}'.format(last_ckpt_path))

        ckpt_file = io.BytesIO(store.read(last_ckpt_path))
        return torch.load(ckpt_file)

    def _create_model(self, run_results, run_id, metadata):
        history, serialized_checkpoint = run_results[0]
        serialized_checkpoint.seek(0)
        best_checkpoint = torch.load(serialized_checkpoint, map_location=torch.device('cpu'))

        model = copy.deepcopy(self.getModel())
        optimizer = copy.deepcopy(self.getOptimizer())

        model.load_state_dict(best_checkpoint['model'])
        model.eval()
        optimizer.load_state_dict(best_checkpoint['optimizer'])

        return self.get_model_class()(**self._get_model_kwargs(
            model, history, optimizer, run_id, metadata))

    def get_model_class(self):
        return TorchModel

    def _get_model_kwargs(self, model, history, optimizer, run_id, metadata):
        return dict(history=history,
                    model=model,
                    optimizer=optimizer,
                    feature_columns=self.getFeatureCols(),
                    input_shapes=self.getInputShapes(),
                    label_columns=self.getLabelCols(),
                    run_id=run_id,
                    _metadata=metadata,
                    loss=self.getLoss(),
                    loss_constructors=self.getLossConstructors())


class TorchModel(HorovodModel, TorchEstimatorParamsWritable, TorchEstimatorParamsReadable):
    """Spark Transformer wrapping a PyTorch model, used for making predictions on a DataFrame.

    Retrieve the underlying PyTorch model by calling `torch_model.getModel()`.

    Args:
        history: List of metrics, one entry per epoch during training.
        model: Trained PyTorch model.
        feature_columns: List of feature column names.
        label_columns: List of label column names.
        optimizer: PyTorch optimizer used during training, containing updated state.
        run_id: ID of the run used to train the model.
        loss: PyTorch loss(es).
        loss_constructors: PyTorch loss constructors.
    """

    optimizer = Param(Params._dummy(), 'optimizer', 'optimizer')
    input_shapes = Param(Params._dummy(), 'input_shapes', 'input layer shapes')
    loss = Param(Params._dummy(), 'loss', 'loss')
    loss_constructors = Param(Params._dummy(), 'loss_constructors',
                              'functions that construct the loss')

    @keyword_only
    def __init__(self,
                 history=None,
                 model=None,
                 feature_columns=None,
                 input_shapes=None,
                 label_columns=None,
                 optimizer=None,
                 run_id=None,
                 _metadata=None,
                 loss=None,
                 loss_constructors=None):
        super(TorchModel, self).__init__()

        if label_columns:
            self.setOutputCols([col + '__output' for col in label_columns])

        self._setDefault(optimizer=None,
                         loss=None,
                         loss_constructors=None,
                         input_shapes=None)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def setLoss(self, value):
        return self._set(loss=value)

    def getLoss(self):
        return self.getOrDefault(self.loss)

    def setLossConstructors(self, value):
        return self._set(loss_constructors=value)

    def getLossConstructors(self):
        return self.getOrDefault(self.loss_constructors)

    def setInputShapes(self, value):
        return self._set(input_shapes=value)

    def getInputShapes(self):
        return self.getOrDefault(self.input_shapes)

    def setOptimizer(self, value):
        return self._set(optimizer=value)

    def _get_optimizer(self):
        return self.getOrDefault(self.optimizer)

    def getOptimizer(self):
        model = self.getModel()
        if model:
            _optimizer = self._get_optimizer()
            optimizer_cls = _optimizer.__class__
            optimizer_state = _optimizer.state_dict()
            optimzer = optimizer_cls(model.parameters(), lr=1)
            optimzer.load_state_dict(optimizer_state)
            return optimzer
        else:
            return self._get_optimizer()

    # To run locally on OS X, need export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    def _transform(self, df):
        import copy
        from pyspark.sql.types import StructField, StructType
        from pyspark.ml.linalg import VectorUDT

        model_pre_predict = self.getModel()
        deserialize = deserialize_fn()
        serialize = serialize_fn()
        serialized_model = serialize(model_pre_predict)

        input_shapes = self.getInputShapes()
        label_cols = self.getLabelColumns()
        output_cols = self.getOutputCols()
        feature_cols = self.getFeatureColumns()
        metadata = self._get_metadata()

        final_output_cols = util.get_output_cols(df.schema, output_cols)

        def predict(rows):
            from pyspark import Row
            from pyspark.ml.linalg import DenseVector, SparseVector

            model = deserialize(serialized_model)
            # Perform predictions.
            for row in rows:
                fields = row.asDict().copy()

                # Note: if the col is SparseVector, torch.tensor(col) correctly converts it to a
                # dense torch tensor.
                data = [torch.tensor([row[col]]).reshape(shape) for
                        col, shape in zip(feature_cols, input_shapes)]

                with torch.no_grad():
                    preds = model(*data)

                if not isinstance(preds, list) and not isinstance(preds, tuple):
                    preds = [preds]

                for label_col, output_col, pred in zip(label_cols, output_cols, preds):
                    meta = metadata[label_col]
                    col_type = meta['spark_data_type']
                    # dtype for dense and spark tensor is always np.float64
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
                    elif pred.shape.numel() == 1:
                        # If the column is scalar type, int, float, etc.
                        value = pred.item()
                        python_type = util.spark_scalar_to_python_type(col_type)
                        if issubclass(python_type, numbers.Integral):
                            value = round(value)
                        field = python_type(value)
                    else:
                        field = DenseVector(pred.reshape(-1))

                    fields[output_col] = field

                values = [fields[col] for col in final_output_cols]

                yield Row(*values)

        spark0 = SparkSession._instantiatedSession

        final_output_fields = []

        # copy input schema
        for field in df.schema.fields:
            final_output_fields.append(copy.deepcopy(field))

        # append output schema
        override_fields = df.limit(1).rdd.mapPartitions(predict).toDF().schema.fields[-len(output_cols):]
        for name, override, label in zip(output_cols, override_fields, label_cols):
            # default data type as label type
            data_type = metadata[label]['spark_data_type']()

            if type(override.dataType) == VectorUDT:
                # Override output to vector. This is mainly for torch's classification loss
                # where label is a scalar but model output is a vector.
                data_type = VectorUDT()
            final_output_fields.append(StructField(name=name, dataType=data_type, nullable=True))

        final_output_schema = StructType(final_output_fields)

        pred_rdd = df.rdd.mapPartitions(predict)

        # Use the schema from previous section to construct the final DF with prediction
        return spark0.createDataFrame(pred_rdd, schema=final_output_schema)
