# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
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

from pyspark import keyword_only
from pyspark.ml.param.shared import HasOutputCols, Param, Params, TypeConverters

from horovod.spark.common import util


class EstimatorParams(Params):
    num_proc = Param(Params._dummy(), 'num_proc', 'number of processes')
    train_reader_num_workers = Param(Params._dummy(),
                                     'train_reader_num_workers',
                                     'number of parallel worker processes to read train data')
    val_reader_num_workers = Param(Params._dummy(), 'val_reader_num_workers',
                                   'number of parallel worker processes to read validation data')
    reader_pool_type = Param(Params._dummy(), 'reader_pool_type', 'type of worker pool to read data')
    optimizer = Param(Params._dummy(), 'optimizer', 'optimizer')
    model = Param(Params._dummy(), 'model', 'model')
    backend = Param(Params._dummy(), 'backend', 'backend')
    store = Param(Params._dummy(), 'store', 'store')
    metrics = Param(Params._dummy(), 'metrics', 'metrics')
    loss = Param(Params._dummy(), 'loss', 'loss')

    gradient_compression = Param(Params._dummy(), 'gradient_compression', 'Horovod gradient compression option')
    compress_sparse_cols = Param(Params._dummy(),
                                 'compress_sparse_cols',
                                 'flag indicating whether SparseVector columns should be compressed. '
                                 'requires additional compute time but saves intermediate disk space. '
                                 'recommended to avoid unless using a lot of sparse data',
                                 typeConverter=TypeConverters.toBoolean)

    loss_weights = Param(Params._dummy(), 'loss_weights', 'loss weights',
                         typeConverter=TypeConverters.toListFloat)
    sample_weight_col = Param(Params._dummy(), 'sample_weight_col',
                              'name of the column containing sample weights',
                              typeConverter=TypeConverters.toString)
    feature_cols = Param(Params._dummy(), "feature_cols", "feature column names",
                         typeConverter=TypeConverters.toListString)
    label_cols = Param(Params._dummy(), 'label_cols', 'label column names',
                       typeConverter=TypeConverters.toListString)
    continuous_cols = Param(Params._dummy(), "continuous_cols", "continuous column names",
                        typeConverter=TypeConverters.toListString)
    categorical_cols = Param(Params._dummy(), "categorical_cols", "categorical column names",
                        typeConverter=TypeConverters.toListString)
    validation = Param(Params._dummy(), 'validation',
                       'one of: float validation split [0, 1), or string validation column name')
    callbacks = Param(Params._dummy(), 'callbacks', 'callbacks')
    batch_size = Param(Params._dummy(), 'batch_size', 'batch size',
                       typeConverter=TypeConverters.toInt)
    val_batch_size = Param(Params._dummy(), 'val_batch_size', 'validation batch size',
                           typeConverter=TypeConverters.toInt)
    epochs = Param(Params._dummy(), 'epochs', 'epochs', typeConverter=TypeConverters.toInt)
    train_steps_per_epoch = Param(Params._dummy(), 'train_steps_per_epoch',
                                  'number of training (batches) steps per epoch',
                                  typeConverter=TypeConverters.toInt)
    validation_steps_per_epoch = Param(Params._dummy(), 'validation_steps_per_epoch',
                                       'number of steps (batches) for validation per epoch',
                                       typeConverter=TypeConverters.toInt)

    random_seed = Param(Params._dummy(), 'random_seed',
                        'random seed to use for DL frameworks',
                        typeConverter=TypeConverters.toInt)

    shuffle_buffer_size = Param(Params._dummy(),
                                'shuffle_buffer_size',
                                '(Deprecated) shuffling buffer size used for training samples',
                                typeConverter=TypeConverters.toInt)

    shuffle = Param(Params._dummy(),
                    'shuffle',
                    'Whether to shuffle training samples or not. Defaults to True',
                    typeConverter=TypeConverters.toBoolean)

    verbose = Param(Params._dummy(), 'verbose', 'verbose flag (0=silent, 1=enabled, other values used by frameworks)',
                    typeConverter=TypeConverters.toInt)

    partitions_per_process = Param(Params._dummy(), 'partitions_per_process',
                                   'partitions for parquet form of the DataFrame per process',
                                   typeConverter=TypeConverters.toInt)

    run_id = Param(Params._dummy(), 'run_id',
                   'unique ID for this run, if run already exists, '
                   'then training will resume from last checkpoint in the store',
                   typeConverter=TypeConverters.toString)

    transformation_fn = Param(Params._dummy(), 'transformation_fn',
                              'functions that construct the transformation '
                              'function that applies custom transformations to '
                              'every batch before train and validation steps')

    transformation_edit_fields = Param(Params._dummy(), 'transformation_edit_fields',
                                       'edit fields for petastorm TransformSpec that\'s applied to '
                                       'every batch, A list of 4-tuples with the following fields: '
                                       '(name, numpy_dtype, shape, is_nullable)')

    transformation_removed_fields = Param(Params._dummy(), 'transformation_removed_fields',
                                       'removed fields for petastorm TransformSpec that\'s applied to '
                                       'every batch, A list of field names that will be removed from the original schema.')

    label_shapes = Param(Params._dummy(), 'label_shapes', 'specifies the shape (or shapes) of the label column (or columns)')

    inmemory_cache_all = Param(Params._dummy(), 'inmemory_cache_all',
                               'Cache the data in memory for training and validation.',
                               typeConverter=TypeConverters.toBoolean)

    use_gpu = Param(Params._dummy(), 'use_gpu',
                    'Whether to use the GPU for training. '
                    'Setting this to False will skip binding to GPU even when GPU is available. '
                    'Defaults to True.',
                    typeConverter=TypeConverters.toBoolean)

    mp_start_method = Param(Params._dummy(), 'mp_start_method',
                    'The start method of multiprocessing. '
                    'This controls the start up method of python multiprocessing. On unix-based systems, acceptable values are '
                    '["fork", "spawn", "forkserver"]. Only "Spawn" is accepted on Windows. '
                    'If not set, default values will be internally set by python, please refer to: '
                    'https://docs.python.org/3/library/multiprocessing.html#multiprocessing.get_start_method.'
                    'This param defaults to None.',
                    typeConverter=TypeConverters.toString)

    backward_passes_per_step = Param(Params._dummy(), 'backward_passes_per_step',
                                     'Number of backward passes to perform before calling hvd.allreduce. '
                                     'This allows accumulating updates over multiple mini-batches before reducing and applying them. '
                                     'This param defaults to 1.',
                                     typeConverter=TypeConverters.toInt)

    tensorflow_dataset_prefetch_buffer_size = Param(Params._dummy(), 'tensorflow_dataset_prefetch_buffer_size',
                                      'The size of the prefetch buffer for the TensorFlow dataset. '
                                      'This param defaults to 0.',
                                      typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(EstimatorParams, self).__init__()

        self._setDefault(
            num_proc=None,
            store=None,
            backend=None,
            model=None,
            optimizer=None,
            loss=None,
            loss_weights=None,
            sample_weight_col=None,
            metrics=[],
            feature_cols=None,
            label_cols=None,
            continuous_cols=None,
            categorical_cols=None,
            validation=None,
            gradient_compression=None,
            compress_sparse_cols=False,
            batch_size=32,
            val_batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=[],
            random_seed=None,
            shuffle_buffer_size=None,
            shuffle=True,
            partitions_per_process=10,
            run_id=None,
            train_steps_per_epoch=None,
            validation_steps_per_epoch=None,
            transformation_fn=None,
            transformation_edit_fields=None,
            transformation_removed_fields=None,
            train_reader_num_workers=2,
            val_reader_num_workers=2,
            reader_pool_type='thread',
            label_shapes=None,
            inmemory_cache_all=False,
            use_gpu=True,
            mp_start_method=None,
            backward_passes_per_step=1,
            tensorflow_dataset_prefetch_buffer_size=0)

    def _check_params(self, metadata):
        model = self.getModel()
        if not model:
            raise ValueError('Model parameter is required')

        util.check_validation(self.getValidation())

        feature_columns = self.getFeatureCols()
        missing_features = [col for col in feature_columns if col not in metadata]
        if missing_features:
            raise ValueError('Feature columns {} not found in training DataFrame metadata'
                             .format(missing_features))

        label_columns = self.getLabelCols()
        missing_labels = [col for col in label_columns if col not in metadata]
        if missing_labels:
            raise ValueError('Label columns {} not found in training DataFrame metadata'
                             .format(missing_labels))

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setNumProc(self, value):
        return self._set(num_proc=value)

    def getNumProc(self):
        return self.getOrDefault(self.num_proc)

    def setModel(self, value):
        return self._set(model=value)

    def getModel(self):
        return self.getOrDefault(self.model)

    def setBackend(self, value):
        return self._set(backend=value)

    def getBackend(self):
        return self.getOrDefault(self.backend)

    def setStore(self, value):
        return self._set(store=value)

    def getStore(self):
        return self.getOrDefault(self.store)

    def setLoss(self, value):
        return self._set(loss=value)

    def getLoss(self):
        return self.getOrDefault(self.loss)

    def setLossWeights(self, value):
        return self._set(loss_weights=value)

    def getLossWeights(self):
        return self.getOrDefault(self.loss_weights)

    def setSampleWeightCol(self, value):
        return self._set(sample_weight_col=value)

    def getSampleWeightCol(self):
        return self.getOrDefault(self.sample_weight_col)

    def setMetrics(self, value):
        return self._set(metrics=value)

    def getMetrics(self):
        return self.getOrDefault(self.metrics)

    def setFeatureCols(self, value):
        return self._set(feature_cols=value)

    def getFeatureCols(self):
        return self.getOrDefault(self.feature_cols)

    def setLabelCols(self, value):
        return self._set(label_cols=value)

    def getLabelCols(self):
        return self.getOrDefault(self.label_cols)

    def setContinuousCols(self, value):
        return self._set(continuous_cols=value)

    def getContinuousCols(self):
        return self.getOrDefault(self.continuous_cols)

    def setCategoricalCols(self, value):
        return self._set(categorical_cols=value)

    def getCategoricalCols(self):
        return self.getOrDefault(self.categorical_cols)

    def setValidation(self, value):
        return self._set(validation=value)

    def getValidation(self):
        return self.getOrDefault(self.validation)

    def setCallbacks(self, value):
        return self._set(callbacks=value)

    def getCallbacks(self):
        return self.getOrDefault(self.callbacks)

    def setBatchSize(self, value):
        return self._set(batch_size=value)

    def getBatchSize(self):
        return self.getOrDefault(self.batch_size)

    def setValBatchSize(self, value):
        return self._set(val_batch_size=value)

    def getValBatchSize(self):
        return self.getOrDefault(self.val_batch_size)

    def setEpochs(self, value):
        return self._set(epochs=value)

    def getEpochs(self):
        return self.getOrDefault(self.epochs)

    def setTrainStepsPerEpoch(self, value):
        return self._set(train_steps_per_epoch=value)

    def getTrainStepsPerEpoch(self):
        return self.getOrDefault(self.train_steps_per_epoch)

    def setValidationStepsPerEpoch(self, value):
        return self._set(validation_steps_per_epoch=value)

    def getValidationStepsPerEpoch(self):
        return self.getOrDefault(self.validation_steps_per_epoch)

    def setVerbose(self, value):
        return self._set(verbose=value)

    def getVerbose(self):
        return self.getOrDefault(self.verbose)

    def setGradientCompression(self, value):
        return self._set(gradient_compression=value)

    def getGradientCompression(self):
        return self.getOrDefault(self.gradient_compression)

    def setCompressSparseCols(self, value):
        return self._set(compress_sparse_cols=value)

    def getCompressSparseCols(self):
        return self.getOrDefault(self.compress_sparse_cols)

    def setRandomSeed(self, value):
        return self._set(random_seed=value)

    def getRandomSeed(self):
        return self.getOrDefault(self.random_seed)

    def setShufflingBufferSize(self, value):
        return self._set(shuffle_buffer_size=value)

    def setShuffle(self, value):
        return self._set(shuffle=value)

    def getShufflingBufferSize(self):
        return self.getOrDefault(self.shuffle_buffer_size)

    def getShuffle(self):
        return self.getOrDefault(self.shuffle)

    def setOptimizer(self, value):
        return self._set(optimizer=value)

    def getOptimizer(self):
        return self.getOrDefault(self.optimizer)

    def setPartitionsPerProcess(self, value):
        return self._set(partitions_per_process=value)

    def getPartitionsPerProcess(self):
        return self.getOrDefault(self.partitions_per_process)

    def setRunId(self, value):
        return self._set(run_id=value)

    def getRunId(self):
        return self.getOrDefault(self.run_id)

    def setTransformationFn(self, value):
        return self._set(transformation_fn=value)

    def getTransformationFn(self):
        return self.getOrDefault(self.transformation_fn)

    def setTransformationEditFields(self, value):
        return self._set(transformation_edit_fields=value)

    def getTransformationEditFields(self):
        return self.getOrDefault(self.transformation_edit_fields)

    def setTransformationRemovedFields(self, value):
        return self._set(transformation_removed_fields=value)

    def getTransformationRemovedFields(self):
        return self.getOrDefault(self.transformation_removed_fields)

    def setTrainReaderNumWorker(self, value):
        return self._set(train_reader_num_workers=value)

    def getTrainReaderNumWorker(self):
        return self.getOrDefault(self.train_reader_num_workers)

    def setValReaderNumWorker(self, value):
        return self._set(val_reader_num_workers=value)

    def getValReaderNumWorker(self):
        return self.getOrDefault(self.val_reader_num_workers)

    def setReaderPoolType(self, value):
        return self._set(reader_pool_type=value)

    def getReaderPoolType(self):
        return self.getOrDefault(self.reader_pool_type)

    def setLabelShapes(self, value):
        return self._set(label_shapes=value)

    def getLabelShapes(self):
        return self.getOrDefault(self.label_shapes)

    def setInMemoryCacheAll(self, value):
        return self._set(inmemory_cache_all=value)

    def getInMemoryCacheAll(self):
        return self.getOrDefault(self.inmemory_cache_all)

    def setUseGpu(self, value):
        self._set(use_gpu=value)

    def getUseGpu(self):
        return self.getOrDefault(self.use_gpu)

    def setMpStartMethod(self, value):
        self._set(mp_start_method=value)

    def getMpStartMethod(self):
        return self.getOrDefault(self.mp_start_method)

    def setBackwardPassesPerStep(self, value):
        self._set(backward_passes_per_step=value)

    def getBackwardPassesPerStep(self):
        return self.getOrDefault(self.backward_passes_per_step)

    def setTensorflowDatasetPrefetchBufferSize(self, value):
        self._set(tensorflow_dataset_prefetch_buffer_size=value)

    def getTensorflowDatasetPrefetchBufferSize(self):
        return self.getOrDefault(self.tensorflow_dataset_prefetch_buffer_size)

class ModelParams(HasOutputCols):
    history = Param(Params._dummy(), 'history', 'history')
    model = Param(Params._dummy(), 'model', 'model')
    feature_columns = Param(Params._dummy(), 'feature_columns', 'feature columns')
    label_columns = Param(Params._dummy(), 'label_columns', 'label columns')
    run_id = Param(Params._dummy(), 'run_id',
                   'unique ID for the run that generated this model, if no ID was given by the '
                   'user, defaults to current timestamp at the time of fit()',
                   typeConverter=TypeConverters.toString)
    _metadata = Param(Params._dummy(), '_metadata',
                      'metadata contains the shape and type of input and output')

    def __init__(self):
        super(ModelParams, self).__init__()

    # Only for internal use
    def _get_metadata(self):
        return self.getOrDefault(self._metadata)

    @keyword_only
    def setParams(self, **kwargs):
        return self._set(**kwargs)

    def setHistory(self, value):
        return self._set(history=value)

    def getHistory(self):
        return self.getOrDefault(self.history)

    def setModel(self, value):
        return self._set(model=value)

    def getModel(self):
        return self.getOrDefault(self.model)

    def setFeatureColumns(self, value):
        return self._set(feature_columns=value)

    def getFeatureColumns(self):
        return self.getOrDefault(self.feature_columns)

    def setLabelColoumns(self, value):
        return self._set(label_columns=value)

    def getLabelColumns(self):
        return self.getOrDefault(self.label_columns)

    def setRunId(self, value):
        return self._set(run_id=value)

    def getRunId(self):
        return self.getOrDefault(self.run_id)

    # copied from https://github.com/apache/spark/tree/master/python/pyspark/ml/param/shared.py
    # has been removed from pyspark.ml.param.HasOutputCol in pyspark 3.0.0
    # added here to keep ModelParams API consistent between pyspark 2 and 3
    # https://github.com/apache/spark/commit/b19fd487dfe307542d65391fd7b8410fa4992698#diff-3d1fb305acc7bab18e5d91f2b69018c7
    # https://github.com/apache/spark/pull/26232
    # https://issues.apache.org/jira/browse/SPARK-29093
    def setOutputCols(self, value):
        """
        Sets the value of :py:attr:`outputCols`.
        """
        return self._set(outputCols=value)
