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

import collections
import logging
import os
import sys
import warnings

from packaging import version

import mock
import numpy as np
import pytest
import tensorflow as tf

import pyspark.sql.types as T
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql.functions import udf

import horovod.spark.keras as hvd
from horovod.spark.common import constants, util
from horovod.spark.keras import remote
from horovod.spark.keras.estimator import EstimatorParams
from horovod.spark.keras.util import _custom_sparse_to_dense_fn, _serialize_param_value, TFKerasUtil

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import temppath
from spark_common import CallbackBackend, create_mnist_data, create_xor_data, create_xor_data_with_val, local_store, spark_session

try:
    import nvtabular
    HAS_NVTABULAR=True
except ImportError:
    HAS_NVTABULAR=False


def create_xor_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(8, input_dim=2))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model


def create_mnist_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                                     activation='relu',
                                     input_shape=(8, 8, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def get_mock_fit_fn():
    def fit(model, data_module, steps_per_epoch, validation_steps, callbacks, verbose):
        for callback in callbacks:
            callback.set_model(model)
            callback.on_epoch_end(0, logs={})
        return mock.Mock()
    return fit


def get_sgd_optimizer():
    optimizer = tf.keras.optimizers.SGD(lr=0.1)
    return optimizer


class SparkKerasTests(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super(SparkKerasTests, self).__init__(*args, **kwargs)
        logging.getLogger('py4j.java_gateway').setLevel(logging.INFO)
        warnings.simplefilter('module')

    def test_fit_model(self):
        model = create_xor_model()
        optimizer = get_sgd_optimizer()
        loss = 'binary_crossentropy'

        with spark_session('test_fit_model') as spark:
            df = create_xor_data(spark)

            with local_store() as store:
                keras_estimator = hvd.KerasEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    feature_cols=['features'],
                    label_cols=['y'],
                    batch_size=1,
                    random_seed=1,
                    epochs=3,
                    verbose=2,
                    use_gpu=False,
                    mp_start_method='spawn')

                assert not keras_estimator.getUseGpu()
                assert 'spawn' == keras_estimator.getMpStartMethod()

                keras_estimator.setMpStartMethod('forkserver')
                assert 'forkserver' == keras_estimator.getMpStartMethod()

                keras_model = keras_estimator.fit(df)

                trained_model = keras_model.getModel()
                pred = trained_model.predict([np.ones([1, 2], dtype=np.float32)])
                assert len(pred) == 1
                assert pred.dtype == np.float32

    @pytest.mark.skipif(not HAS_NVTABULAR, reason='NVTabular unavailable')
    def test_fit_model_nvtabular_vector(self):
        from horovod.spark.keras.datamodule import NVTabularDataModule

        model = create_xor_model()
        optimizer = get_sgd_optimizer()
        loss = 'binary_crossentropy'

        with spark_session('test_fit_model_nvtabular_vector') as spark:
            df = create_xor_data(spark)
            df = df.withColumnRenamed('features', 'dense_input')

            with local_store() as store:
                keras_estimator = hvd.KerasEstimator(
                    num_proc=1,
                    data_module=NVTabularDataModule,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    feature_cols=['dense_input'],
                    label_cols=['y'],
                    categorical_cols=None,
                    continuous_cols=['dense_input'],
                    batch_size=1,
                    random_seed=1,
                    epochs=3,
                    verbose=2,
                    use_gpu=True)

                keras_model = keras_estimator.fit(df)

                trained_model = keras_model.getModel()
                pred = trained_model.predict([np.ones([1, 2], dtype=np.float32)])
                assert len(pred) == 1
                assert pred.dtype == np.float32

    @pytest.mark.skipif(not HAS_NVTABULAR, reason='NVTabular unavailable')
    def test_fit_model_nvtabular_scalar(self):
        from horovod.spark.keras.datamodule import NVTabularDataModule

        np.random.seed(1234)
        continuous = np.random.rand(1000, 2)
        weights = np.array([3.142, 1.618])
        labels = np.dot(continuous, weights)
        train_examples = [(continuous[i][0].item(), continuous[i][1].item(), labels[i].item()) for i in range(1000)]

        with spark_session('test_fit_model_nvtabular_scalar') as spark:
            with local_store() as store:
                df = spark.createDataFrame(train_examples, schema=['c1', 'c2', 'labels'])

                c1 = tf.keras.layers.Input(shape=(1,), name='c1')
                c2 = tf.keras.layers.Input(shape=(1,), name='c2')
                merged = tf.keras.layers.Concatenate(axis=1)([c1, c2])
                output = tf.keras.layers.Dense(1, activation='linear', input_shape=[2])(merged)
                model = tf.keras.Model(inputs=[c1, c2], outputs=output)
                model.summary()

                optimizer = get_sgd_optimizer()
                loss = tf.keras.losses.MeanSquaredError()

                keras_estimator = hvd.KerasEstimator(
                    num_proc=2,
                    data_module=NVTabularDataModule,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    feature_cols=['c1', 'c2'],
                    label_cols=['labels'],
                    categorical_cols=None,
                    continuous_cols=['c1', 'c2'],
                    batch_size=16,
                    random_seed=1,
                    epochs=3,
                    verbose=2,
                    use_gpu=True)

                keras_estimator.fit(df)

    def test_fit_model_multiclass(self):
        model = create_mnist_model()
        optimizer = tf.keras.optimizers.Adadelta(1.0)
        loss = tf.keras.losses.categorical_crossentropy

        for num_cores in [2, constants.TOTAL_BUFFER_MEMORY_CAP_GIB + 1]:
            with spark_session('test_fit_model_multiclass', cores=num_cores) as spark:
                df = create_mnist_data(spark)

                with local_store() as store:
                    keras_estimator = hvd.KerasEstimator(
                        num_proc=num_cores,
                        store=store,
                        model=model,
                        optimizer=optimizer,
                        loss=loss,
                        metrics=['accuracy'],
                        feature_cols=['features'],
                        label_cols=['label_vec'],
                        batch_size=2,
                        epochs=2,
                        verbose=2)

                    keras_model = keras_estimator.fit(df).setOutputCols(['label_prob'])
                    pred_df = keras_model.transform(df)

                    argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
                    pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))

                    preds = pred_df.collect()
                    assert len(preds) == df.count()

                    row = preds[0]
                    label_prob = row.label_prob.toArray().tolist()
                    assert label_prob[int(row.label_pred)] == max(label_prob)

    @mock.patch('horovod.spark.keras.remote._pin_gpu_fn')
    @mock.patch('horovod.spark.keras.util.TFKerasUtil.fit_fn')
    def test_restore_from_checkpoint(self, mock_fit_fn, mock_pin_gpu_fn):
        mock_fit_fn.return_value = get_mock_fit_fn()
        mock_pin_gpu_fn.return_value = mock.Mock()

        model = create_xor_model()
        optimizer = get_sgd_optimizer()
        loss = 'binary_crossentropy'

        with spark_session('test_restore_from_checkpoint') as spark:
            df = create_xor_data(spark)

            backend = CallbackBackend()

            run_id = 'run01'
            with local_store() as store:
                keras_estimator = hvd.KerasEstimator(
                    backend=backend,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    feature_cols=['features'],
                    label_cols=['y'],
                    batch_size=1,
                    epochs=3,
                    verbose=2,
                    run_id=run_id)

                keras_estimator._load_model_from_checkpoint = mock.Mock(
                    side_effect=keras_estimator._load_model_from_checkpoint)

                ckpt_path = store.get_checkpoint_path(run_id)
                assert not store.exists(ckpt_path)
                keras_estimator._load_model_from_checkpoint.assert_not_called()
                keras_model = keras_estimator.fit(df)

                trained_model = keras_model.getModel()
                pred = trained_model.predict([np.ones([1, 2], dtype=np.float64)])
                assert len(pred) == 1

                assert store.exists(ckpt_path)

                keras_estimator.fit(df)
                keras_estimator._load_model_from_checkpoint.assert_called()

    @mock.patch('horovod.spark.keras.remote._pin_gpu_fn')
    @mock.patch('horovod.spark.keras.util.TFKerasUtil.fit_fn')
    def test_keras_direct_parquet_train(self, mock_fit_fn, mock_pin_gpu_fn):
        mock_fit_fn.return_value = get_mock_fit_fn()
        mock_pin_gpu_fn.return_value = mock.Mock()

        with spark_session('test_keras_direct_parquet_train') as spark:
            df = create_xor_data_with_val(spark)

            backend = CallbackBackend()
            with local_store() as store:
                store.get_train_data_path = lambda v=None: store._train_path
                store.get_val_data_path = lambda v=None: store._val_path

                # Make sure we cover val dataloader cases
                for validation in [None, 'val']:
                    with util.prepare_data(backend.num_processes(),
                                           store,
                                           df,
                                           feature_columns=['features'],
                                           label_columns=['y'],
                                           validation=validation):
                        model = create_xor_model()
                        optimizer = get_sgd_optimizer()
                        loss = 'binary_crossentropy'

                        for inmemory_cache_all in [False, True]:
                            for reader_pool_type in ['process', 'thread']:
                                est = hvd.KerasEstimator(
                                    backend=backend,
                                    store=store,
                                    model=model,
                                    optimizer=optimizer,
                                    loss=loss,
                                    feature_cols=['features'],
                                    label_cols=['y'],
                                    batch_size=1,
                                    epochs=3,
                                    reader_pool_type=reader_pool_type,
                                    validation=validation,
                                    inmemory_cache_all=inmemory_cache_all,
                                    verbose=2)

                                transformer = est.fit_on_parquet()
                                predictions = transformer.transform(df)
                                assert predictions.count() == df.count()

    @mock.patch('horovod.spark.keras.remote._pin_gpu_fn')
    @mock.patch('horovod.spark.keras.util.TFKerasUtil.fit_fn')
    def test_keras_model_checkpoint_callback(self, mock_fit_fn, mock_pin_gpu_fn):
        from horovod.tensorflow.keras.callbacks import BestModelCheckpoint

        def _get_mock_fit_fn(checkpoint_callback_provided):
            def fit(model, data_module, steps_per_epoch, validation_steps, callbacks,
                    verbose):
                returned_model_checkpoint_present = False
                model_checkpoint_present = False
                for callback in callbacks:
                    callback.set_model(model)
                    if checkpoint_callback_provided:
                        callback.on_epoch_end(0, logs={'binary_crossentropy': 0.3})
                    else:
                        callback.on_epoch_end(0, logs={'binary_crossentropy': 0.3})

                    if checkpoint_callback_provided and isinstance(callback, BestModelCheckpoint):
                        self.assertIsNotNone(callback.filepath)
                        self.assertTrue(callback.save_best_only)
                        self.assertEqual(callback.monitor, 'binary_crossentropy')
                        returned_model_checkpoint_present = True

                    if not checkpoint_callback_provided and isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                        self.assertFalse(callback.save_best_only)
                        self.assertFalse(callback.save_best_only)
                        self.assertEqual(callback.monitor, 'val_loss')
                        model_checkpoint_present = True

                if checkpoint_callback_provided:
                    self.assertTrue(returned_model_checkpoint_present)
                    self.assertFalse(model_checkpoint_present)
                else:
                    self.assertFalse(returned_model_checkpoint_present)
                    self.assertTrue(model_checkpoint_present)

                return mock.Mock()

            return fit

        mock_pin_gpu_fn.return_value = mock.Mock()

        with spark_session('test_keras_model_chekcpoint_callbacks') as spark:
            df = create_xor_data(spark)

            backend = CallbackBackend()
            with local_store() as store:
                store.get_train_data_path = lambda v=None: store._train_path
                store.get_val_data_path = lambda v=None: store._val_path

                with util.prepare_data(backend.num_processes(),
                                       store,
                                       df,
                                       feature_columns=['features'],
                                       label_columns=['y']):
                    model = create_xor_model()
                    optimizer = get_sgd_optimizer()
                    loss = 'binary_crossentropy'

                    # Test when the checkpoint callback is not set, the correct one is created
                    mock_fit_fn.return_value = _get_mock_fit_fn(checkpoint_callback_provided=False)
                    est = hvd.KerasEstimator(
                        backend=backend,
                        store=store,
                        model=model,
                        optimizer=optimizer,
                        loss=loss,
                        feature_cols=['features'],
                        label_cols=['y'],
                        batch_size=1,
                        epochs=3,
                        verbose=2)

                    transformer = est.fit_on_parquet()
                    predictions = transformer.transform(df)
                    assert predictions.count() == df.count()

                    # Test if checkpoint call back is correctly set to the model
                    mock_fit_fn.return_value = _get_mock_fit_fn(checkpoint_callback_provided=True)
                    checkpoint_callback = BestModelCheckpoint(monitor='binary_crossentropy')
                    est = hvd.KerasEstimator(
                        backend=backend,
                        store=store,
                        model=model,
                        optimizer=optimizer,
                        loss=loss,
                        feature_cols=['features'],
                        label_cols=['y'],
                        batch_size=1,
                        epochs=3,
                        verbose=2,
                        checkpoint_callback=checkpoint_callback)

                    transformer = est.fit_on_parquet()
                    predictions = transformer.transform(df)
                    assert predictions.count() == df.count()

    @mock.patch('horovod.spark.keras.estimator.remote.RemoteTrainer')
    def test_model_serialization(self, mock_remote_trainer):
        model = create_xor_model()
        optimizer = get_sgd_optimizer()
        loss = 'binary_crossentropy'

        def train(serialized_model, train_rows, val_rows, avg_row_size):
            return None, serialized_model, 2
        mock_remote_trainer.return_value = train

        with spark_session('test_model_serialization') as spark:
            df = create_xor_data(spark)

            keras_estimator = hvd.KerasEstimator(
                model=model,
                optimizer=optimizer,
                loss=loss,
                feature_cols=['features'],
                label_cols=['y'],
                batch_size=1,
                epochs=3,
                verbose=2)

            backend = CallbackBackend()
            with local_store() as store:
                with temppath() as saved_path:
                    keras_estimator.save(saved_path)
                    keras_estimator_loaded = hvd.KerasEstimator.load(saved_path)

                keras_model = keras_estimator_loaded.fit(df, params={
                    keras_estimator_loaded.backend: backend,
                    keras_estimator_loaded.store: store
                })

                trained_model = keras_model.getModel()
                pred = trained_model.predict([np.ones([1, 2], dtype=np.float32)])
                assert len(pred) == 1
                assert pred.dtype == np.float32

    def test_serialize_param_value(self):
        serialized_backend = _serialize_param_value(EstimatorParams.backend.name, 'dummy_value', None, None)
        assert serialized_backend is None

        serialized_store = _serialize_param_value(EstimatorParams.store.name, 'dummy_value', None, None)
        assert serialized_store is None

        serialized_dummy_param = _serialize_param_value('dummy_param_name', None, None, None)
        assert serialized_dummy_param is None

    def test_custom_sparse_to_dense_fn(self):
        dense_shape = 10
        custom_sparse_to_dense = _custom_sparse_to_dense_fn()
        dense_vector = tf.constant([3., 1., 3., 6., 10., 30., 60., 0, 0, 0])
        sparse_vector = custom_sparse_to_dense(dense_vector, dense_shape)
        sparse_vector_values = self.evaluate(sparse_vector)[0]
        assert sparse_vector_values[1] == 10
        assert sparse_vector_values[3] == 30
        assert sparse_vector_values[6] == 60
        assert len(sparse_vector_values) == dense_shape

    def test_reshape(self):
        metadata = \
            {
                'col1': {
                    'dtype': float,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': 1,
                    'shape': 1
                },
                'col2': {
                    'dtype': SparseVector,
                    'intermediate_format': constants.CUSTOM_SPARSE,
                    'max_size': 5,
                    'shape': 10
                },
                'label': {
                    'dtype': float,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': 1,
                    'shape': 1
                },
            }

        feature_columns = ['col1', 'col2']
        label_columns = ['label']
        sample_weight_col = 'sample_weight'

        Row = collections.namedtuple('row', ['col1', 'col2', 'sample_weight', 'label'])

        col11 = tf.constant([3.])
        col21 = tf.constant([3., 1., 3., 6., 10., 30., 60., 0, 0, 0, 0])
        label1 = tf.constant([1.])
        sw1 = tf.constant([.06])
        row1 = Row(col1=col11, col2=col21, label=label1, sample_weight=sw1)

        reshape_fn = TFKerasUtil._reshape_fn(
            sample_weight_col, feature_columns, label_columns, metadata)

        reshaped_row = reshape_fn(row1)
        reshaped_row_value = self.evaluate(reshaped_row)

        assert np.allclose(reshaped_row_value['sample_weight'], np.array([0.06]))
        assert np.allclose(reshaped_row_value['col1'], np.array([3.]))
        assert np.allclose(reshaped_row_value['col2'],
                           np.array([[0., 10., 0., 30., 0., 0., 60., 0., 0., 0.]]))
        assert np.allclose(reshaped_row_value['label'], np.array([1.]))

    def test_prep_data_tf_keras_fn_with_sparse_col(self):
        has_sparse_col = True

        feature_columns = ['col1', 'col2']
        label_columns = ['label1', 'label2']
        sample_weight_col = 'sample_weight'

        col1 = tf.constant([3.])
        col2 = tf.constant([3., 1., 3., 6., 10., 30., 60., 0, 0, 0])
        label1 = tf.constant([1., 2., 3., 4.])
        label2 = tf.constant([1., 2., 3., 4.])
        sw1 = tf.constant([.06])

        input_shapes = [[-1, 1], [-1, 2, 5]]
        output_shapes = [[-1, 4], [-1, 2, 2]]
        output_names = ['label1', 'label2']

        prep_data_tf_keras = \
            TFKerasUtil._prep_data_fn(has_sparse_col, sample_weight_col,
                                      feature_columns, label_columns, input_shapes,
                                      output_shapes, output_names)

        row = {'col1': col1, 'col2': col2, 'label1': label1, 'label2': label2, sample_weight_col: sw1}

        prepped_row = prep_data_tf_keras(row)
        prepped_row_vals = self.evaluate(prepped_row)

        assert np.array_equal(prepped_row_vals[0][0], np.array([[3.]]))
        assert np.array_equal(prepped_row_vals[0][1],
                              np.array([[[3., 1., 3., 6., 10.], [30., 60., 0., 0., 0.]]]))

        assert np.array_equal(prepped_row_vals[1][0], np.array([[1., 2., 3., 4.]]))
        assert np.array_equal(prepped_row_vals[1][1], np.array([[[1., 2.], [3., 4.]]]))

        assert np.allclose(prepped_row_vals[2]['label1'], np.array([0.06]))
        assert np.allclose(prepped_row_vals[2]['label2'], np.array([0.06]))

    def test_prep_data_tf_keras_fn_without_sparse_col(self):
        has_sparse_col = False

        feature_columns = ['col1', 'col2']
        label_columns = ['label1', 'label2']
        sample_weight_col = 'sample_weight'

        col1 = tf.constant([3.])
        col2 = tf.constant([float(i) for i in range(10)])
        label1 = tf.constant([1., 2., 3., 4.])
        label2 = tf.constant([1., 2., 3., 4.])
        sw1 = tf.constant([.06])

        input_shapes = [[-1, 1], [-1, 2, 5]]
        output_shapes = [[-1, 4], [-1, 2, 2]]
        output_names = ['label1', 'label2']

        prep_data_tf_keras = \
            TFKerasUtil._prep_data_fn(has_sparse_col, sample_weight_col,
                                      feature_columns, label_columns, input_shapes,
                                      output_shapes, output_names)

        Row = collections.namedtuple('row', ['col1', 'col2', sample_weight_col, 'label1', 'label2'])
        row = Row(col1=col1, col2=col2, label1=label1, label2=label2, sample_weight=sw1)

        prepped_row = prep_data_tf_keras(row)
        prepped_row_vals = self.evaluate(prepped_row)

        assert np.array_equal(prepped_row_vals[0][0], np.array([[3.]]))
        assert np.array_equal(prepped_row_vals[0][1],
                              np.array([[[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.]]]))

        assert np.array_equal(prepped_row_vals[1][0], np.array([[1., 2., 3., 4.]]))
        assert np.array_equal(prepped_row_vals[1][1], np.array([[[1., 2.], [3., 4.]]]))

        assert np.allclose(prepped_row_vals[2]['label1'], np.array([0.06]))
        assert np.allclose(prepped_row_vals[2]['label2'], np.array([0.06]))
