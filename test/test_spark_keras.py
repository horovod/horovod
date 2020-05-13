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

import collections
import warnings

import mock
import numpy as np
import sys
import tensorflow as tf

import pyspark.sql.types as T
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.sql.functions import udf

from horovod.run.runner import is_gloo_used
import horovod.spark.keras as hvd
from horovod.spark.common import constants, util
from horovod.spark.keras import remote
from horovod.spark.keras.estimator import EstimatorParams
from horovod.spark.keras.util import _custom_sparse_to_dense_fn, _serialize_param_value, BareKerasUtil, TFKerasUtil

from common import temppath
from spark_common import CallbackBackend, create_mnist_data, create_xor_data, local_store, spark_session


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
    def fit(model, train_data, val_data, steps_per_epoch, validation_steps, callbacks, verbose):
        for callback in callbacks:
            callback.set_model(model)
            callback.on_epoch_end(0, logs={})
        return mock.Mock()
    return fit


class SparkKerasTests(tf.test.TestCase):
    def __init__(self, *args, **kwargs):
        super(SparkKerasTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_fit_model(self):
        if sys.version_info < (3, 0, 0) and is_gloo_used():
            self.skipTest('Horovod on Spark over Gloo only supported on Python3')

        model = create_xor_model()
        optimizer = tf.keras.optimizers.SGD(lr=0.1)
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
                    epochs=3,
                    verbose=2)

                keras_model = keras_estimator.fit(df)

                trained_model = keras_model.getModel()
                pred = trained_model.predict([np.ones([1, 2], dtype=np.float32)])
                assert len(pred) == 1
                assert pred.dtype == np.float32

    def test_fit_model_multiclass(self):
        if sys.version_info < (3, 0, 0) and is_gloo_used():
            self.skipTest('Horovod on Spark over Gloo only supported on Python3')

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
        optimizer = tf.keras.optimizers.SGD(lr=0.1)
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
                    optimizer = tf.keras.optimizers.SGD(lr=0.1)
                    loss = 'binary_crossentropy'

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

    @mock.patch('horovod.spark.keras.estimator.remote.RemoteTrainer')
    def test_model_serialization(self, mock_remote_trainer):
        model = create_xor_model()
        optimizer = tf.keras.optimizers.SGD(lr=0.1)
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

    def test_calculate_shuffle_buffer_size_small_row_size(self):
        hvd_size = 4
        local_size = 2
        hvd_mock = mock.MagicMock()
        hvd_mock.local_size.return_value = local_size
        hvd_mock.allgather.return_value = [local_size for _ in range(hvd_size)]

        avg_row_size = 100
        train_row_count_per_worker = 100

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn()
        shuffle_size = calculate_shuffle_buffer_size(hvd_mock, avg_row_size, train_row_count_per_worker)
        assert shuffle_size == train_row_count_per_worker

    def test_calculate_shuffle_buffer_size(self):
        # case with 2 workers, one with 5 ranks and second with 3 ranks
        hvd_mock = mock.MagicMock()
        hvd_mock.allgather.return_value = [5, 5, 5, 5, 5, 3, 3, 3]
        hvd_mock.local_size.return_value = 2

        avg_row_size = 100000
        train_row_count_per_worker = 1000000

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn()
        shuffle_size = calculate_shuffle_buffer_size(hvd_mock, avg_row_size, train_row_count_per_worker)

        assert int(shuffle_size) == int(constants.TOTAL_BUFFER_MEMORY_CAP_GIB * constants.BYTES_PER_GIB / avg_row_size / 5)

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

    def test_convert_custom_sparse_to_dense_bare_keras_fn(self):
        convert_custom_sparse_to_dense_bare_keras = BareKerasUtil._convert_custom_sparse_to_dense_fn()
        custom_sparse_row = np.array([2, 1, 2, 0.1, 0.2])
        sparse_row = convert_custom_sparse_to_dense_bare_keras(custom_sparse_row, 4)
        assert np.array_equal(sparse_row, np.array([0., 0.1, 0.2, 0.]))

    def test_prepare_data_bare_keras_fn(self):
        metadata = \
            {
                'col1': {
                    'dtype': float,
                    'intermediate_format': 'nochange',
                    'max_size': 1,
                    'shape': 1
                },
                'col2': {
                    'dtype': 'float',
                    'intermediate_format': 'nochange',
                    'max_size': 1,
                    'shape': 1
                },
                'col3': {
                    'dtype': SparseVector,
                    'intermediate_format': 'custom_sparse_format',
                    'max_size': 7,
                    'shape': 10
                }
            }
        prepare_data_bare_keras = BareKerasUtil._prepare_data_fn(metadata)

        col1 = np.array([1., 2., 3.])
        col1_prepared = prepare_data_bare_keras(col1, 'col1', [-1, 3])
        assert col1_prepared.shape == (1, 3)
        assert np.array_equal(col1_prepared, np.array([[1., 2., 3.]]))

        col3 = [np.array([3., 0., 2., 5., 0., 0.2, 0.5, 0, 0]),
                np.array([4., 0., 2., 5., 6., 0.2, 0.5, 0.6, 0])]

        col3_prepared = prepare_data_bare_keras(col3, 'col3', [-1, 10])

        assert col3_prepared.shape == (2, 10)
        assert np.array_equal(col3_prepared, np.array(
            [[0., 0., 0.2, 0., 0., 0.5, 0., 0., 0., 0.], [0.2, 0., 0.5, 0., 0., 0.6, 0., 0., 0., 0.]]))

    def test_batch_generator_fn(self):
        shuffle_buffer_size = 10
        rows_in_row_group = 100
        batch_size = 32

        def _create_numpy_array(n_rows, shape):
            return np.array([[i for i in range(j, j + shape)] for j in range(n_rows)])

        def dummy_reader():
            Row = collections.namedtuple('row', ['col1', 'col2', 'sample_weight', 'label'])

            col11 = _create_numpy_array(rows_in_row_group, 1)
            col21 = _create_numpy_array(rows_in_row_group, 10)
            label1 = _create_numpy_array(rows_in_row_group, 8)
            sw1 = np.array([i / 100. for i in range(rows_in_row_group)])

            row1 = Row(col1=col11, col2=col21, label=label1, sample_weight=sw1)

            col12 = _create_numpy_array(rows_in_row_group, 1)
            col22 = _create_numpy_array(rows_in_row_group, 10)
            label2 = _create_numpy_array(rows_in_row_group, 8)
            sw2 = np.array([i / 100. for i in range(rows_in_row_group)])
            row2 = Row(col1=col12, col2=col22, label=label2, sample_weight=sw2)

            while True:
                yield row1
                yield row2

        metadata = \
            {
                'col1': {
                    'dtype': float,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': 1,
                    'shape': 1
                },
                'col2': {
                    'dtype': DenseVector,
                    'intermediate_format': constants.ARRAY,
                    'max_size': 10,
                    'shape': 10
                },
                'label': {
                    'dtype': float,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': 1,
                    'shape': 1
                },
            }

        reader = dummy_reader()

        feature_columns = ['col1', 'col2']
        label_columns = ['label']
        sample_weight_col = 'sample_weight'

        input_shapes = [[-1, 1], [-1, 2, 5]]
        output_shapes = [[-1, 2, 4]]

        batch_generator = BareKerasUtil._batch_generator_fn(
            feature_columns, label_columns, sample_weight_col,
            input_shapes, output_shapes, batch_size, metadata)

        for shuffle in [True, False]:
            batch_gen = batch_generator(reader, shuffle_buffer_size, shuffle=shuffle)

            for _ in range(10):
                batch = next(batch_gen)
                assert batch[0][0][0].shape == (1,)
                assert batch[0][1][0].shape == (2, 5)
                assert batch[1][0][0].shape == (2, 4)
                # sample weight has to be a singel np array with shape (batch_size,)
                assert batch[2][0].shape == (batch_size,)

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
