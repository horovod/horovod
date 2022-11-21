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

import logging
import os
import sys
import unittest
import warnings

import mock
from unittest.mock import Mock
import pytest
import numpy as np
from packaging import version

import torch
import torch.nn as nn
from torch.nn import functional as F

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import FloatType, IntegerType

# Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x and 2.6.x: https://github.com/horovod/horovod/pull/3263
skip_lightning_tests = False
try:
    # tensorflow has to be imported BEFORE pytorch_lightning, otherwise we see the segfault right away
    import tensorflow as tf
    from packaging import version
    if version.parse('2.5.0') <= version.parse(tf.__version__) < version.parse('2.7.0'):
        skip_lightning_tests = True
except ImportError:
    pass

import pytorch_lightning as pl

import horovod
import horovod.torch as hvd
from horovod.torch.elastic import run

import horovod.spark.lightning as hvd_spark
from horovod.spark.lightning import remote
from horovod.spark.lightning.estimator import EstimatorParams, _torch_param_serialize, MIN_PL_VERSION
from horovod.spark.lightning.legacy import to_lightning_module

from horovod.common.util import gloo_built
from horovod.spark.common import constants, util

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import tempdir, spawn, is_built
from spark_common import CallbackBackend, create_noisy_xor_data, create_noisy_xor_data_with_val, create_xor_data, local_store, spark_session


class XOR(pl.LightningModule):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
        # The Lightning checkpoint also saves the arguments passed into the LightningModule init
        # under the "hyper_parameters" key in the checkpoint.
        self.save_hyperparameters()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_nb):
        x, y = batch['features'], batch['y'].unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch['features'], batch['y'].unsqueeze(1)
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log('avg_val_loss', avg_loss)


class LegacyXOR(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LegacyXOR, self).__init__()
        self.lin1 = nn.Linear(input_dim, 8)
        self.lin2 = nn.Linear(8, output_dim)

    def forward(self, features):
        x = features.float()
        x = self.lin1(x)
        x = torch.tanh(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)
        return x


def create_xor_model(input_dim=2, output_dim=1):
    return XOR(input_dim, output_dim)


def create_legacy_xor_model(input_dim=2, output_dim=1):
    return LegacyXOR(input_dim, output_dim)


@pytest.mark.skipif(version.parse(pl.__version__) < version.parse(MIN_PL_VERSION), reason='Pytorch lightning version is not supported.')
class SparkLightningTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SparkLightningTests, self).__init__(*args, **kwargs)
        logging.getLogger('py4j.java_gateway').setLevel(logging.INFO)
        warnings.simplefilter('module')

    def test_fit_model(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        model = create_xor_model()

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)

            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    random_seed=1,
                    verbose=2,
                    mp_start_method='spawn',
                    trainer_args={"num_sanity_val_steps": 0})

                assert 'spawn' == torch_estimator.getMpStartMethod()
                torch_estimator.setMpStartMethod('forkserver')
                assert 'forkserver' == torch_estimator.getMpStartMethod()

                torch_model = torch_estimator.fit(df)

                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    def test_terminate_on_nan_flag(self):
        model = create_xor_model()

        with spark_session('test_terminate_on_nan_flag') as spark:
            df = create_noisy_xor_data(spark)

            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    terminate_on_nan=True,
                    profiler="pytorch",
                    trainer_args={"num_sanity_val_steps": 0})
                assert torch_estimator.getTerminateOnNan() == True

    def test_legacy_fit_model(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        model = create_legacy_xor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = F.binary_cross_entropy

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)

            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    sample_weight_col='weight')

                torch_model = torch_estimator.fit(df)

                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    def test_restore_from_checkpoint(self):
        
        model = create_xor_model()

        with spark_session('test_restore_from_checkpoint') as spark:
            df = create_noisy_xor_data(spark)

            ctx = CallbackBackend()

            run_id = 'run01'
            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    backend=ctx,
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    run_id=run_id,
                    trainer_args={"num_sanity_val_steps": 0})

                torch_estimator._read_checkpoint = Mock(side_effect=torch_estimator._read_checkpoint)

                ckpt_path = store.get_checkpoint_path(run_id)
                assert not store.exists(ckpt_path)
                torch_estimator._read_checkpoint.assert_not_called()
                torch_estimator.fit(df)

                assert store.exists(ckpt_path)
                torch_estimator.fit(df)
                torch_estimator._read_checkpoint.assert_called()

    def test_legacy_restore_from_checkpoint(self):

        model = create_legacy_xor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = nn.BCELoss()

        with spark_session('test_restore_from_checkpoint') as spark:
            df = create_noisy_xor_data(spark)

            ctx = CallbackBackend()

            run_id = 'run01'
            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    backend=ctx,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    run_id=run_id)

                torch_estimator._read_checkpoint = Mock(side_effect=torch_estimator._read_checkpoint)

                ckpt_path = store.get_checkpoint_path(run_id)
                assert not store.exists(ckpt_path)
                torch_estimator._read_checkpoint.assert_not_called()
                torch_estimator.fit(df)

                assert store.exists(ckpt_path)
                torch_estimator.fit(df)
                torch_estimator._read_checkpoint.assert_called()

    def test_transform_multi_class(self):
        # set dim as 2, to mock a multi class model.
        model = create_xor_model(output_dim=2)

        with spark_session('test_transform_multi_class') as spark:
            df = create_xor_data(spark)
            metadata = util._get_metadata(df)

            torch_model = hvd_spark.TorchModel(history=None,
                                               model=model,
                                               input_shapes=[[-1, 2]],
                                               feature_columns=['features'],
                                               label_columns=['y'],
                                               _metadata=metadata)
            out_df = torch_model.transform(df)

            # in multi class model, model output is a vector but label is number.
            expected_types = {
                'x1': IntegerType,
                'x2': IntegerType,
                'features': VectorUDT,
                'weight': FloatType,
                'y': FloatType,
                'y__output': VectorUDT
            }

            for field in out_df.schema.fields:
                assert type(field.dataType) == expected_types[field.name]

    def test_prepare_data(self):
        with spark_session('test_prepare_data') as spark:
            df = create_xor_data(spark)

            train_rows = df.count()
            schema_cols = ['features', 'y']
            metadata = util._get_metadata(df)
            assert metadata['features']['intermediate_format'] == constants.ARRAY

            to_petastorm = util.to_petastorm_fn(schema_cols, metadata)
            modified_df = df.rdd.map(to_petastorm).toDF()
            data = modified_df.collect()

            prepare_data = remote._prepare_data_fn(metadata)
            features = torch.tensor([data[i].features for i in range(train_rows)])
            features_prepared = prepare_data('features', features)
            assert np.array_equal(features_prepared, features)

    def test_torch_param_serialize(self):
        serialized_backend = _torch_param_serialize(EstimatorParams.backend.name, 'dummy_value')
        assert serialized_backend is None

        serialized_store = _torch_param_serialize(EstimatorParams.store.name, 'dummy_value')
        assert serialized_store is None

        serialized_dummy_param = _torch_param_serialize('dummy_param_name', None)
        assert serialized_dummy_param is None

    def test_direct_parquet_train(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        with spark_session('test_direct_parquet_train') as spark:
            df = create_noisy_xor_data_with_val(spark)

            backend = CallbackBackend()
            with local_store() as store:
                store.get_train_data_path = lambda v=None: store._train_path
                store.get_val_data_path = lambda v=None: store._val_path

                # Make sure to cover val dataloader cases
                for validation in [None, 'val']:
                    with util.prepare_data(backend.num_processes(),
                                           store,
                                           df,
                                           feature_columns=['features'],
                                           label_columns=['y'],
                                           validation=validation):
                        model = create_xor_model()

                        for inmemory_cache_all in [False, True]:
                            for reader_pool_type in ['process', 'thread']:
                                est = hvd_spark.TorchEstimator(
                                    backend=backend,
                                    store=store,
                                    model=model,
                                    input_shapes=[[-1, 2]],
                                    feature_cols=['features'],
                                    label_cols=['y'],
                                    validation=validation,
                                    batch_size=1,
                                    epochs=3,
                                    verbose=2,
                                    inmemory_cache_all=inmemory_cache_all,
                                    reader_pool_type=reader_pool_type,
                                    trainer_args={"num_sanity_val_steps": 0})

                                transformer = est.fit_on_parquet()
                                predictions = transformer.transform(df)
                                assert predictions.count() == df.count()

    def test_direct_parquet_train_with_no_val_column(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        with spark_session('test_direct_parquet_train_with_no_val_column') as spark:
            df_train = create_noisy_xor_data(spark)
            df_val = create_noisy_xor_data(spark)

            def to_petastorm(df):
                metadata = None
                if util._has_vector_column(df):
                    to_petastorm = util.to_petastorm_fn(["features", "y"], metadata)
                    df = df.rdd.map(to_petastorm).toDF()
                return df

            df_train = to_petastorm(df_train)
            df_val = to_petastorm(df_val)

            df_train.show(1)
            print(df_train.count())
            df_val.show(1)
            print(df_val.count())

            backend = CallbackBackend()
            with local_store() as store:
                store.get_train_data_path = lambda v=None: store._train_path
                store.get_val_data_path = lambda v=None: store._val_path

                print(store.get_train_data_path())
                print(store.get_val_data_path())

                df_train \
                    .coalesce(4) \
                    .write \
                    .mode('overwrite') \
                    .parquet(store.get_train_data_path())

                df_val \
                    .coalesce(4) \
                    .write \
                    .mode('overwrite') \
                    .parquet(store.get_val_data_path())

                model = create_xor_model()

                inmemory_cache_all = True
                reader_pool_type = 'process'
                est = hvd_spark.TorchEstimator(
                    backend=backend,
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    batch_size=64,
                    epochs=2,
                    verbose=2,
                    inmemory_cache_all=inmemory_cache_all,
                    reader_pool_type=reader_pool_type,
                    trainer_args={"num_sanity_val_steps": 0})

                # set validation to any random strings would work.
                est.setValidation("True")

                transformer = est.fit_on_parquet()

                predictions = transformer.transform(df_train)
                assert predictions.count() == df_train.count()

    def test_legacy_calculate_loss_with_sample_weight(self):
        labels = torch.tensor([[1.0, 2.0, 3.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0]])

        def fn_minus(output, label, reduction=None):
            losses = label - output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        def fn_add(output, label, reduction=None):
            losses = label + output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        kwargs = dict(model=Mock(), optimizer=Mock(), feature_cols=[], sample_weights_col='', validation=0)
        model = to_lightning_module(loss_fns=[fn_minus], loss_weights=[1], label_cols=['a'],  **kwargs)
        loss = model._calculate_loss(outputs, labels, sample_weights=torch.tensor([1.0, 6.0, 3.0]))
        assert loss == 5.0

        labels = torch.tensor([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        model = to_lightning_module(loss_fns=[fn_minus, fn_add], loss_weights=[0.2, 0.8], label_cols=['a', 'b'], **kwargs)
        loss = model._calculate_loss(outputs, labels, sample_weights=torch.tensor([1.0, 6.0, 3.0]))
        assert loss == torch.tensor(9.0)

    def test_legacy_calculate_loss_without_sample_weight(self):
        labels = torch.tensor([[1.0, 2.0, 3.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0]])

        def fn_minus(output, label, reduction=None):
            losses = label - output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        def fn_add(output, label, reduction=None):
            losses = label + output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        kwargs = dict(model=Mock(), optimizer=Mock(), feature_cols=[], sample_weights_col=None, validation=0)
        model = to_lightning_module(loss_fns=[fn_minus], loss_weights=[1], label_cols=['a'],  **kwargs)
        loss = model._calculate_loss(outputs, labels)
        assert loss == 1.0

        labels = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        model = to_lightning_module(loss_fns=[fn_minus, fn_add], loss_weights=[0.2, 0.8], label_cols=['a', 'b'], **kwargs)
        loss = model._calculate_loss(outputs, labels)
        assert torch.isclose(loss, torch.tensor(2.6))

    """
    Test that horovod.spark.run_elastic works properly in a simple setup.
    """
    def test_happy_run_elastic(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        if not gloo_built():
            self.skipTest("Gloo is not available")

        with spark_session('test_happy_run_elastic'):
            res = horovod.spark.run_elastic(fn, args=(2, 5, 4),
                                            num_proc=2, min_num_proc=2, max_num_proc=2,
                                            start_timeout=10, verbose=2)
            self.assertListEqual([([0, 3, 0, 1, 1, 3, 0, 1], 0),
                                  ([0, 3, 0, 1, 1, 3, 0, 1], 1)], res)

    """
    Test that horovod.spark.run_elastic works properly in a fault-tolerant situation.
    """
    def test_happy_run_elastic_fault_tolerant(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        if not gloo_built():
            self.skipTest("Gloo is not available")

        with spark_session('test_happy_run_elastic_fault_tolerant', max_failures=3):
            with tempdir() as dir:
                # these files make training function fail in given rank, epoch and batch
                with open(os.path.sep.join([dir, 'rank_1_epoch_2_batch_4_fail']), 'w'), \
                     open(os.path.sep.join([dir, 'rank_0_epoch_3_batch_1_fail']), 'w'), \
                     open(os.path.sep.join([dir, 'rank_1_epoch_4_batch_2_fail']), 'w'):
                    pass
                res = horovod.spark.run_elastic(fn, args=(2, 5, 5, dir),
                                                env={'HOROVOD_LOG_LEVEL': 'DEBUG'},
                                                num_proc=2, min_num_proc=2, max_num_proc=2,
                                                start_timeout=5, verbose=2)
                self.assertListEqual([([0, 4, 0, 4, 1, 4, 0, 4], 0),
                                      ([0, 4, 0, 4, 1, 4, 0, 4], 1)], res)

    """
    Test that horovod.spark.run_elastic in a fault-tolerant mode fails on too many failures.
    """
    def test_happy_run_elastic_fault_tolerant_fails(self):
        self.skipTest('elastic horovod does not support shutdown from the spark driver '
                      'while elastic driver is waiting for hosts to come up')

        if not gloo_built():
            self.skipTest("Gloo is not available")

        with spark_session('test_happy_run_elastic_fault_tolerant_fails', max_failures=2):
            with tempdir() as dir:
                # these files make training function fail in given rank, epoch and batch
                # we have as many failures as Spark has max_failures (per task / index)
                with open(os.path.sep.join([dir, 'rank_1_epoch_2_batch_4_fail']), 'w'), \
                     open(os.path.sep.join([dir, 'rank_1_epoch_3_batch_1_fail']), 'w'):
                    pass
                res = horovod.spark.run_elastic(fn, args=(2, 5, 5, dir),
                                                env={'HOROVOD_LOG_LEVEL': 'DEBUG'},
                                                num_proc=2, min_num_proc=2, max_num_proc=2,
                                                start_timeout=5, verbose=2)
                self.assertListEqual([([0, 4, 0, 4, 1, 4, 0, 4], 0),
                                      ([0, 4, 0, 4, 1, 4, 0, 4], 1)], res)


    """
    Test dummy callback function from pytorch lightning trainer.
    """
    def test_dummy_callback(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        from pytorch_lightning.callbacks import Callback
        model = create_xor_model()

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)

            for num_proc in [1, 2]:
                for epochs in [2, 3]:

                    class MyDummyCallback(Callback):
                        def __init__(self):
                            self.epcoh_end_counter = 0
                            self.train_epcoh_end_counter = 0

                        def on_init_start(self, trainer):
                            print('Starting to init trainer!')

                        def on_init_end(self, trainer):
                            print('Trainer is initialized.')

                        def on_epoch_end(self, trainer, model):
                            print('A epoch ended.')
                            self.epcoh_end_counter += 1

                        def on_train_epoch_end(self, trainer, model, unused=None):
                            print('A train epoch ended.')
                            self.train_epcoh_end_counter += 1

                        def on_train_end(self, trainer, model):
                            print('Training ends')
                            assert self.train_epcoh_end_counter == epochs

                    dm_callback = MyDummyCallback()
                    callbacks = [dm_callback]

                    with local_store() as store:
                        torch_estimator = hvd_spark.TorchEstimator(
                            num_proc=num_proc,
                            store=store,
                            model=model,
                            input_shapes=[[-1, 2]],
                            feature_cols=['features'],
                            label_cols=['y'],
                            validation=0.2,
                            batch_size=4,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            trainer_args={"num_sanity_val_steps": 0})

                        torch_model = torch_estimator.fit(df)

                        # TODO: Find a way to pass log metrics from remote, and assert base on the logger.
                        trained_model = torch_model.getModel()
                        pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                        assert len(pred) == 1
                        assert pred.dtype == torch.float32

    """
    Test callback function for learning rate scheduler and monitor.
    """
    def test_lr_scheduler_callback(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        from pytorch_lightning.callbacks import LearningRateMonitor

        class LRTestingModel(XOR):
            def configure_optimizers(self):
                optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

                def lambda_func(epoch):
                    return epoch // 30

                lr_scheduler = {
                    'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func),
                    'name': 'my_logging_name'
                }
                return [optimizer], [lr_scheduler]

        model = LRTestingModel()

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)

            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks = [lr_monitor]

            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    callbacks=callbacks)

                torch_model = torch_estimator.fit(df)

                # TODO: Find a way to pass log metrics from remote, and assert base on the logger.
                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    """
    Test callback function for model checkpoint.
    """
    def test_model_checkpoint_callback(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)
            model = create_xor_model()

            with tempdir() as dir:
                checkpoint_callback = ModelCheckpoint(dirpath=dir)
                callbacks = [checkpoint_callback]

                with local_store() as store:
                    torch_estimator = hvd_spark.TorchEstimator(
                        num_proc=2,
                        store=store,
                        model=model,
                        input_shapes=[[-1, 2]],
                        feature_cols=['features'],
                        label_cols=['y'],
                        validation=0.2,
                        batch_size=4,
                        epochs=2,
                        verbose=2,
                        callbacks=callbacks,
                        trainer_args={"num_sanity_val_steps": 0})

                    torch_model = torch_estimator.fit(df)

                    # TODO: Find a way to pass log metrics from remote, and assert base on the logger.
                    trained_model = torch_model.getModel()
                    pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                    assert len(pred) == 1
                    assert pred.dtype == torch.float32

    """
    Test callback function for early stop.
    """
    def test_early_stop_callback(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        if version.parse(torch.__version__) >= version.parse('1.13'):
            self.skipTest('Torch 1.13+ fails EarlyStopping CB usage with Horovod.')

        from pytorch_lightning.callbacks.early_stopping import EarlyStopping

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)
            model = create_xor_model()

            early_stop_callback = EarlyStopping(monitor='val_loss',
                                                min_delta=0.00,
                                                patience=3,
                                                verbose=True,
                                                mode='max')
            callbacks = [early_stop_callback]

            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    callbacks=callbacks,
                    trainer_args={"num_sanity_val_steps": 0})

                torch_model = torch_estimator.fit(df)

                # TODO: Find a way to pass log metrics from remote, and assert base on the logger.
                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    """
    Test train model with inmemory_cache_all
    """
    def test_train_with_inmemory_cache_all(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)
            model = create_xor_model()

            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    num_proc=1, # Normally inmem dataloader is for single worker training with small data
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    verbose=2,
                    inmemory_cache_all=True,
                    trainer_args={"num_sanity_val_steps": 0})

                torch_model = torch_estimator.fit(df)

                # TODO: Find a way to pass log metrics from remote, and assert base on the logger.
                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    """
    Test train model with custom data module (using PytorchAsyncDataLoader)
    """
    def test_train_with_custom_data_module(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        from horovod.spark.data_loaders.pytorch_data_loaders import PytorchAsyncDataLoader
        class CustomDataModule(pl.LightningDataModule):
            """Custom DataModule for Lightning Estimator, using PytorchAsyncDataLoader"""
            def __init__(self, train_dir: str, val_dir: str, has_val: bool=True,
                         train_batch_size: int=32, val_batch_size: int=32, shuffle: bool=True,
                         num_reader_epochs=None, cur_shard: int=0, shard_count: int=1, schema_fields=None,
                         storage_options=None, steps_per_epoch_train: int=1, steps_per_epoch_val: int=1, verbose=True, **kwargs):
                super().__init__()
                self.train_dir = train_dir
                self.val_dir = val_dir
                self.has_val = has_val
                self.train_batch_size = train_batch_size
                self.val_batch_size = val_batch_size
                self.shuffle = shuffle
                self.num_reader_epochs = num_reader_epochs
                self.cur_shard = cur_shard
                self.shard_count = shard_count
                self.schema_fields = schema_fields
                self.storage_options = storage_options
                self.steps_per_epoch_train = steps_per_epoch_train
                self.steps_per_epoch_val = steps_per_epoch_val
                self.verbose = verbose

            def setup(self, stage=None):
                # Assign train/val datasets for use in dataloaders
                from petastorm import make_batch_reader
                if stage == 'fit' or stage is None:
                    self.train_reader = make_batch_reader(self.train_dir, num_epochs=self.num_reader_epochs,
                                                          cur_shard=self.cur_shard, shard_count=self.shard_count,
                                                          shuffle_rows=self.shuffle,
                                                          shuffle_row_groups=self.shuffle,
                                                          hdfs_driver='libhdfs',
                                                          schema_fields=self.schema_fields,
                                                          storage_options=self.storage_options)
                    if self.has_val:
                        self.val_reader = make_batch_reader(self.val_dir, num_epochs=self.num_reader_epochs,
                                                            cur_shard=self.cur_shard, shard_count=self.shard_count,
                                                            hdfs_driver='libhdfs',
                                                            schema_fields=self.schema_fields,
                                                            storage_options=self.storage_options)

            def teardown(self, stage=None):
                if stage == "fit" or stage is None:
                    if self.verbose:
                        print("Tear down petastorm readers")
                    self.train_reader.stop()
                    self.train_reader.join()
                    if self.has_val:
                        self.val_reader.stop()
                        self.val_reader.join()

            def train_dataloader(self):
                if self.verbose:
                    print("Setup train dataloader")
                kwargs = dict(reader=self.train_reader, batch_size=self.train_batch_size,
                              name="train dataloader",
                              shuffling_queue_capacity=0,
                              limit_step_per_epoch=self.steps_per_epoch_train,
                              verbose=self.verbose)
                return PytorchAsyncDataLoader(**kwargs)

            def val_dataloader(self):
                if not self.has_val:
                    return None
                if self.verbose:
                    print("setup val dataloader")
                kwargs = dict(reader=self.val_reader, batch_size=self.val_batch_size,
                              name="val dataloader",
                              shuffling_queue_capacity = 0,
                              limit_step_per_epoch=self.steps_per_epoch_val,
                              verbose=self.verbose)
                return PytorchAsyncDataLoader(**kwargs)

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)
            model = create_xor_model()

            with local_store() as store:
                torch_estimator = hvd_spark.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    input_shapes=[[-1, 2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    validation=0.2,
                    batch_size=4,
                    epochs=2,
                    data_module=CustomDataModule,
                    verbose=2,
                    trainer_args={"num_sanity_val_steps": 0})

                torch_model = torch_estimator.fit(df)

                # TODO: Find a way to pass log metrics from remote, and assert base on the logger.
                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    """
    Test override trainer args.
    """
    def test_model_override_trainer_args(self):
        if skip_lightning_tests:
            self.skipTest('Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x: '
                          'https://github.com/horovod/horovod/pull/3263')

        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

        with spark_session('test_fit_model') as spark:
            df = create_noisy_xor_data(spark)
            model = create_xor_model()

            with tempdir() as dir:

                with local_store() as store:
                    torch_estimator = hvd_spark.TorchEstimator(
                        num_proc=2,
                        store=store,
                        model=model,
                        input_shapes=[[-1, 2]],
                        feature_cols=['features'],
                        label_cols=['y'],
                        validation=0.2,
                        batch_size=4,
                        epochs=2,
                        verbose=2,
                        trainer_args={'stochastic_weight_avg': True})

                    torch_model = torch_estimator.fit(df)

                    # TODO: Find a way to pass log metrics from remote, and assert base on the logger.
                    trained_model = torch_model.getModel()
                    pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                    assert len(pred) == 1
                    assert pred.dtype == torch.float32

def check_fail(dir, rank, epoch, batch):
    if dir:
        fail = os.path.sep.join([dir, 'rank_{}_epoch_{}_batch_{}_fail'.format(rank, epoch, batch)])
        if os.path.exists(fail):
            logging.info('rank %s: failing epoch %s batch %s', rank, epoch, batch)
            os.unlink(fail)
            raise Exception('training failed, restart the task')


def fn(batches_per_commit, batches_per_epoch, epochs, dir=None):
    @run
    def train(state, dir):
        state.rendezvous += 1
        logging.info('rank %s: rendezvous %s', hvd.rank(), state.rendezvous)

        for state.epoch in range(state.epoch, epochs):
            logging.info('rank %s: start epoch %s at batch %s', hvd.rank(), state.epoch, state.batch)

            for state.batch in range(state.batch, batches_per_epoch):
                check_fail(dir, hvd.rank(), state.epoch, state.batch)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                # TODO: this sleep makes the fault tolerant test fail
                #       torch all gather throws an RuntimeError which should be a HorovodInternalError
                #import time
                #time.sleep(0.2)

                if state.batch % batches_per_commit == 0:
                    logging.info('rank %s: allgather', hvd.rank())
                    hvd.allgather(torch.tensor([hvd.rank(), state.epoch, state.batch, state.rendezvous]), 'state').tolist()
                    logging.info('rank %s: commit epoch %s batch %s', hvd.rank(), state.epoch, state.batch)
                    state.commits += 1
                    state.commit()

            logging.info('rank %s: allgather', hvd.rank())
            hvd.allgather(torch.tensor([hvd.rank(), state.epoch, state.batch, state.rendezvous]), 'state').tolist()
            logging.info('rank %s: commit epoch %s', hvd.rank(), state.epoch)
            state.commits += 1
            state.commit()
            state.batch = 0

        res = hvd.allgather(torch.tensor([hvd.rank(), state.epoch, state.batch, state.rendezvous]), 'state').tolist()
        logging.info('rank %s: returning', hvd.rank())
        return res, hvd.rank()

    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)-15s %(levelname)1.1s %(filename)s:%(lineno)d %(funcName)s() - %(message)s')

    hvd.init()

    batch_size = 32
    data = torch.randn(batch_size, 2)
    target = torch.LongTensor(batch_size).random_() % 2

    v = 1.0
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model.load_state_dict({
        '0.weight': torch.tensor([[v, v], [v, v]]),
        '0.bias': torch.tensor([v, v])
    })

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0, commits=0, rendezvous=0)
    return train(state, dir)
