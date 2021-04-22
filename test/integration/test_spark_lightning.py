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
import logging
import os
import sys
import unittest
import warnings

import mock
import pytest
import numpy as np
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import FloatType, IntegerType

import pytorch_lightning as pl

import horovod
import horovod.torch as hvd
from horovod.torch.elastic import run

import horovod.spark.lightning as hvd_spark
from horovod.spark.lightning import remote
from horovod.spark.lightning.estimator import EstimatorParams, _torch_param_serialize, MIN_PL_VERSION
from horovod.spark.lightning.legacy import to_lightning_module

from horovod.common.util import gloo_built, mpi_built
from horovod.runner.mpi_run import is_open_mpi
from horovod.spark.common import constants, util

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import tempdir, spawn, is_built
from spark_common import CallbackBackend, create_noisy_xor_data, create_xor_data, local_store, spark_session


class XOR(pl.LightningModule):
    def __init__(self, input_dim=2, output_dim=1):
        super(XOR, self).__init__()
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
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch['features'], batch['y'].unsqueeze(1)
        y_hat = self(x)
        return {'val_loss': F.binary_cross_entropy(y_hat, y.float())}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() if len(outputs) > 0 else float('inf')
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


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

@pytest.mark.skipif(LooseVersion(pl.__version__) < LooseVersion(MIN_PL_VERSION), reason='Pytorch lightning version is not supported.')
class SparkLightningTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SparkLightningTests, self).__init__(*args, **kwargs)
        logging.getLogger('py4j.java_gateway').setLevel(logging.INFO)
        warnings.simplefilter('module')

    def test_fit_model(self):
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
                    verbose=2)

                torch_model = torch_estimator.fit(df)

                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    def test_legacy_fit_model(self):
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

    # TODO: Add this test back after checkpoint call back is supported
    def test_restore_from_checkpoint(self):
        self.skipTest('There is a bug in current lightning version for checkpoint'
                      'call back. Will add this test back when it is solved.')

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
                    run_id=run_id)

                torch_estimator._read_checkpoint = mock.Mock(side_effect=torch_estimator._read_checkpoint)

                ckpt_path = store.get_checkpoint_path(run_id)
                assert not store.exists(ckpt_path)
                torch_estimator._read_checkpoint.assert_not_called()
                torch_estimator.fit(df)

                assert store.exists(ckpt_path)
                torch_estimator.fit(df)
                torch_estimator._read_checkpoint.assert_called()

    #TODO: Add this test back after checkpoint call back is supported
    def test_legacy_restore_from_checkpoint(self):
        self.skipTest('There is a bug in current lightning version for checkpoint'
                      'call back. Will add this test back when it is solved.')

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

                torch_estimator._read_checkpoint = mock.Mock(side_effect=torch_estimator._read_checkpoint)

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

    @mock.patch('horovod.torch.allgather')
    @mock.patch('horovod.torch.local_size')
    def test_calculate_shuffle_buffer_size_small_row_size(self, mock_local_size, mock_allgather):
        import horovod.torch as hvd
        hvd.init()

        hvd_size = 4
        local_size = 2
        mock_local_size.return_value = local_size
        mock_allgather.return_value = torch.tensor([local_size for _ in range(hvd_size)])

        avg_row_size = 100
        train_row_count_per_worker = 100

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn(
            train_row_count_per_worker, avg_row_size, None)
        shuffle_size = calculate_shuffle_buffer_size()
        assert shuffle_size == train_row_count_per_worker

    @mock.patch('horovod.torch.allgather')
    @mock.patch('horovod.torch.local_size')
    def test_calculate_shuffle_buffer_size(self, mock_local_size, mock_allgather):
        import horovod.torch as hvd
        hvd.init()

        # case with 2 workers, one with 5 ranks and second with 3 ranks
        mock_allgather.return_value = torch.tensor([5, 5, 5, 5, 5, 3, 3, 3])
        mock_local_size.return_value = 2

        avg_row_size = 100000
        train_row_count_per_worker = 1000000

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn(
            train_row_count_per_worker, avg_row_size, None)
        shuffle_size = calculate_shuffle_buffer_size()

        actual = int(shuffle_size)
        expected = int(constants.TOTAL_BUFFER_MEMORY_CAP_GIB * constants.BYTES_PER_GIB / avg_row_size / 5)
        assert actual == expected

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
        with spark_session('test_direct_parquet_train') as spark:
            df = create_noisy_xor_data(spark)

            backend = CallbackBackend()
            with local_store() as store:
                store.get_train_data_path = lambda v=None: store._train_path
                store.get_val_data_path = lambda v=None: store._val_path

                with util.prepare_data(backend.num_processes(),
                                       store,
                                       df,
                                       feature_columns=['features'],
                                       label_columns=['y'],
                                       validation=0.2):
                    model = create_xor_model()

                    for inmemory_cache_all in [False, True]:
                        est = hvd_spark.TorchEstimator(
                            backend=backend,
                            store=store,
                            model=model,
                            input_shapes=[[-1, 2]],
                            feature_cols=['features'],
                            label_cols=['y'],
                            validation=0.2,
                            batch_size=1,
                            epochs=3,
                            verbose=2,
                            inmemory_cache_all=inmemory_cache_all)

                        transformer = est.fit_on_parquet()
                        predictions = transformer.transform(df)
                        assert predictions.count() == df.count()

    def test_legacy_calculate_loss_with_sample_weight(self):
        labels = torch.tensor([[1.0, 2.0, 3.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0]])

        def fn_minus(output, label, reduction=None):
            losses = label-output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        def fn_add(output, label, reduction=None):
            losses = label+output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        kwargs = dict(model=mock.Mock(), optimizer=mock.Mock(), feature_cols=[], sample_weights_col='', validation=0)
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
            losses = label-output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        def fn_add(output, label, reduction=None):
            losses = label+output
            if reduction == 'none':
                return losses
            else:
                return losses.mean()

        kwargs = dict(model=mock.Mock(), optimizer=mock.Mock(), feature_cols=[], sample_weights_col=None, validation=0)
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
        if not gloo_built():
            self.skipTest("Gloo is not available")

        with spark_session('test_happy_run_elastic'):
            res = horovod.spark.run_elastic(fn, args=(2, 5, 4),
                                            num_proc=2, min_np=2, max_np=2,
                                            start_timeout=10, verbose=2)
            self.assertListEqual([([0, 3, 0, 1, 1, 3, 0, 1], 0),
                                  ([0, 3, 0, 1, 1, 3, 0, 1], 1)], res)

    """
    Test that horovod.spark.run_elastic works properly in a fault-tolerant situation.
    """
    def test_happy_run_elastic_fault_tolerant(self):
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
                                                num_proc=2, min_np=2, max_np=2,
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
                                                num_proc=2, min_np=2, max_np=2,
                                                start_timeout=5, verbose=2)
                self.assertListEqual([([0, 4, 0, 4, 1, 4, 0, 4], 0),
                                      ([0, 4, 0, 4, 1, 4, 0, 4], 1)], res)


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
