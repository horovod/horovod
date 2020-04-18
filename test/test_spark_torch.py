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

import sys
import unittest
import warnings

import numpy as np

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import DoubleType, LongType

import mock
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

from horovod.run.runner import is_gloo_used
import horovod.spark.torch as hvd
from horovod.spark.common import constants, util
from horovod.spark.torch import remote
from horovod.spark.torch.estimator import EstimatorParams, _torch_param_serialize

from spark_common import CallbackBackend, create_xor_data, local_store, spark_session


class XOR(nn.Module):
    def __init__(self, input_dim, output_dim):
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


def create_xor_model(input_dim=2, output_dim=1):
    return XOR(input_dim, output_dim)


class SparkTorchTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(SparkTorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_fit_model(self):
        if sys.version_info < (3, 0, 0) and is_gloo_used():
            self.skipTest('Horovod on Spark over Gloo only supported on Python3')

        model = create_xor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = F.binary_cross_entropy

        with spark_session('test_fit_model') as spark:
            df = create_xor_data(spark)

            with local_store() as store:
                torch_estimator = hvd.TorchEstimator(
                    num_proc=2,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    input_shapes=[[2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    batch_size=1,
                    epochs=3,
                    verbose=2,
                    sample_weight_col='weight')

                torch_model = torch_estimator.fit(df)

                trained_model = torch_model.getModel()
                pred = trained_model(torch.ones([1, 2], dtype=torch.int32))
                assert len(pred) == 1
                assert pred.dtype == torch.float32

    def test_restore_from_checkpoint(self):
        model = create_xor_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = nn.BCELoss()

        with spark_session('test_restore_from_checkpoint') as spark:
            df = create_xor_data(spark)

            ctx = CallbackBackend()

            run_id = 'run01'
            with local_store() as store:
                torch_estimator = hvd.TorchEstimator(
                    backend=ctx,
                    store=store,
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    input_shapes=[[2]],
                    feature_cols=['features'],
                    label_cols=['y'],
                    batch_size=1,
                    epochs=1,
                    verbose=2,
                    run_id=run_id)

                torch_estimator._load_checkpoint = mock.Mock(side_effect=torch_estimator._load_checkpoint)

                ckpt_path = store.get_checkpoint_path(run_id)
                assert not store.exists(ckpt_path)
                torch_estimator._load_checkpoint.assert_not_called()
                torch_estimator.fit(df)

                assert store.exists(ckpt_path)
                torch_estimator.fit(df)
                torch_estimator._load_checkpoint.assert_called()

    def test_transform_multi_class(self):
        model = create_xor_model(output_dim=2)

        with spark_session('test_transform_multi_class') as spark:
            df = create_xor_data(spark)
            metadata = util._get_metadata(df)

            torch_model = hvd.TorchModel(history=None,
                                         model=model,
                                         input_shapes=[[2]],
                                         feature_columns=['features'],
                                         label_columns=['y'],
                                         _metadata=metadata)
            out_df = torch_model.transform(df)

            expected_types = {
                'x1': LongType,
                'x2': LongType,
                'features': VectorUDT,
                'weight': DoubleType,
                'y': DoubleType,
                'y__output': VectorUDT
            }

            for field in out_df.schema.fields:
                assert type(field.dataType) == expected_types[field.name]

    def test_pytorch_get_optimizer_with_unscaled_lr(self):
        hvd_size = 4
        init_learning_rate = 0.001
        hvd_mock = mock.MagicMock()
        hvd_mock.size.return_value = hvd_size

        get_optimizer_with_unscaled_lr_fn = remote._get_optimizer_with_unscaled_lr_fn()
        model = create_xor_model()
        current_optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
        optimizer_cls = current_optimizer.__class__
        opt_unscaled_lr = get_optimizer_with_unscaled_lr_fn(hvd_mock, current_optimizer,
                                                            optimizer_cls, model)

        optimizer_state = opt_unscaled_lr.state_dict()
        for i in range(len(optimizer_state['param_groups'])):
            assert optimizer_state['param_groups'][i]['lr'] == init_learning_rate / hvd_size

    def test_calculate_shuffle_buffer_size_small_row_size(self):
        hvd_size = 4
        local_size = 2
        hvd_mock = mock.MagicMock()
        hvd_mock.local_size = lambda: local_size
        hvd_mock.allgather = lambda x: torch.tensor([local_size for _ in range(hvd_size)])

        avg_row_size = 100
        train_row_count_per_worker = 100

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn()
        shuffle_size = calculate_shuffle_buffer_size(hvd_mock, avg_row_size, train_row_count_per_worker)
        assert shuffle_size == train_row_count_per_worker

    def test_calculate_shuffle_buffer_size(self):
        # case with 2 workers, one with 5 ranks and second with 3 ranks
        hvd_mock = mock.MagicMock()
        hvd_mock.allgather = lambda x: torch.tensor([5, 5, 5, 5, 5, 3, 3, 3])
        hvd_mock.local_size = lambda: 2

        avg_row_size = 100000
        train_row_count_per_worker = 1000000

        calculate_shuffle_buffer_size = remote._calculate_shuffle_buffer_size_fn()
        shuffle_size = calculate_shuffle_buffer_size(hvd_mock, avg_row_size, train_row_count_per_worker)

        assert int(shuffle_size) == \
               int(constants.TOTAL_BUFFER_MEMORY_CAP_GIB * constants.BYTES_PER_GIB / avg_row_size / 5)

    def test_metric_class(self):
        hvd_mock = mock.MagicMock()
        hvd_mock.allreduce = lambda tensor, name: 2 * tensor
        hvd_mock.local_size = lambda: 2

        metric_class = remote._metric_cls()
        metric = metric_class('dummy_metric', hvd_mock)
        metric.update(torch.tensor(1.0))
        metric.update(torch.tensor(2.0))

        assert metric.sum.item() == 6.0
        assert metric.n.item() == 2.0
        assert metric.avg.item() == 6.0 / 2.0

    def test_construct_metric_value_holders_one_metric_for_all_labels(self):
        hvd_mock = mock.MagicMock()
        hvd_mock.allreduce = lambda tensor, name: 2 * tensor
        hvd_mock.local_size = lambda: 2
        metric_class = remote._metric_cls()

        def torch_dummy_metric(outputs, labels):
            count = torch.tensor(0.)
            for output, label in zip(outputs, labels):
                count += 1
            return count

        metric_fn_groups = [[torch_dummy_metric], [torch_dummy_metric]]
        label_columns = ['l1', 'l2']

        construct_metric_value_holders = remote._construct_metric_value_holders_fn()
        metric_values = construct_metric_value_holders(metric_class, metric_fn_groups, label_columns,
                                                       hvd_mock)

        assert metric_values[0][0].name == 'group_0_l1'
        assert metric_values[0][1].name == 'group_0_l2'
        assert metric_values[1][0].name == 'group_1_l1'
        assert metric_values[1][1].name == 'group_1_l2'

    def test_prepare_np_data(self):
        with spark_session('test_prepare_np_data') as spark:
            df = create_xor_data(spark)

            train_rows = df.count()
            schema_cols = ['features', 'y']
            metadata = util._get_metadata(df)
            assert metadata['features']['intermediate_format'] == constants.ARRAY

            to_petastorm = util.to_petastorm_fn(schema_cols, metadata)
            modified_df = df.rdd.map(to_petastorm).toDF()
            data = modified_df.collect()

            prepare_np_data = remote._prepare_np_data_fn()
            features = torch.tensor([data[i].features for i in range(train_rows)])
            features_prepared = prepare_np_data(features, 'features', metadata)
            assert np.array_equal(features_prepared, features)

    def test_get_metric_avgs(self):
        get_metric_avgs = remote._get_metric_avgs_fn()

        def _generate_mock_metric(name, val):
            metric = mock.MagicMock()
            metric.name = name
            metric.avg.item.return_value = val
            return metric

        metric11 = _generate_mock_metric('11', 11)
        metric12 = _generate_mock_metric('12', 12)
        metric21 = _generate_mock_metric('21', 21)
        metric22 = _generate_mock_metric('22', 22)

        metric_value_groups = [[metric11, metric12], [metric21, metric22]]
        all_metric_groups_values = get_metric_avgs(metric_value_groups)

        assert all_metric_groups_values[0]['11'] == 11
        assert all_metric_groups_values[0]['12'] == 12
        assert all_metric_groups_values[1]['21'] == 21
        assert all_metric_groups_values[1]['22'] == 22

    def test_update_metrics(self):
        def dummy_metric_add(output, label):
            return output + label

        def dummy_metric_sub(output, label):
            return output - label

        metric_fn_groups = [[dummy_metric_add, dummy_metric_sub], [dummy_metric_add]]

        update_metrics = remote._update_metrics_fn(metric_fn_groups)

        def _generate_mock_metric(name, val):
            metric = mock.MagicMock()
            metric.name = name
            metric.avg.item.return_value = val
            return metric

        metric11 = _generate_mock_metric('11', 11)
        metric12 = _generate_mock_metric('12', 12)
        metric21 = _generate_mock_metric('21', 21)
        metric22 = _generate_mock_metric('22', 22)

        metric_value_groups = [[metric11, metric12], [metric21, metric22]]

        outputs = [15, 4]
        labels = [10, 2]

        updated_metric_value_groups = update_metrics(metric_value_groups, outputs, labels)

        updated_metric_value_groups[0][0].update.assert_called_once_with(25)
        updated_metric_value_groups[0][1].update.assert_called_once_with(2)
        updated_metric_value_groups[1][0].update.assert_called_once_with(25)
        updated_metric_value_groups[1][1].update.assert_called_once_with(6)

    def test_torch_param_serialize(self):
        serialized_backend = _torch_param_serialize(EstimatorParams.backend.name, 'dummy_value')
        assert serialized_backend is None

        serialized_store = _torch_param_serialize(EstimatorParams.store.name, 'dummy_value')
        assert serialized_store is None

        serialized_dummy_param = _torch_param_serialize('dummy_param_name', None)
        assert serialized_dummy_param is None

    def test_torch_direct_parquet_train(self):
        with spark_session('test_torch_direct_parquet_train') as spark:
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
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                    loss = nn.BCELoss()

                    est = hvd.TorchEstimator(
                        backend=backend,
                        store=store,
                        model=model,
                        optimizer=optimizer,
                        input_shapes=[[2]],
                        feature_cols=['features'],
                        label_cols=['y'],
                        batch_size=1,
                        epochs=3,
                        verbose=2)

                    # To make sure that setLoss works with non-list loss.
                    est.setLoss(loss)

                    transformer = est.fit_on_parquet()
                    predictions = transformer.transform(df)
                    assert predictions.count() == df.count()

    def test_calculate_loss_with_sample_weight(self):
        calculate_loss = remote._calculate_loss_fn()

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

        loss = calculate_loss(outputs, labels, [1], [fn_minus], sample_weights=torch.tensor([1.0, 6.0, 3.0]))
        assert loss == 5.0

        labels = torch.tensor([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        loss = calculate_loss(outputs, labels, [0.2, 0.8], [fn_minus, fn_add], sample_weights=torch.tensor([1.0, 6.0, 3.0]))
        assert loss == torch.tensor(9.0)

    def test_calculate_loss_without_sample_weight(self):
        calculate_loss = remote._calculate_loss_fn()

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

        loss = calculate_loss(outputs, labels, [1], [fn_minus])
        assert loss == 1.0

        labels = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]])
        outputs = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])

        loss = calculate_loss(outputs, labels, [0.2, 0.8], [fn_minus, fn_add])
        assert torch.isclose(loss, torch.tensor(2.6))
