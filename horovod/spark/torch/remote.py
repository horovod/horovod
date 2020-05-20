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

import contextlib
import io
import math
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning import Trainer

from horovod.spark.common import constants
from horovod.spark.common.util import _get_assigned_gpu_or_default, to_list
from horovod.spark.common.store import DBFSLocalStore
from horovod.spark.torch.util import deserialize_fn

PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER
METRIC_PRINT_FREQUENCY = constants.METRIC_PRINT_FREQUENCY
TOTAL_BUFFER_MEMORY_CAP_GIB = constants.TOTAL_BUFFER_MEMORY_CAP_GIB
BYTES_PER_GIB = constants.BYTES_PER_GIB
CUSTOM_SPARSE = constants.CUSTOM_SPARSE


def RemoteTrainer(estimator, metadata, last_checkpoint_state, run_id, dataset_idx, train_rows, val_rows, avg_row_size):
    # Estimator parameters
    input_shapes = estimator.getInputShapes()
    label_shapes = estimator.getLabelShapes()
    feature_columns = estimator.getFeatureCols()
    label_columns = estimator.getLabelCols()
    num_labels = len(label_columns)
    should_validate = estimator.getValidation()
    batch_size = estimator.getBatchSize()
    epochs = estimator.getEpochs()
    train_steps_per_epoch = estimator.getTrainStepsPerEpoch()
    validation_steps_per_epoch = estimator.getValidationStepsPerEpoch()
    sample_weight_col = estimator.getSampleWeightCol()
    metric_fn_groups = estimator.getMetrics()
    user_shuffle_buffer_size = estimator.getShufflingBufferSize()
    user_verbose = estimator.getVerbose()
    train_minibatch_fn = estimator.getTrainMinibatchFn()
    train_minibatch = train_minibatch_fn if train_minibatch_fn else _train_minibatch_fn()
    loss_fns_pre_train = to_list(estimator.getLoss(), num_labels)
    loss_constructors = to_list(estimator.getLossConstructors(), num_labels)
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None

    # If loss weight is not provided, use equal loss for all the labels
    loss_weights = estimator.getLossWeights()
    if not loss_weights:
        loss_weights = [float(1) / num_labels for _ in range(num_labels)]
    else:
        if not isinstance(loss_weights, list) or \
                len(loss_weights) != len(label_columns):
            raise ValueError('loss_weights needs to be a list with the same '
                             'length as the label_columns.')

    # Data reader parameters
    train_reader_worker_count = estimator.getTrainReaderNumWorker()
    val_reader_worker_count = estimator.getValReaderNumWorker()

    # Utility functions
    deserialize = deserialize_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn(
        train_rows, avg_row_size, user_shuffle_buffer_size)

    schema_fields = feature_columns + label_columns
    make_petastorm_reader = _make_petastorm_reader_fn(transformation, schema_fields,
                                                      batch_size, calculate_shuffle_buffer_size)

    # Storage
    store = estimator.getStore()
    remote_store = store.to_remote(run_id, dataset_idx)
    is_dbfs = isinstance(store, DBFSLocalStore)

    def train(serialized_model):
        model = deserialize(serialized_model)
        trainer = Trainer(distributed_backend='horovod', gpus=1 if torch.cuda.is_available() else 0,
                          max_epochs=epochs)

        with make_petastorm_reader(trainer, model, remote_store.train_data_path, 'train_dataloader',
                                   train_reader_worker_count), \
                make_petastorm_reader(trainer, model, remote_store.val_data_path, 'val_dataloader',
                                      val_reader_worker_count, should_validate):

            trainer.fit(model)

        serialized_checkpoint = io.BytesIO()
        torch.save({'model': model.state_dict()}, serialized_checkpoint)
        serialized_checkpoint.seek(0)
        return serialized_checkpoint
    return train


def _make_petastorm_reader_fn(transformation, schema_fields, batch_size, calculate_shuffle_buffer_size):
    @contextlib.contextmanager
    def make_petastorm_reader(trainer, model, data_path, dataloader_attr, reader_worker_count, should_read=True):
        from petastorm import TransformSpec, make_reader, make_batch_reader
        from petastorm.pytorch import BatchedDataLoader
        import horovod.torch as hvd

        if not should_read or trainer.is_overridden(dataloader_attr, model):
            yield
            return

        transform_spec = TransformSpec(transformation) if transformation else None

        # In general, make_batch_reader is faster than make_reader for reading the dataset.
        # However, we found out that make_reader performs data transformations much faster than
        # make_batch_reader with parallel worker processes. Therefore, the default reader
        # we choose is make_batch_reader unless there are data transformations.
        reader_factory_kwargs = dict()
        if transform_spec:
            reader_factory = make_reader
            reader_factory_kwargs['pyarrow_serialize'] = True
        else:
            reader_factory = make_batch_reader

        # Petastorm: read data from the store with the correct shard for this rank
        # setting num_epochs=None will cause an infinite iterator
        # and enables ranks to perform training and validation with
        # unequal number of samples
        with reader_factory(data_path,
                            num_epochs=1,
                            cur_shard=hvd.rank(),
                            reader_pool_type='process',
                            workers_count=reader_worker_count,
                            shard_count=hvd.size(),
                            hdfs_driver=PETASTORM_HDFS_DRIVER,
                            schema_fields=schema_fields,
                            transform_spec=transform_spec,
                            **reader_factory_kwargs) as reader:
            def dataloader_fn():
                return BatchedDataLoader(reader, batch_size=batch_size,
                                         shuffling_queue_capacity=calculate_shuffle_buffer_size())
            try:
                print('PATCH: {} {}'.format(dataloader_attr, dataloader_fn.__code__))
                setattr(model, dataloader_attr, dataloader_fn)
                yield
            finally:
                setattr(model, dataloader_attr, None)
    return make_petastorm_reader


def _train_minibatch_fn():
    def train_minibatch(model, optimizer, transform_outputs, loss_fn, inputs, labels, sample_weights):
        optimizer.zero_grad()
        outputs = model(*inputs)
        outputs, labels = transform_outputs(outputs, labels)
        loss = loss_fn(outputs, labels, sample_weights)
        loss.backward()
        optimizer.step()
        return outputs, loss
    return train_minibatch


def _get_optimizer_with_unscaled_lr_fn():
    def get_optimizer_with_unscaled_lr(hvd, current_optimizer, optimizer_cls, model):
        optimizer_state = current_optimizer.state_dict()
        # scale down the learning rate with the number of horovod workers
        for i in range(len(optimizer_state['param_groups'])):
            optimizer_state['param_groups'][i]['lr'] = \
                optimizer_state['param_groups'][i]['lr'] / hvd.size()
        optimizer = optimizer_cls(model.parameters(), lr=1)
        optimizer.load_state_dict(optimizer_state)
        return optimizer

    return get_optimizer_with_unscaled_lr


def _calculate_shuffle_buffer_size_fn(train_rows, avg_row_size, user_shuffle_buffer_size):
    def calculate_shuffle_buffer_size():
        """
        Determines the shuffling buffer size such that each worker gets at most 1GB for shuffling
        buffer such that on a single machine, among all the workers on that machine, at most
        memory_cap_gb GB are allocated for shuffling buffer. Also, it ensures that the buffer size
        is identical among all the workers.

        example 1:
        memory_cap_gb = 4
        machine1: 8 workers
        machine2: 3 workers
        shuffle_buffer_size = 0.5 GB

        example 2:
        memory_cap_gb = 4
            machine1: 2 workers
            machine2: 3 workers
        shuffle_buffer_size = 1 GB

        example 3:
        memory_cap_gb = 4
            machine1: 2 workers
            machine2: 8 workers
            machine3: 5 workers
        shuffle_buffer_size = 0.5 GB
        """
        import horovod.torch as hvd

        if user_shuffle_buffer_size:
            return user_shuffle_buffer_size

        local_size = hvd.local_size()
        local_sizes = hvd.allgather(torch.tensor([local_size]))
        max_local_size = torch.max(local_sizes).item()

        if max_local_size > TOTAL_BUFFER_MEMORY_CAP_GIB:
            shuffle_buffer_size = TOTAL_BUFFER_MEMORY_CAP_GIB * BYTES_PER_GIB / avg_row_size / max_local_size
        else:
            shuffle_buffer_size = BYTES_PER_GIB / avg_row_size
        return int(min(shuffle_buffer_size, train_rows / hvd.size()))

    return calculate_shuffle_buffer_size


def _construct_metric_value_holders_fn():
    def construct_metric_value_holders(metric_class, metric_fn_groups, label_columns, hvd):
        metric_values = []
        for group_number, metric_group in enumerate(metric_fn_groups):
            metric_group_val = []
            for label_col in label_columns:
                metric_group_val.append(
                    metric_class('group_' + str(group_number) + '_' + label_col, hvd))

            metric_values.append(metric_group_val)
        return metric_values
    return construct_metric_value_holders


def _metric_cls():
    # Horovod: average metrics from distributed training.
    class Metric(object):
        def __init__(self, name, hvd):
            self.name = name
            self.sum = torch.tensor(0.)
            self.n = torch.tensor(0.)
            self.hvd = hvd

        def update(self, val):
            self.sum += self.hvd.allreduce(val.detach().cpu(), name=self.name)
            self.n += 1

        @property
        def avg(self):
            return self.sum / self.n

    return Metric


def _prepare_np_data_fn():
    def prepare_np_data(rows, col_name, metadata):
        intermediate_format = metadata[col_name]['intermediate_format']
        if intermediate_format != CUSTOM_SPARSE:
            return rows

        shape = metadata[col_name]['shape']
        num_rows = rows.shape[0]
        dense_rows = torch.zeros([num_rows, shape])
        for r in range(num_rows):
            size = rows[r][0].long()
            dense_rows[r][rows[r][1:size + 1].long()] = \
                rows[r][size + 1:2 * size + 1]
        return dense_rows

    return prepare_np_data


def _get_metric_avgs_fn():
    def get_metric_avgs(metric_value_groups):
        all_metric_groups_values = []
        for metric_value_group in metric_value_groups:
            metric_avgs = {}
            for metric in metric_value_group:
                metric_avgs[metric.name] = metric.avg.item()
            all_metric_groups_values.append(metric_avgs)
        return all_metric_groups_values

    return get_metric_avgs


def _update_metrics_fn(metric_fn_groups):
    def update_metrics(metric_value_groups, outputs, labels):
        """
        metric_value_groups is a list of metric functions. For example, for a model with 3
        outputs, we can define these two metric groups
        [
            [metric_fn1],
            [metric_fn21,metric_fn22,metric_fn23],
        ]

        In this example, first metric group provides only one metric function. This
        function will be used to calculate the metric on all of the model outputs. Second
        metric groups, however, defines one metric function per output.
        """

        num_outputs = len(outputs)
        for metric_fn_group, metric_value_group in zip(metric_fn_groups, metric_value_groups):
            if len(metric_fn_group) == 1:
                _metric_fn_group = [metric_fn_group[0] for _ in range(num_outputs)]
            else:
                _metric_fn_group = metric_fn_group

            for metric_val, metric_fn, output_group, label_group in \
                    zip(metric_value_group, _metric_fn_group, outputs, labels):
                metric_val.update(metric_fn(output_group, label_group))

        return metric_value_groups

    return update_metrics


def _write_metrics_summary_fn():
    def write_metrics_summary(stage, epoch, loss_metric, metric_value_groups, log_writer):
        if not log_writer:
            return

        log_writer.add_scalar('{}/{}'.format(stage, loss_metric.name),
                              loss_metric.avg.item(), epoch)

        for idx, metric_value_group in enumerate(metric_value_groups):
            for metric in metric_value_group:
                log_writer.add_scalar('{}/{}:{}'.format(stage, metric.name, idx),
                                      metric.avg.item(), epoch)

    return write_metrics_summary


def _calculate_loss_fn():
    def calculate_loss(outputs, labels, loss_weights, loss_fns, sample_weights=None):
        if sample_weights is not None:
            # when reduction='none', loss function returns the value of all the losses
            # from all the samples. We multiply each sample's weight to its loss and
            # then take the mean of the weight adjusted losses from all the samples in the
            # batch. Note that this approach is not "weighted average" because the sum of
            # the sample weights in each batch does not necessarily add up to one. If we add
            # the weights and divide the sum to the sum of weights, the impact of two
            # samples with identical weights but in different batches will not be equal on
            # the calculated gradients.
            losses = []
            for output, label, loss_fn, loss_weight in zip(outputs, labels,
                                                           loss_fns, loss_weights):
                weight_adjusted_sample_losses = \
                    loss_fn(output, label, reduction='none').flatten() * sample_weights
                output_loss = weight_adjusted_sample_losses.mean()
                losses.append(output_loss * loss_weight)
        else:
            losses = [loss_fn(output, label) * loss_weight for
                      output, label, loss_fn, loss_weight in
                      zip(outputs, labels, loss_fns, loss_weights)]

        loss = sum(losses)
        return loss

    return calculate_loss
