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
from datetime import datetime, timezone

import torch
from torch.utils.tensorboard import SummaryWriter

from horovod.spark.common import constants
from horovod.spark.common.util import _get_assigned_gpu_or_default, to_list
from horovod.spark.common.store import DBFSLocalStore
from horovod.spark.torch.util import deserialize_fn

PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER
METRIC_PRINT_FREQUENCY = constants.METRIC_PRINT_FREQUENCY
TOTAL_BUFFER_MEMORY_CAP_GIB = constants.TOTAL_BUFFER_MEMORY_CAP_GIB
BYTES_PER_GIB = constants.BYTES_PER_GIB
CUSTOM_SPARSE = constants.CUSTOM_SPARSE


def RemoteTrainer(estimator, metadata, last_checkpoint_state, run_id, dataset_idx):
    # Estimator parameters
    gradient_compression = estimator.getGradientCompression()
    input_shapes = estimator.getInputShapes()
    label_shapes = estimator.getLabelShapes()
    feature_columns = estimator.getFeatureCols()
    label_columns = estimator.getLabelCols()
    num_labels = len(label_columns)
    should_validate = estimator.getValidation()
    batch_size = estimator.getBatchSize()
    val_batch_size = estimator.getValBatchSize() if estimator.getValBatchSize() else batch_size
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
    inmemory_cache_all = estimator.getInMemoryCacheAll()

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
    reader_pool_type = estimator.getReaderPoolType()

    # Utility functions
    deserialize = deserialize_fn()
    get_optimizer_with_unscaled_lr = _get_optimizer_with_unscaled_lr_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn()
    construct_metric_value_holders = _construct_metric_value_holders_fn()
    metric_cls = _metric_cls()
    prepare_np_data = _prepare_np_data_fn()
    get_metric_avgs = _get_metric_avgs_fn()
    update_metrics = _update_metrics_fn(metric_fn_groups)
    write_metrics_summary = _write_metrics_summary_fn()
    calculate_loss = _calculate_loss_fn()

    # Storage
    store = estimator.getStore()
    remote_store = store.to_remote(run_id, dataset_idx)
    is_dbfs = isinstance(store, DBFSLocalStore)
    storage_options = store.storage_options

    @contextlib.contextmanager
    def empty_batch_reader():
        yield None

    def train(serialized_model, optimizer_cls, model_opt_state_serialized,
              train_rows, val_rows, avg_row_size):
        from petastorm import TransformSpec, make_reader, make_batch_reader
        from petastorm.pytorch import BatchedDataLoader, InMemBatchedDataLoader
        import torch
        import horovod.torch as hvd

        # Deserializing objects
        model_opt_state = torch.load(model_opt_state_serialized)
        model = deserialize(serialized_model)

        if loss_fns_pre_train:
            loss_fns = loss_fns_pre_train
        if loss_constructors:
            local_vars = locals()
            loss_fns = [loss_constructor(**local_vars) for loss_constructor in loss_constructors]

        # Horovod: initialize library.
        hvd.init()

        if user_verbose:
            import horovod as _horovod
            print(f"Shared lib path is pointing to: {_horovod.common.process_sets._basics.MPI_LIB_CTYPES}")

        # If user specifies any user_shuffle_buffer_size (even 0), we should honor it.
        if user_shuffle_buffer_size is None:
            shuffle_buffer_size = \
                calculate_shuffle_buffer_size(hvd, avg_row_size, train_rows / hvd.size())
        else:
            if user_shuffle_buffer_size < 0:
                raise ValueError("user_shuffle_buffer_size cannot be negative!")
            shuffle_buffer_size = user_shuffle_buffer_size

        cuda_available = torch.cuda.is_available()
        # We need to check all ranks have same device type for traning.
        # Horovod doesn't support heterogeneous allreduce for gradients.
        cuda_avail_list = hvd.allgather_object(cuda_available, name='device type')
        if cuda_avail_list.count(cuda_available) != hvd.size():
            raise RuntimeError("All ranks don't have same device type!")

        if cuda_available:
            # Horovod: pin GPU to local rank or the assigned GPU from spark.
            torch.cuda.set_device(_get_assigned_gpu_or_default(default=hvd.local_rank()))
            # Move model to GPU.
            model.cuda()

        # Optimizer object needs to be re-instantiated. Internally, it uses memory addresses of
        # objects as their identity and therefore it cannot be serialized and then
        # deserialized. The deserialized optimizer object stores the names of the parameters
        # with their old memory addresses but in reality those are different than the
        # reconstructed deserialized object and that creates problem.
        # Learning rate is a required parameters in SGD optimizer. It will be overridden with
        # load_state_dict.
        optimizer = optimizer_cls(model.parameters(), lr=1)
        optimizer_state = model_opt_state['optimizer']

        if last_checkpoint_state is not None:
            model.load_state_dict(last_checkpoint_state['model'])
            optimizer.load_state_dict(last_checkpoint_state['optimizer'])
        else:
            # scale the learning rate with the number of horovod workers
            for i in range(len(optimizer_state['param_groups'])):
                optimizer_state['param_groups'][i]['lr'] = \
                    optimizer_state['param_groups'][i]['lr'] * hvd.size()

            optimizer.load_state_dict(optimizer_state)

        # Horovod: broadcast parameters & optimizer state.
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

        for group in optimizer.param_groups:
            for p in group['params']:
                if id(p) not in optimizer.state_dict()['state']:
                    p.grad = p.data.new(p.size()).zero_()
        optimizer.step()
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        dist_optimizer_args = dict(optimizer=optimizer,
                                   named_parameters=model.named_parameters())
        if gradient_compression:
            # Pass the compression arg only if it is specified by the user.
            dist_optimizer_args['compression'] = gradient_compression
        # Horovod: wrap optimizer with DistributedOptimizer.
        optimizer = hvd.DistributedOptimizer(**dist_optimizer_args)

        # This function takes the current optimizer and constructs a new optimizer with the
        # same state except with learning rate scaled down with the number of horovod workers.
        # This is important the retraining of the model. User may retrain the model with
        # different number of workers and we need the raw learning rate to adjust with the
        # new number of workers.

        transform_spec = None
        if transformation:
            transform_spec = TransformSpec(transformation)

        schema_fields = feature_columns + label_columns
        if sample_weight_col:
            schema_fields.append(sample_weight_col)

        if train_steps_per_epoch is None:
            steps_per_epoch = int(math.floor(float(train_rows) / batch_size / hvd.size()))
        else:
            steps_per_epoch = train_steps_per_epoch

        with remote_store.get_local_output_dir() as run_output_dir:
            logs_dir = os.path.join(run_output_dir, remote_store.logs_subdir)
            log_writer = SummaryWriter(logs_dir) if hvd.rank() == 0 else None
            ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)

            def save_checkpoint():
                model.cpu()
                optimizer_with_scaled_down_lr = \
                    get_optimizer_with_unscaled_lr(hvd, optimizer, optimizer_cls, model)
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer_with_scaled_down_lr.state_dict(),
                }
                torch.save(state, ckpt_file)
                if cuda_available:
                    model.cuda()

            if hvd.rank() == 0 and user_verbose:
                print(f"Training parameters: Epochs: {epochs}\n"
                      f"Train rows: {train_rows}, Train batch size: {batch_size}, Train_steps_per_epoch: {steps_per_epoch}\n"
                      f"Shuffle buffer size: {shuffle_buffer_size}\n"
                      f"Checkpoint file: {ckpt_file}, Logs dir: {logs_dir}\n")
            # In general, make_batch_reader is faster than make_reader for reading the dataset.
            # However, we found out that make_reader performs data transformations much faster than
            # make_batch_reader with parallel worker processes. Therefore, the default reader
            # we choose is make_batch_reader unless there are data transformations.
            reader_factory = None
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
            with reader_factory(remote_store.train_data_path,
                                num_epochs=None,
                                cur_shard=hvd.rank(),
                                reader_pool_type=reader_pool_type,
                                workers_count=train_reader_worker_count,
                                shard_count=hvd.size(),
                                hdfs_driver=PETASTORM_HDFS_DRIVER,
                                schema_fields=schema_fields,
                                transform_spec=transform_spec,
                                storage_options=storage_options,
                                # Don't shuffle row groups without shuffling.
                                shuffle_row_groups=True if shuffle_buffer_size > 0 else False
                                **reader_factory_kwargs) as train_reader:
                with reader_factory(remote_store.val_data_path,
                                    num_epochs=None,
                                    cur_shard=hvd.rank(),
                                    reader_pool_type=reader_pool_type,
                                    workers_count=val_reader_worker_count,
                                    shard_count=hvd.size(),
                                    hdfs_driver=PETASTORM_HDFS_DRIVER,
                                    schema_fields=schema_fields,
                                    transform_spec=transform_spec,
                                    storage_options=storage_options,
                                    shuffle_row_groups=False,
                                    **reader_factory_kwargs) \
                    if should_validate else empty_batch_reader() as val_reader:

                    if inmemory_cache_all:
                        # Petastorm introduced InMemBatchedDataLoader class in v0.11.0
                        train_loader = InMemBatchedDataLoader(train_reader,
                                                              batch_size=batch_size,
                                                              num_epochs=epochs,
                                                              rows_capacity=steps_per_epoch*batch_size,
                                                              shuffle=True)
                    else:
                        train_loader = BatchedDataLoader(train_reader,
                                                         batch_size=batch_size,
                                                         shuffling_queue_capacity=shuffle_buffer_size)
                    train_loader_iter = iter(train_loader)

                    def prepare_batch(row):
                        inputs = [
                            prepare_np_data(
                                row[col].float(), col, metadata).reshape(shape)
                            for col, shape in zip(feature_columns, input_shapes)]
                        labels = [
                            prepare_np_data(
                                row[col].float(), col, metadata)
                            for col in label_columns]

                        sample_weights = row.get(sample_weight_col, None)
                        if sample_weights is not None:
                            sample_weights = sample_weights.float()
                        if cuda_available:
                            inputs = [input.cuda() for input in inputs]
                            labels = [label.cuda() for label in labels]
                            if sample_weights is not None:
                                sample_weights = sample_weights.cuda()
                        return inputs, labels, sample_weights

                    def transform_outputs(outputs, labels):
                        if not isinstance(outputs, tuple) and not isinstance(outputs,  list):
                            outputs = [outputs]

                        # reshape labels to match the output shape of the model
                        if hasattr(outputs[0], 'shape'):
                            if label_shapes:
                                labels = [label.reshape(label_shape)
                                          for label, label_shape in zip(labels, label_shapes)]
                            else:
                                # If label_shapes parameter is not provided, reshape the label
                                # columns data to match the shape of the model output
                                labels = [label.reshape(output.shape) if
                                          output.shape.numel() == label.shape.numel() else label
                                          for label, output in zip(labels, outputs)]

                        return outputs, labels

                    def aggregate_metrics(stage, epoch, loss, metric_value_groups):
                        all_metric_groups_values = get_metric_avgs(metric_value_groups)
                        if remote_store.saving_runs:
                            write_metrics_summary(
                                stage, epoch, loss, all_metric_groups_values, log_writer)
                        return {
                            loss.name: loss.avg.item(),
                            'all_metrics': all_metric_groups_values
                        }

                    def loss_fn(outputs, labels, sample_weights):
                        loss = calculate_loss(outputs, labels, loss_weights, loss_fns, sample_weights)
                        return loss

                    def print_metrics(batch_idx, loss, metric_value_groups, phase):
                        if user_verbose > 0 and hvd.rank() == 0 and \
                                batch_idx % METRIC_PRINT_FREQUENCY == 0:
                            print("{phase}\tepoch:\t{epoch}\tstep\t{batch_idx}:\t{metrics}".
                                  format(phase=phase,
                                         epoch=epoch,
                                         batch_idx=batch_idx,
                                         metrics=aggregate_metrics(phase, epoch, loss,
                                                                   metric_value_groups)))

                    def _train(epoch):
                        model.train()
                        train_loss = metric_cls('loss', hvd)
                        metric_value_groups = construct_metric_value_holders(
                            metric_cls, metric_fn_groups, label_columns, hvd)

                        # iterate on one epoch
                        for batch_idx in range(steps_per_epoch):
                            row = next(train_loader_iter)
                            inputs, labels, sample_weights = prepare_batch(row)
                            outputs, loss = train_minibatch(model, optimizer, transform_outputs,
                                                            loss_fn, inputs, labels, sample_weights)
                            update_metrics(metric_value_groups, outputs, labels)
                            train_loss.update(loss)
                            print_metrics(batch_idx, train_loss, metric_value_groups, 'train')

                        return aggregate_metrics('train', epoch, train_loss, metric_value_groups)

                    if should_validate:
                        if validation_steps_per_epoch is None:
                            validation_steps = int(math.ceil(float(val_rows) / val_batch_size / hvd.size()))
                        else:
                            validation_steps = validation_steps_per_epoch

                        if hvd.rank() == 0 and user_verbose:
                            print(f"Val rows: {val_rows}, Val batch size: {val_batch_size}, Val_steps_per_epoch: {validation_steps}\n")

                        if inmemory_cache_all:
                            # Petastorm introduced InMemBatchedDataLoader class in v0.11.0
                            val_loader = InMemBatchedDataLoader(val_reader,
                                                                batch_size=val_batch_size,
                                                                num_epochs=epochs,
                                                                rows_capacity=validation_steps*val_batch_size,
                                                                shuffle=False)
                        else:
                            val_loader = BatchedDataLoader(val_reader,
                                                           batch_size=val_batch_size,
                                                           shuffling_queue_capacity=0)
                        val_loader_iter = iter(val_loader)

                        def _validate(epoch):
                            model.eval()
                            val_loss = metric_cls('loss', hvd)

                            metric_value_groups = construct_metric_value_holders(
                                metric_cls, metric_fn_groups, label_columns, hvd)

                            # iterate on one epoch
                            for batch_idx in range(validation_steps):
                                row = next(val_loader_iter)
                                inputs, labels, sample_weights = prepare_batch(row)

                                outputs = model(*inputs)
                                outputs, labels = transform_outputs(outputs, labels)

                                loss = calculate_loss(
                                    outputs, labels, loss_weights, loss_fns, sample_weights)
                                val_loss.update(loss)
                                update_metrics(metric_value_groups, outputs, labels)
                                print_metrics(batch_idx, val_loss, metric_value_groups, 'val')
                            return aggregate_metrics('val', epoch, val_loss, metric_value_groups)

                    history = []
                    for epoch in range(epochs):
                        epoch_metrics = {
                            'epoch': epoch,
                            'train': _train(epoch)
                        }

                        if should_validate:
                            epoch_metrics['validation'] = _validate(epoch)

                        if user_verbose > 0:
                            pdt_dt = datetime.now(timezone.utc)
                            pdt_time_str = pdt_dt.strftime("%Y-%b-%d %H:%M:%S UTC")
                            print(pdt_time_str, epoch_metrics)

                        history.append(epoch_metrics)
                        if hvd.rank() == 0:
                            # Save model after every epoch
                            save_checkpoint()
                            if remote_store.saving_runs:
                                remote_store.sync(run_output_dir)

            if hvd.rank() == 0:
                best_checkpoint = torch.load(ckpt_file)
                serialized_checkpoint = io.BytesIO()
                torch.save(best_checkpoint, serialized_checkpoint)
                serialized_checkpoint.seek(0)
                return history, serialized_checkpoint

    return train


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


def _calculate_shuffle_buffer_size_fn():
    def calculate_shuffle_buffer_size(hvd, avg_row_size, train_row_count_per_worker):
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
        local_size = hvd.local_size()
        local_sizes = hvd.allgather(torch.tensor([local_size]))
        max_local_size = torch.max(local_sizes).item()

        if max_local_size > TOTAL_BUFFER_MEMORY_CAP_GIB:
            shuffle_buffer_size = TOTAL_BUFFER_MEMORY_CAP_GIB * BYTES_PER_GIB / avg_row_size / max_local_size
        else:
            shuffle_buffer_size = BYTES_PER_GIB / avg_row_size
        return int(min(shuffle_buffer_size, train_row_count_per_worker))

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
