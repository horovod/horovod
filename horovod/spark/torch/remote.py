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

import contextlib
import io
import math
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from horovod.spark.common import constants
from horovod.spark.common.util import to_list
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

    # Petastorm
    make_reader = _make_reader_fn()

    @contextlib.contextmanager
    def empty_batch_reader():
        yield None

    def train(serialized_model, optimizer_cls, model_opt_state_serialized,
              train_rows, val_rows, avg_row_size):
        from petastorm import TransformSpec
        from petastorm.pytorch import DataLoader
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

        if not user_shuffle_buffer_size:
            shuffle_buffer_size = \
                calculate_shuffle_buffer_size(hvd, avg_row_size, train_rows / hvd.size())
        else:
            shuffle_buffer_size = user_shuffle_buffer_size

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Horovod: pin GPU to local rank.
            torch.cuda.set_device(hvd.local_rank())
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
            steps_per_epoch = int(math.ceil(float(train_rows) / batch_size / hvd.size()))
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

            # Petastorm: read data from the store with the correct shard for this rank
            # setting num_epochs=None will cause an infinite iterator
            # and enables ranks to perform training and validation with
            # unequal number of samples
            with make_reader(remote_store.train_data_path,
                             num_epochs=None,
                             cur_shard=hvd.rank(),
                             reader_pool_type='process',
                             workers_count=train_reader_worker_count,
                             pyarrow_serialize=True,
                             shard_count=hvd.size(),
                             hdfs_driver=PETASTORM_HDFS_DRIVER,
                             schema_fields=schema_fields,
                             transform_spec=transform_spec,
                             is_petastorm_compatible=True) as train_reader:
                with make_reader(remote_store.val_data_path,
                                 num_epochs=None,
                                 cur_shard=hvd.rank(),
                                 reader_pool_type='process',
                                 pyarrow_serialize=True,
                                 workers_count=val_reader_worker_count,
                                 shard_count=hvd.size(),
                                 hdfs_driver=PETASTORM_HDFS_DRIVER,
                                 schema_fields=schema_fields,
                                 transform_spec=transform_spec,
                                 is_petastorm_compatible=True) \
                        if should_validate else empty_batch_reader() as val_reader:

                    train_loader = DataLoader(train_reader,
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
                        if cuda_available:
                            inputs = [input.cuda() for input in inputs]
                            labels = [label.cuda() for label in labels]
                        return inputs, labels, sample_weights

                    def transform_outputs(outputs, labels):
                        if type(outputs) != tuple and type(outputs) != list:
                            outputs = [outputs]

                        # reshape labels to match the output shape of the model
                        if hasattr(outputs[0], 'shape'):
                            labels = [label.reshape(output.shape)
                                      if output.shape.numel() == label.shape.numel() else label
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
                            print("epoch:\t{epoch}\tstep\t{batch_idx}:\t{metrics}".
                                  format(epoch=epoch,
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
                        val_loader = DataLoader(val_reader, batch_size=batch_size)
                        val_loader_iter = iter(val_loader)
                        if validation_steps_per_epoch is None:
                            validation_steps = int(math.ceil(float(val_rows) / batch_size / hvd.size()))
                        else:
                            validation_steps = validation_steps_per_epoch

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
                            print(epoch_metrics)

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


def _make_reader_fn():
    from petastorm.reader import NullCache, LocalDiskCache, FilesystemResolver, \
        logger, dataset_metadata, PetastormMetadataError, \
        ThreadPool, PyArrowSerializer, PickleSerializer, ProcessPool, DummyPool, Reader, \
        PyDictReaderWorker, six

    def make_reader(dataset_url,
                    schema_fields=None,
                    reader_pool_type='thread', workers_count=10, pyarrow_serialize=False,
                    results_queue_size=50,
                    shuffle_row_groups=True, shuffle_row_drop_partitions=1,
                    predicate=None,
                    rowgroup_selector=None,
                    num_epochs=1,
                    cur_shard=None, shard_count=None,
                    cache_type='null', cache_location=None, cache_size_limit=None,
                    cache_row_size_estimate=None, cache_extra_settings=None,
                    hdfs_driver='libhdfs3',
                    transform_spec=None,
                    is_petastorm_compatible=False):
        """
        Creates an instance of Reader for reading Petastorm datasets. A Petastorm dataset is a dataset generated using
        :func:`~petastorm.etl.dataset_metadata.materialize_dataset` context manager as explained
        `here <https://petastorm.readthedocs.io/en/latest/readme_include.html#generating-a-dataset>`_.
        See :func:`~petastorm.make_batch_reader` to read from a Parquet store that was not generated using
        :func:`~petastorm.etl.dataset_metadata.materialize_dataset`.
        :param dataset_url: an filepath or a url to a parquet directory,
            e.g. ``'hdfs://some_hdfs_cluster/user/yevgeni/parquet8'``, or ``'file:///tmp/mydataset'``,
            or ``'s3://bucket/mydataset'``, or ``'gs://bucket/mydataset'``.
        :param schema_fields: Can be: a list of unischema fields and/or regex pattern strings; ``None`` to read all fields;
                an NGram object, then it will return an NGram of the specified fields.
        :param reader_pool_type: A string denoting the reader pool type. Should be one of ['thread', 'process', 'dummy']
            denoting a thread pool, process pool, or running everything in the master thread. Defaults to 'thread'
        :param workers_count: An int for the number of workers to use in the reader pool. This only is used for the
            thread or process pool. Defaults to 10
        :param pyarrow_serialize: Whether to use pyarrow for serialization. Currently only applicable to process pool.
            Defaults to False.
        :param results_queue_size: Size of the results queue to store prefetched row-groups. Currently only applicable to
            thread reader pool type.
        :param shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
        :param shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
            break up a row group into for increased shuffling in exchange for worse performance (extra reads).
            For example if you specify 2 each row group read will drop half of the rows within every row group and
            read the remaining rows in separate reads. It is recommended to keep this number below the regular row
            group size in order to not waste reads which drop all rows.
        :param predicate: instance of :class:`.PredicateBase` object to filter rows to be returned by reader. The predicate
            will be passed a single row and must return a boolean value indicating whether to include it in the results.
        :param rowgroup_selector: instance of row group selector object to select row groups to be read
        :param num_epochs: An epoch is a single pass over all rows in the dataset. Setting ``num_epochs`` to
            ``None`` will result in an infinite number of epochs.
        :param cur_shard: An int denoting the current shard number. Each node reading a shard should
            pass in a unique shard number in the range [0, shard_count). shard_count must be supplied as well.
            Defaults to None
        :param shard_count: An int denoting the number of shards to break this dataset into. Defaults to None
        :param cache_type: A string denoting the cache type, if desired. Options are [None, 'null', 'local-disk'] to
            either have a null/noop cache or a cache implemented using diskcache. Caching is useful when communication
            to the main data store is either slow or expensive and the local machine has large enough storage
            to store entire dataset (or a partition of a dataset if shard_count is used). By default will be a null cache.
        :param cache_location: A string denoting the location or path of the cache.
        :param cache_size_limit: An int specifying the size limit of the cache in bytes
        :param cache_row_size_estimate: An int specifying the estimated size of a row in the dataset
        :param cache_extra_settings: A dictionary of extra settings to pass to the cache implementation,
        :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
            libhdfs (java through JNI) or libhdfs3 (C++)
        :param transform_spec: An instance of :class:`~petastorm.transform.TransformSpec` object defining how a record
            is transformed after it is loaded and decoded. The transformation occurs on a worker thread/process (depends
            on the ``reader_pool_type`` value).
        :param is_petastorm_compatible: A boolean value that indicates if the dataset is petastorm
            compatible or not. This is useful where the dataset is compatible with petastorm but is not
            generated with petastorm so it may not have the metadata file. Defaults to False.
        :return: A :class:`Reader` object
        """

        if dataset_url is None or not isinstance(dataset_url, six.string_types):
            raise ValueError('dataset_url must be a string')

        dataset_url = dataset_url[:-1] if dataset_url[-1] == '/' else dataset_url
        logger.debug('dataset_url: %s', dataset_url)

        resolver = FilesystemResolver(dataset_url, hdfs_driver=hdfs_driver)
        filesystem = resolver.filesystem()
        dataset_path = resolver.get_dataset_path()

        if cache_type is None or cache_type == 'null':
            cache = NullCache()
        elif cache_type == 'local-disk':
            cache = LocalDiskCache(cache_location, cache_size_limit, cache_row_size_estimate,
                                   **cache_extra_settings or {})
        else:
            raise ValueError('Unknown cache_type: {}'.format(cache_type))

        try:
            dataset_metadata.get_schema_from_dataset_url(dataset_url, hdfs_driver=hdfs_driver)
        except PetastormMetadataError:
            message = 'Currently make_reader supports reading only Petastorm datasets. To read from ' \
                      'a non-Petastorm Parquet store use make_batch_reader'
            if not is_petastorm_compatible:
                raise RuntimeError(message)
            else:
                logger.error(message)

        if reader_pool_type == 'thread':
            reader_pool = ThreadPool(workers_count, results_queue_size)
        elif reader_pool_type == 'process':
            if pyarrow_serialize:
                serializer = PyArrowSerializer()
            else:
                serializer = PickleSerializer()
            reader_pool = ProcessPool(workers_count, serializer)
        elif reader_pool_type == 'dummy':
            reader_pool = DummyPool()
        else:
            raise ValueError('Unknown reader_pool_type: {}'.format(reader_pool_type))

        kwargs = {
            'schema_fields': schema_fields,
            'reader_pool': reader_pool,
            'shuffle_row_groups': shuffle_row_groups,
            'shuffle_row_drop_partitions': shuffle_row_drop_partitions,
            'predicate': predicate,
            'rowgroup_selector': rowgroup_selector,
            'num_epochs': num_epochs,
            'cur_shard': cur_shard,
            'shard_count': shard_count,
            'cache': cache,
            'transform_spec': transform_spec,
        }

        try:
            return Reader(filesystem, dataset_path,
                          worker_class=PyDictReaderWorker,
                          is_batched_reader=False,
                          **kwargs)
        except PetastormMetadataError as e:
            logger.error('Unexpected exception: %s', str(e))
            raise RuntimeError(
                'make_reader has failed. If you were trying to open a Parquet store that was not '
                'created using Petastorm materialize_dataset and it contains only scalar columns, '
                'you may use make_batch_reader to read it.\n'
                'Inner exception: %s', str(e))

    return make_reader