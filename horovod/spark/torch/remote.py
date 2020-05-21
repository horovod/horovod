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
import os
import tempfile

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from horovod.spark.common import constants
from horovod.spark.common.util import _get_assigned_gpu_or_default, to_list
from horovod.spark.common.store import DBFSLocalStore
from horovod.spark.torch.util import deserialize_fn

PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER
METRIC_PRINT_FREQUENCY = constants.METRIC_PRINT_FREQUENCY
TOTAL_BUFFER_MEMORY_CAP_GIB = constants.TOTAL_BUFFER_MEMORY_CAP_GIB
BYTES_PER_GIB = constants.BYTES_PER_GIB
CUSTOM_SPARSE = constants.CUSTOM_SPARSE


def RemoteTrainer(estimator, metadata, ckpt_bytes, run_id, dataset_idx, train_rows, val_rows, avg_row_size, is_legacy):
    # Estimator parameters
    input_shapes = estimator.getInputShapes()
    label_shapes = estimator.getLabelShapes()
    feature_columns = estimator.getFeatureCols()
    label_columns = estimator.getLabelCols()
    should_validate = estimator.getValidation()
    batch_size = estimator.getBatchSize()
    epochs = estimator.getEpochs()
    user_shuffle_buffer_size = estimator.getShufflingBufferSize()
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None

    # Data reader parameters
    train_reader_worker_count = estimator.getTrainReaderNumWorker()
    val_reader_worker_count = estimator.getValReaderNumWorker()

    # Utility functions
    deserialize = deserialize_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn(
        train_rows, avg_row_size, user_shuffle_buffer_size)

    schema_fields = feature_columns + label_columns
    dataloader_cls = _create_dataloader(input_shapes, metadata)
    make_petastorm_reader = _make_petastorm_reader_fn(transformation, schema_fields,
                                                      batch_size, calculate_shuffle_buffer_size,
                                                      dataloader_cls)

    # Storage
    store = estimator.getStore()
    remote_store = store.to_remote(run_id, dataset_idx)
    is_dbfs = isinstance(store, DBFSLocalStore)

    train_steps_per_epoch = estimator.getTrainStepsPerEpoch()
    train_percent = train_rows / train_steps_per_epoch if train_steps_per_epoch else 1.0

    val_steps_per_epoch = estimator.getValidationStepsPerEpoch()
    val_percent = val_rows / val_steps_per_epoch if val_steps_per_epoch else 1.0

    def train(serialized_model):
        with tempfile.NamedTemporaryFile() as last_ckpt_file, remote_store.get_local_output_dir() as run_output_dir:
            if ckpt_bytes:
                last_ckpt_file.write(ckpt_bytes)

            logs_path = os.path.join(run_output_dir, remote_store.logs_subdir)
            logger = TensorBoardLogger(logs_path)

            ckpt_path = os.path.join(run_output_dir, remote_store.checkpoint_filename)
            checkpoint_callback = ModelCheckpoint(filepath=ckpt_path)

            model = deserialize(serialized_model)
            trainer = Trainer(distributed_backend='horovod',
                              gpus=1 if torch.cuda.is_available() else 0,
                              max_epochs=epochs,
                              train_percent_check=train_percent,
                              val_percent_check=val_percent,
                              logger=logger,
                              checkpoint_callback=checkpoint_callback,
                              resume_from_checkpoint=last_ckpt_file.name if ckpt_bytes else None)

            with make_petastorm_reader(trainer, model, remote_store.train_data_path, 'train_dataloader',
                                       train_reader_worker_count), \
                    make_petastorm_reader(trainer, model, remote_store.val_data_path, 'val_dataloader',
                                          val_reader_worker_count, should_validate):

                trainer.fit(model)

            serialized_checkpoint = io.BytesIO()
            module = model if not is_legacy else model._model
            torch.save({'model': module.state_dict()}, serialized_checkpoint)
            serialized_checkpoint.seek(0)
            return serialized_checkpoint
    return train


def _make_petastorm_reader_fn(transformation, schema_fields, batch_size, calculate_shuffle_buffer_size, dataloader_cls):
    @contextlib.contextmanager
    def make_petastorm_reader(trainer, model, data_path, dataloader_attr, reader_worker_count, should_read=True):
        from petastorm import TransformSpec, make_reader, make_batch_reader
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
                return dataloader_cls(reader, batch_size=batch_size,
                                      shuffling_queue_capacity=calculate_shuffle_buffer_size())
            try:
                print('PATCH: {} {}'.format(dataloader_attr, dataloader_fn.__code__))
                setattr(model, dataloader_attr, dataloader_fn)
                yield
            finally:
                setattr(model, dataloader_attr, None)
    return make_petastorm_reader


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


def _create_dataloader(input_shapes, metadata):
    from petastorm.pytorch import BatchedDataLoader

    prepare_data = _prepare_data_fn(metadata)

    class _DataLoader(BatchedDataLoader):
        def _yield_batches(self, keys):
            for batch in super()._yield_batches(keys):
                batch = {
                    k: prepare_data(k, v).reshape(input_shapes[k]) if k in input_shapes else v
                    for k, v in batch.items()
                }
                yield batch

    return _DataLoader


def _prepare_data_fn(metadata):
    def prepare_data(col_name, rows):
        if col_name not in metadata:
            return rows

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
    return prepare_data


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
