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
from distutils.version import LooseVersion

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from horovod.spark.common import constants
from horovod.spark.common.util import _get_assigned_gpu_or_default, to_list
from horovod.spark.common.store import DBFSLocalStore
from horovod.spark.lightning.util import deserialize_fn

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
    sample_weight_col = estimator.getSampleWeightCol()
    should_validate = estimator.getValidation()
    batch_size = estimator.getBatchSize()
    val_batch_size = estimator.getValBatchSize() if estimator.getValBatchSize() else batch_size
    epochs = estimator.getEpochs()
    user_shuffle_buffer_size = estimator.getShufflingBufferSize()
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None
    inmemory_cache_all = estimator.getInMemoryCacheAll()

    # Data reader parameters
    train_reader_worker_count = estimator.getTrainReaderNumWorker()
    val_reader_worker_count = estimator.getValReaderNumWorker()

    # Utility functions
    deserialize = deserialize_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn(
        train_rows, avg_row_size, user_shuffle_buffer_size)

    schema_fields = feature_columns + label_columns
    if sample_weight_col:
        schema_fields.append(sample_weight_col)

    dataloader_cls = _create_dataloader(feature_columns, input_shapes, metadata)
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

    # disable call back for now. Because petastorm can not reset index during training.
    callbacks = None #_make_callbacks()

    def train(serialized_model):
        with tempfile.TemporaryDirectory() as last_ckpt_dir, remote_store.get_local_output_dir() as run_output_dir:
            last_ckpt_file = os.path.join(last_ckpt_dir, 'last.ckpt')
            if ckpt_bytes:
                with open(last_ckpt_file, 'wb') as f:
                    f.write(ckpt_bytes)

            logs_path = os.path.join(run_output_dir, remote_store.logs_subdir)
            logger = TensorBoardLogger(logs_path)

            ckpt_path = os.path.join(run_output_dir, remote_store.checkpoint_filename)
            os.makedirs(ckpt_path, exist_ok=True)

            # disable checkpoint call back for now, waiting for the fix of
            # https://github.com/PyTorchLightning/pytorch-lightning/issues/6343
            checkpoint_callback = None# ModelCheckpoint(dirpath=ckpt_path)

            model = deserialize(serialized_model)
            kwargs = {'accelerator': 'horovod',
                'gpus': (1 if torch.cuda.is_available() else 0),
                'callbacks': callbacks,
                'max_epochs': epochs,
                'limit_train_batches': train_percent,
                'limit_val_batches': val_percent,
                'logger': logger,
                'checkpoint_callback': checkpoint_callback,
                'resume_from_checkpoint': (last_ckpt_file if ckpt_bytes else None),
                'num_sanity_val_steps': 0
            }
            print("Creating trainer with: \n ", kwargs)
            trainer = Trainer(**kwargs)

            print(f"pytorch_lightning version={pl.__version__}")

            # print row group
            # pq.ParquetFile(remote_store.train_data_path)
            # for rowgroup in range(pq_file.metadata.num_row_groups):
            #     row_group = pq_file.metadata.row_group(rowgroup)
            #     print(row_group)

            with make_petastorm_reader(model, remote_store.train_data_path, 'train_dataloader',
                                       train_reader_worker_count), \
                    make_petastorm_reader(model, remote_store.val_data_path, 'val_dataloader',
                                          val_reader_worker_count, should_validate):

                trainer.fit(model)

            serialized_checkpoint = io.BytesIO()
            module = model if not is_legacy else model._model
            torch.save({'model': module.state_dict()}, serialized_checkpoint)
            serialized_checkpoint.seek(0)
            return serialized_checkpoint
    return train

def _reset_loader(loader):
    from petastorm.pytorch import BatchedDataLoader
    from pytorch_lightning.trainer.supporters import CombinedLoader

    if isinstance(loader, CombinedLoader):
        for loader in loader.loaders:
            loader.reader.reset()
    else:
        loader.reader.reset()

def _make_callbacks():
    class ResetCallback(Callback):
        def on_train_end(self, trainer, model):
            _reset_loader(trainer.train_dataloader)

        def on_validation_end(self, trainer, model):
            for loader in trainer.val_dataloaders:
                loader.reader.reset()

        def on_sanity_check_end(self, trainer, model):
            for loader in trainer.val_dataloaders:
                _reset_loader(loader)

    return [ResetCallback()]


def _make_petastorm_reader_fn(transformation, schema_fields, batch_size, calculate_shuffle_buffer_size, dataloader_cls):

    @contextlib.contextmanager
    def make_petastorm_reader(model, data_path, dataloader_attr, reader_worker_count, should_read=True):
        from petastorm import TransformSpec, make_reader, make_batch_reader
        import horovod.torch as hvd

        is_loader_overridden = False
        if LooseVersion(pl.__version__) >= LooseVersion('1.0.0'):
            from pytorch_lightning.utilities.model_helpers import is_overridden
            is_loader_overridden = is_overridden(dataloader_attr, model)

        if not should_read or is_loader_overridden:
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


def _create_dataloader(feature_columns, input_shapes, metadata):
    from petastorm.pytorch import BatchedDataLoader

    shape_dict = {col:shape for col, shape in zip(feature_columns, input_shapes)}
    prepare_data = _prepare_data_fn(metadata)

    class _DataLoader(BatchedDataLoader):
        def _yield_batches(self, keys):
            for batch in super()._yield_batches(keys):
                batch = {
                    k: prepare_data(k, v).reshape(shape_dict[k]) if k in shape_dict else v
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
