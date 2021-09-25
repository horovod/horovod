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
import math
from distutils.version import LooseVersion

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger

from horovod.spark.common import constants
from horovod.spark.common.util import _get_assigned_gpu_or_default
from horovod.spark.lightning.datamodule import PetastormDataModule
from horovod.spark.lightning.util import deserialize_fn

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
    terminate_on_nan = estimator.getTerminateOnNan()
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None
    inmemory_cache_all = estimator.getInMemoryCacheAll()
    callbacks = estimator.getCallbacks() or []
    train_steps_per_epoch = estimator.getTrainStepsPerEpoch()
    val_steps_per_epoch = estimator.getValidationStepsPerEpoch()
    num_gpus = estimator.getNumGPUs()
    data_module = estimator.getDataModule() if estimator.getDataModule() else PetastormDataModule
    loader_num_epochs = estimator.getLoaderNumEpochs()
    verbose = (estimator.getVerbose() > 0)

    # get logger
    logger = estimator.getLogger()
    log_every_n_steps = estimator.getLogEveryNSteps()
    print(f"logger is configured. _experiment_key={logger._experiment_key}, {var(logger)}")
    comet_experiment_key = logger._experiment_key

    # Data reader parameters
    train_reader_worker_count = estimator.getTrainReaderNumWorker()
    val_reader_worker_count = estimator.getValReaderNumWorker()
    reader_pool_type = estimator.getReaderPoolType()

    # Utility functions
    deserialize = deserialize_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn(
        train_rows, avg_row_size, user_shuffle_buffer_size)

    schema_fields = feature_columns + label_columns
    if sample_weight_col:
        schema_fields.append(sample_weight_col)

    # Storage
    store = estimator.getStore()
    remote_store = store.to_remote(run_id, dataset_idx)
    storage_options = store.storage_options

    profiler = estimator.getProfiler()

    def train(serialized_model):
        import horovod.torch as hvd
        # Horovod: initialize library.
        hvd.init()

        with tempfile.TemporaryDirectory() as last_ckpt_dir, remote_store.get_local_output_dir() as run_output_dir:
            last_ckpt_file = os.path.join(last_ckpt_dir, 'last.ckpt')
            if ckpt_bytes:
                with open(last_ckpt_file, 'wb') as f:
                    f.write(ckpt_bytes)

            # TODO: Pass the logger from estimator constructor
            logs_path = os.path.join(run_output_dir, remote_store.logs_subdir)
            os.makedirs(logs_path, exist_ok=True)
            print(f"Made directory {logs_path} for horovod rank {hvd.rank()}")

            # Use default logger if no logger is supplied
            train_logger = logger
            print(f"Train_logger is _experiment_key={train_logger._experiment_key} , {comet_experiment_key}, {var(train_logger)}")
            logger._experiment_key = comet_experiment_key

            if train_logger is None:
                train_logger = TensorBoardLogger(logs_path)

            # TODO: find out a way to use ckpt_path created from remote store, but all other parameters ingest from estimator config
            # ckpt_path = os.path.join(run_output_dir, remote_store.checkpoint_filename)
            # os.makedirs(ckpt_path, exist_ok=True)
            # model_checkpoint_callback = ModelCheckpoint(dirpath=ckpt_path)
            # callbacks.append(model_checkpoint_callback)

            is_model_checkpoint_callback_exist = False
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    is_model_checkpoint_callback_exist = True
                    break

            if remote_store.saving_runs and hvd.rank() == 0:
                class _SyncCallback(Callback):
                    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                        remote_store.sync(logs_path)

                callbacks.append(_SyncCallback())

            model = deserialize(serialized_model)

            _train_steps_per_epoch = train_steps_per_epoch if train_steps_per_epoch else \
                int(math.floor(float(train_rows) / batch_size / hvd.size()))

            _val_steps_per_epoch = val_steps_per_epoch if val_steps_per_epoch else \
                int(math.floor(float(val_rows) / val_batch_size / hvd.size()))

            print(f"Training data of rank[{hvd.local_rank()}]: train_rows:{train_rows}, batch_size:{batch_size}, _train_steps_per_epoch:{_train_steps_per_epoch}.")

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

            _num_gpus = num_gpus
            if _num_gpus is None:
                _num_gpus = 1 if cuda_available else 0

            kwargs = {'accelerator': 'horovod',
                      'gpus': _num_gpus,
                      'callbacks': callbacks,
                      'max_epochs': epochs,
                      'logger': train_logger,
                      'log_every_n_steps': log_every_n_steps,
                      'resume_from_checkpoint': (last_ckpt_file if ckpt_bytes else None),
                      'checkpoint_callback': is_model_checkpoint_callback_exist,
                      'num_sanity_val_steps': 0,
                      'reload_dataloaders_every_epoch': False,
                      'progress_bar_refresh_rate': _train_steps_per_epoch // 10,
                      'terminate_on_nan': terminate_on_nan,
                      'profiler': profiler
                      }
            print("Creating trainer with: \n ", kwargs)
            trainer = Trainer(**kwargs)

            if trainer.profiler:
                print(f"Set profiler's logs_path to {logs_path}")
                trainer.profiler.dirpath = logs_path

            print(f"pytorch_lightning version={pl.__version__}")

            dataset = data_module(train_dir=remote_store.train_data_path,
                                  val_dir=remote_store.val_data_path,
                                  num_train_epochs=epochs,
                                  has_val=should_validate is not None,
                                  train_batch_size=batch_size, val_batch_size=val_batch_size,
                                  shuffle_size=calculate_shuffle_buffer_size(),
                                  num_reader_epochs=loader_num_epochs,
                                  reader_pool_type=reader_pool_type, reader_worker_count=train_reader_worker_count,
                                  transform_spec=transformation, inmemory_cache_all=inmemory_cache_all,
                                  cur_shard=hvd.rank(), shard_count=hvd.size(),
                                  schema_fields=schema_fields, storage_options=storage_options,
                                  steps_per_epoch_train=_train_steps_per_epoch,
                                  steps_per_epoch_val=_val_steps_per_epoch,
                                  verbose=verbose)
            trainer.fit(model, dataset)

            serialized_checkpoint = io.BytesIO()
            module = model if not is_legacy else model._model

            # TODO: find a way to pass trainer.logged_metrics out.
            output = {'model': module.state_dict()}

            torch.save(output, serialized_checkpoint)

            if remote_store.saving_runs and hvd.rank() == 0:
                remote_store.sync(logs_path)

            serialized_checkpoint.seek(0)
            return serialized_checkpoint
    return train


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
