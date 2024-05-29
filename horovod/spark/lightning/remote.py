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
import warnings
from packaging import version

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger

from horovod.spark.common import constants
from horovod.spark.common.util import _get_assigned_gpu_or_local_rank, _set_mp_start_method
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
    random_seed = estimator.getRandomSeed()
    user_shuffle_buffer_size = estimator.getShufflingBufferSize()
    if user_shuffle_buffer_size is not None:
        warnings.warn('shuffle_buffer_size is deprecated and will be removed in future releases, '\
                      'use shuffle instead', DeprecationWarning)
    shuffle = estimator.getShuffle()
    terminate_on_nan = estimator.getTerminateOnNan()
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None
    transformation_edit_fields = estimator.getTransformationEditFields()
    transformation_removed_fields = estimator.getTransformationRemovedFields()
    inmemory_cache_all = estimator.getInMemoryCacheAll()
    callbacks = estimator.getCallbacks() or []
    train_steps_per_epoch = estimator.getTrainStepsPerEpoch()
    val_steps_per_epoch = estimator.getValidationStepsPerEpoch()
    num_gpus = estimator.getNumGPUs()
    data_module = estimator.getDataModule() if estimator.getDataModule() else PetastormDataModule
    loader_num_epochs = estimator.getLoaderNumEpochs()
    verbose = (estimator.getVerbose() > 0)
    trainer_args = estimator.getTrainerArgs()
    debug_data_loader = estimator.getDebugDataLoader()
    train_async_data_loader_queue_size = estimator.getTrainAsyncDataLoaderQueueSize()
    val_async_data_loader_queue_size = estimator.getValAsyncDataLoaderQueueSize()
    should_use_gpu = estimator.getUseGpu()
    mp_start_method = estimator.getMpStartMethod()

    # get logger
    logger = estimator.getLogger()
    log_every_n_steps = estimator.getLogEveryNSteps()
    print(f"logger is configured: {logger}")

    # Comet logger's expriment key is not serialize correctly. Need to remember the key, and
    # resume the logger experiment from GPU instance.
    if isinstance(logger, CometLogger):
        logger_experiment_key = logger._experiment_key
        print(f"logger vars: {vars(logger)}")
    else:
        logger_experiment_key = None

    # Data reader parameters
    train_reader_worker_count = estimator.getTrainReaderNumWorker()
    val_reader_worker_count = estimator.getValReaderNumWorker()
    reader_pool_type = estimator.getReaderPoolType()

    # Utility functions
    deserialize = deserialize_fn()

    schema_fields = feature_columns + label_columns
    if sample_weight_col:
        schema_fields.append(sample_weight_col)

    # Storage
    store = estimator.getStore()
    remote_store = store.to_remote(run_id, dataset_idx)
    storage_options = store.storage_options

    profiler = estimator.getProfiler()

    def train(serialized_model):
        # If not empty, set it before everything else.
        if mp_start_method:
            _set_mp_start_method(mp_start_method, verbose)

        import horovod.torch as hvd

        if random_seed is not None:
            pl.utilities.seed.seed_everything(seed=random_seed)

        # Horovod: initialize library.
        hvd.init()

        if verbose:
            import horovod as _horovod
            print(f"Shared lib path is pointing to: {_horovod.common.process_sets._basics.MPI_LIB_CTYPES}")

        _checkpoint_callback = None
        require_checkpoint = False

        with remote_store.get_local_output_dir() as run_output_dir:
            logs_path = os.path.join(run_output_dir, remote_store.logs_subdir)
            os.makedirs(logs_path, exist_ok=True)
            print(f"Made directory {logs_path} for horovod rank {hvd.rank()}")
            ckpt_dir = run_output_dir
            ckpt_filename = remote_store.checkpoint_filename

            if logger is None:
                # Use default logger if no logger is supplied
                train_logger = TensorBoardLogger(logs_path)
                print(f"Setup logger: Using TensorBoardLogger: {train_logger}")

            elif isinstance(logger, CometLogger):
                if logger._experiment_key:
                    # use logger passed in.
                    train_logger = logger
                    train_logger._save_dir = logs_path
                    print(f"Setup logger: change save_dir of the logger to {logs_path}")

                elif logger_experiment_key:
                    # Resume logger experiment with new log path if key passed correctly from CPU.
                    train_logger = CometLogger(
                        save_dir=logs_path,
                        api_key=logger.api_key,
                        experiment_key=logger_experiment_key,
                    )

                    print(f"Setup logger: Resume comet logger: {vars(train_logger)}")

                else:
                    print(f"Failed to setup or resume comet logger. origin logger: {vars(logger)}")

            else:
                # use logger passed in.
                train_logger = logger
                train_logger.save_dir = logs_path
                print(f"Setup logger: Using logger passed from estimator: {train_logger}")

            # Lightning requires to add checkpoint callbacks for all ranks.
            # Otherwise we are seeing hanging in training.
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    cb.dirpath = ckpt_dir
                    cb.filename = ckpt_filename
                    _checkpoint_callback = cb
                    require_checkpoint = True
                    break
            if not _checkpoint_callback:
                # By default 'monitor'=None which saves a checkpoint only for the last epoch.
                _checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                                       filename=ckpt_filename,
                                                       verbose=True)
                callbacks.append(_checkpoint_callback)

            if remote_store.saving_runs and hvd.rank() == 0:
                # Horovod: sync checkpoint and logging files only on rank 0 to
                # prevent other ranks from corrupting them.
                class _SyncCallback(Callback):
                    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
                        remote_store.sync(run_output_dir)

                callbacks.append(_SyncCallback())

            model = deserialize(serialized_model)

            _train_steps_per_epoch = train_steps_per_epoch if train_steps_per_epoch else \
                int(math.floor(float(train_rows) / batch_size / hvd.size()))

            _val_steps_per_epoch = val_steps_per_epoch if val_steps_per_epoch else \
                int(math.floor(float(val_rows) / val_batch_size / hvd.size()))

            if verbose:
                print(f"Training data of rank[{hvd.local_rank()}]: Epochs: {epochs}, "
                      f"Shuffle: {shuffle}, Random seed: {random_seed}\n"
                      f"Train rows: {train_rows}, Train batch size: {batch_size}, Train_steps_per_epoch: {_train_steps_per_epoch}\n"
                      f"Val rows: {val_rows}, Val batch size: {val_batch_size}, Val_steps_per_epoch: {_val_steps_per_epoch}\n"
                      f"Checkpoint file: {remote_store.checkpoint_path}, Logs dir: {remote_store.logs_path}\n")

            if not should_use_gpu and verbose:
                print("Skip pinning current process to the GPU.")

            cuda_available = torch.cuda.is_available()

            if cuda_available and not should_use_gpu:
                print("GPU is available but use_gpu is set to False."
                      "Training will proceed without GPU support.")
                cuda_available = False

            # We need to check all ranks have same device type for traning.
            # Horovod doesn't support heterogeneous allreduce for gradients.
            cuda_avail_list = hvd.allgather_object(cuda_available, name='device type')
            if cuda_avail_list.count(cuda_available) != hvd.size():
                raise RuntimeError("All ranks don't have same device type!")

            if cuda_available:
                # Horovod: pin GPU to local rank or the assigned GPU from spark.
                torch.cuda.set_device(_get_assigned_gpu_or_local_rank(local_rank=hvd.local_rank()))
                # Move model to GPU.
                model.cuda()

            _num_gpus = num_gpus
            if _num_gpus is None:
                _num_gpus = 1 if cuda_available else 0

            # Set bar refresh to 1 / epoch, detailed loss and metrics is avaialbe in logger,
            # no need to print in screen here. User can still override this in trainer_args
            progress_bar_refresh_rate = _train_steps_per_epoch

            kwargs = {'accelerator': 'horovod',
                      'gpus': _num_gpus,
                      'callbacks': callbacks,
                      'max_epochs': epochs,
                      'logger': train_logger,
                      'log_every_n_steps': log_every_n_steps,
                      'num_sanity_val_steps': 0,
                      'reload_dataloaders_every_epoch': False,
                      'progress_bar_refresh_rate': progress_bar_refresh_rate,
                      'terminate_on_nan': terminate_on_nan,
                      'profiler': profiler
                      }
            if trainer_args:
                kwargs.update(trainer_args)

            if verbose and hvd.rank() == 0:
                print("Creating trainer with: \n ", kwargs)

            trainer = Trainer(**kwargs)

            if profiler != 'simple' and trainer.profiler:
                print(f"Set profiler's logs_path for {hvd.rank()} to {logs_path}")
                trainer.profiler.dirpath = logs_path
                # filename where the profiler results will be saved instead of
                # printing to stdout. The .txt extension will be used automatically.
                trainer.profiler.filename = "profile"

            if verbose and hvd.rank() == 0:
                print(f"pytorch_lightning version={pl.__version__}")

            data_module_kwargs = {
                'train_dir': remote_store.train_data_path,
                'val_dir': remote_store.val_data_path,
                'num_train_epochs': epochs,
                'has_val': should_validate is not None,
                'train_batch_size': batch_size,
                'val_batch_size': val_batch_size,
                'shuffle': shuffle,
                'num_reader_epochs': loader_num_epochs,
                'reader_pool_type': reader_pool_type,
                'reader_worker_count': train_reader_worker_count,
                'transformation': transformation,
                'transformation_edit_fields': transformation_edit_fields,
                'transformation_removed_fields': transformation_removed_fields,
                'inmemory_cache_all': inmemory_cache_all,
                'cur_shard': hvd.rank(),
                'shard_count': hvd.size(),
                'schema_fields': schema_fields,
                'storage_options': storage_options,
                'steps_per_epoch_train': _train_steps_per_epoch,
                'steps_per_epoch_val': _val_steps_per_epoch,
                'verbose': verbose,
                'debug_data_loader': debug_data_loader,
                'train_async_data_loader_queue_size': train_async_data_loader_queue_size,
                'val_async_data_loader_queue_size': val_async_data_loader_queue_size,
                'seed': random_seed,
            }
            if debug_data_loader and hvd.rank() == 0:
                print(f"Creating data module with args:\n {data_module_kwargs}")

            dataset = data_module(**data_module_kwargs)

            trainer.fit(model, dataset)

            if hvd.rank() == 0:
                if remote_store.saving_runs and trainer.profiler:
                    # One more file sync to push profiler result.
                    remote_store.sync(logs_path)

                # rank 0 overwrites model with best checkpoint and returns.
                if require_checkpoint:
                    if verbose:
                        print("load from checkpoint best model path:",
                              _checkpoint_callback.best_model_path)
                    best_model = model.load_from_checkpoint(_checkpoint_callback.best_model_path)
                else:
                    best_model = model
                serialized_checkpoint = io.BytesIO()
                module = best_model if not is_legacy else best_model._model

                output = {'model': module.state_dict(), 'logged_metrics': trainer.logged_metrics}

                torch.save(output, serialized_checkpoint)

                return serialized_checkpoint
    return train


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
