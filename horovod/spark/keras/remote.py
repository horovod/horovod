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

import h5py
import tensorflow as tf

from distutils.version import LooseVersion

from horovod.spark.common import constants
from horovod.spark.common.store import DBFSLocalStore
from horovod.spark.common.util import _get_assigned_gpu_or_default, _set_mp_start_method
from horovod.runner.common.util import codec


PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER
TOTAL_BUFFER_MEMORY_CAP_GIB = constants.TOTAL_BUFFER_MEMORY_CAP_GIB
BYTES_PER_GIB = constants.BYTES_PER_GIB


def RemoteTrainer(estimator, metadata, keras_utils, run_id, dataset_idx):
    # Estimator parameters
    label_columns = estimator.getLabelCols()
    feature_columns = estimator.getFeatureCols()
    user_callbacks = estimator.getCallbacks()
    batch_size = estimator.getBatchSize()
    val_batch_size = estimator.getValBatchSize() if estimator.getValBatchSize() else batch_size
    epochs = estimator.getEpochs()
    train_steps_per_epoch = estimator.getTrainStepsPerEpoch()
    validation_steps_per_epoch = estimator.getValidationStepsPerEpoch()
    sample_weight_col = estimator.getSampleWeightCol()
    custom_objects = estimator.getCustomObjects()
    should_validate = estimator.getValidation()
    random_seed = estimator.getRandomSeed()
    user_shuffle_buffer_size = estimator.getShufflingBufferSize()
    user_verbose = estimator.getVerbose()
    checkpoint_callback = estimator.getCheckpointCallback()
    inmemory_cache_all = estimator.getInMemoryCacheAll()
    should_use_gpu = estimator.getUseGpu()
    mp_start_method = estimator.getMpStartMethod()

    # Data reader parameters
    train_reader_worker_count = estimator.getTrainReaderNumWorker()
    val_reader_worker_count = estimator.getValReaderNumWorker()
    reader_pool_type = estimator.getReaderPoolType()

    # Model parameters
    input_shapes, output_shapes = estimator.get_model_shapes()
    output_names = estimator.getModel().output_names
    label_shapes = estimator.getLabelShapes()

    # Keras implementation
    keras_module = keras_utils.keras()
    floatx = keras_module.backend.floatx()
    get_horovod = keras_utils.horovod_fn()
    get_keras = keras_utils.keras_fn()
    make_dataset = keras_utils.make_dataset_fn(
        feature_columns=feature_columns,
        label_columns=label_columns,
        sample_weight_col=sample_weight_col,
        metadata=metadata,
        input_shapes=input_shapes,
        label_shapes=label_shapes if label_shapes else output_shapes,
        output_names=output_names)
    fit = keras_utils.fit_fn(epochs)
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None

    # Utility functions
    deserialize_keras_model = _deserialize_keras_model_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn()
    pin_gpu = _pin_gpu_fn()

    # Storage
    store = estimator.getStore()
    is_dbfs = isinstance(store, DBFSLocalStore)
    remote_store = store.to_remote(run_id, dataset_idx)
    storage_options = store.storage_options

    def SyncCallback(root_path, sync_to_store_fn, keras):
        class _SyncCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                sync_to_store_fn(root_path)

        return _SyncCallback()

    @contextlib.contextmanager
    def empty_batch_reader():
        yield None

    def train(serialized_model, train_rows, val_rows, avg_row_size):
        # If not empty, set it before everything else.
        if mp_start_method:
            _set_mp_start_method(mp_start_method, user_verbose)

        from petastorm import TransformSpec, make_reader, make_batch_reader
        import horovod as _horovod
        k = get_keras()
        k.backend.set_floatx(floatx)

        hvd = get_horovod()
        hvd.init()

        # Verbose mode 1 will print a progress bar
        verbose = user_verbose if hvd.rank() == 0 else 0

        if should_use_gpu:
            if verbose:
                print("Pinning current process to the GPU.")
            pin_gpu(hvd, tf, k)
        else:
            if verbose:
                print("Skip pinning current process to the GPU.")

        if random_seed is not None:
            if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
                tf.random.set_random_seed(random_seed)
            else:
                tf.random.set_seed(random_seed)

        # If user specifies any user_shuffle_buffer_size (even 0), we should honor it.
        if user_shuffle_buffer_size is None:
            shuffle_buffer_size = calculate_shuffle_buffer_size(
                hvd, avg_row_size, train_rows / hvd.size())
        else:
            if user_shuffle_buffer_size < 0:
                raise ValueError("user_shuffle_buffer_size cannot be negative!")
            shuffle_buffer_size = user_shuffle_buffer_size

        # needs to be deserialized in the with scope
        with k.utils.custom_object_scope(custom_objects):
            model = deserialize_keras_model(
                serialized_model, lambda x: hvd.load_model(x))

        # Horovod: adjust learning rate based on number of processes.
        scaled_lr = k.backend.get_value(model.optimizer.lr) * hvd.size()
        k.backend.set_value(model.optimizer.lr, scaled_lr)


        if verbose:
            print(f"Shared lib path is pointing to: {_horovod.common.process_sets._basics.MPI_LIB_CTYPES}")

        transform_spec = None
        if transformation:
            transform_spec = TransformSpec(transformation)

        # The inital_lr needs to be set to scaled learning rate in the checkpointing callbacks.
        for callback in user_callbacks:
            if isinstance(callback, _horovod._keras.callbacks.LearningRateScheduleCallbackImpl):
                callback.initial_lr = scaled_lr

        with remote_store.get_local_output_dir() as run_output_dir:
            callbacks = [
                # Horovod: broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),

                # Horovod: average metrics among workers at the end of every epoch.
                #
                # Note: This callback must be in the list before the ReduceLROnPlateau,
                # TensorBoard, or other metrics-based callbacks.
                hvd.callbacks.MetricAverageCallback(),
            ]

            callbacks += user_callbacks

            # Horovod: save checkpoints only on the first worker to prevent other workers from
            # corrupting them.
            if hvd.rank() == 0:
                ckpt_file = os.path.join(run_output_dir, remote_store.checkpoint_filename)
                logs_dir = os.path.join(run_output_dir, remote_store.logs_subdir)

                # This callback checkpoints the model that ultimately is wrapped and returned after
                # Estimator.fit is called.
                _checkpoint_callback = checkpoint_callback
                if _checkpoint_callback:
                    _checkpoint_callback.filepath = ckpt_file
                else:
                    if is_dbfs and LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
                        # Because DBFS local file APIs does not support random write which is
                        # required by h5 format, save_weights_only=True is needed for switching
                        # to the TensorFlow SavedModel format.
                        _checkpoint_callback = k.callbacks.ModelCheckpoint(ckpt_file,
                                                                           save_weights_only=True)
                    else:
                        _checkpoint_callback = k.callbacks.ModelCheckpoint(ckpt_file)
                callbacks.append(_checkpoint_callback)

                if remote_store.saving_runs:
                    tb_callback = None
                    for i, c in enumerate(callbacks):
                        if isinstance(c, k.callbacks.TensorBoard):
                            tb_callback = c
                            print(f"Found TensorBoard callback, updating log_dir to {logs_dir}")
                            tb_callback.log_dir = logs_dir
                            break
                    if tb_callback:
                        # Rather than a possibly arbitrary order, we always place the TensorBoard
                        # callback right before the SyncCallback
                        callbacks.pop(i)
                    callbacks.append(tb_callback or k.callbacks.TensorBoard(logs_dir))
                    callbacks.append(SyncCallback(run_output_dir, remote_store.sync, k))

            if train_steps_per_epoch is None:
                steps_per_epoch = int(math.ceil(train_rows / batch_size / hvd.size()))
            else:
                steps_per_epoch = train_steps_per_epoch

            if validation_steps_per_epoch is None:
                # math.ceil because if val_rows is smaller than val_batch_size we still get the at least
                # one step. float(val_rows) because val_rows/val_batch_size evaluates to zero before
                # math.ceil
                validation_steps = int(math.ceil(float(val_rows) / val_batch_size / hvd.size())) \
                    if should_validate else None
            else:
                validation_steps = validation_steps_per_epoch

            schema_fields = feature_columns + label_columns
            if sample_weight_col:
                schema_fields.append(sample_weight_col)

            if verbose:
                print(f"Training parameters: Epochs: {epochs}, Scaled lr: {scaled_lr}, "
                      f"Shuffle size: {shuffle_buffer_size}, random_seed: {random_seed}\n"
                      f"Train rows: {train_rows}, Train batch size: {batch_size}, Train_steps_per_epoch: {steps_per_epoch}\n"
                      f"Val rows: {val_rows}, Val batch size: {val_batch_size}, Val_steps_per_epoch: {validation_steps}\n"
                      f"Checkpoint file: {remote_store.checkpoint_path}, Logs dir: {remote_store.logs_path}\n")
            # In general, make_batch_reader is faster than make_reader for reading the dataset.
            # However, we found out that make_reader performs data transformations much faster than
            # make_batch_reader with parallel worker processes. Therefore, the default reader
            # we choose is make_batch_reader unless there are data transformations.
            reader_factory_kwargs = dict()
            if transform_spec:
                reader_factory = make_reader
                reader_factory_kwargs['pyarrow_serialize'] = True
                is_batch_reader = False
            else:
                reader_factory = make_batch_reader
                is_batch_reader = True

            with reader_factory(remote_store.train_data_path,
                                num_epochs=1,
                                cur_shard=hvd.rank(),
                                reader_pool_type=reader_pool_type,
                                workers_count=train_reader_worker_count,
                                shard_count=hvd.size(),
                                hdfs_driver=PETASTORM_HDFS_DRIVER,
                                schema_fields=schema_fields,
                                transform_spec=transform_spec,
                                storage_options=storage_options,
                                # Don't shuffle row groups if shuffle_buffer_size is 0 (non-shuffle case).
                                shuffle_row_groups=True if shuffle_buffer_size > 0 else False,
                                **reader_factory_kwargs) as train_reader:
                with reader_factory(remote_store.val_data_path,
                                    num_epochs=1,
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

                    train_data = make_dataset(train_reader, batch_size, shuffle_buffer_size,
                                              is_batch_reader, shuffle=True if shuffle_buffer_size > 0 else False,
                                              cache=inmemory_cache_all, seed=random_seed)
                    val_data = make_dataset(val_reader, val_batch_size, shuffle_buffer_size,
                                            is_batch_reader, shuffle=False, cache=inmemory_cache_all) \
                        if val_reader else None

                    history = fit(model, train_data, val_data, steps_per_epoch,
                                  validation_steps, callbacks, verbose)

            # Dataset API usage currently displays a wall of errors upon termination.
            # This global model registration ensures clean termination.
            # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
            globals()['_DATASET_FINALIZATION_HACK'] = model

            if hvd.rank() == 0:
                if is_dbfs:
                    if LooseVersion(tf.__version__) < LooseVersion("2.0.0"):
                        model.load_weights(ckpt_file)
                    else:
                        # needs to be deserialized in the with scope
                        with k.utils.custom_object_scope(custom_objects):
                            model = k.models.load_model(ckpt_file)
                    serialized_model = keras_utils.serialize_model(model)
                else:
                    if LooseVersion(tf.__version__) >= LooseVersion("2.0.0"):
                        with k.utils.custom_object_scope(custom_objects):
                            model = k.models.load_model(ckpt_file)
                        serialized_model = keras_utils.serialize_model(model)
                    else:
                        with open(ckpt_file, 'rb') as f:
                            serialized_model = codec.dumps_base64(f.read())

                return history.history, serialized_model, hvd.size()
    return train


def _deserialize_keras_model_fn():
    def deserialize_keras_model(model_bytes, load_model_fn):
        """Deserialize model from byte array encoded in base 64."""
        model_bytes = codec.loads_base64(model_bytes)
        bio = io.BytesIO(model_bytes)
        with h5py.File(bio, 'r') as f:
            return load_model_fn(f)
    return deserialize_keras_model


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
        local_sizes = hvd.allgather([local_size])
        max_local_size = int(max(local_sizes))

        if max_local_size > TOTAL_BUFFER_MEMORY_CAP_GIB:
            shuffle_buffer_size = TOTAL_BUFFER_MEMORY_CAP_GIB * BYTES_PER_GIB / avg_row_size / max_local_size
        else:
            shuffle_buffer_size = BYTES_PER_GIB / avg_row_size

        return int(min(shuffle_buffer_size, train_row_count_per_worker))

    return calculate_shuffle_buffer_size


def _pin_gpu_fn():
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    return _pin_gpu_tensorflow2_fn() if LooseVersion(tf.__version__) >= LooseVersion('2.0.0') \
        else _pin_gpu_tensorflow1_fn()


def _pin_gpu_tensorflow2_fn():
    def fn(hvd, tf, keras):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(
                gpus[_get_assigned_gpu_or_default(default=hvd.local_rank())], 'GPU')
    return fn


def _pin_gpu_tensorflow1_fn():
    def fn(hvd, tf, keras):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = \
            str(_get_assigned_gpu_or_default(default=hvd.local_rank()))
        keras.backend.set_session(tf.Session(config=config))
    return fn


def _pin_cpu_fn():
    return _pin_cpu_tensorflow2_fn() if LooseVersion(tf.__version__) >= LooseVersion('2.0.0') \
        else _pin_cpu_tensorflow1_fn()


def _pin_cpu_tensorflow2_fn():
    def fn(tf, keras):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    return fn


def _pin_cpu_tensorflow1_fn():
    def fn(tf, keras):
        config = tf.ConfigProto(device_count={'GPU': 0})
        config.inter_op_parallelism_threads = 1
        config.intra_op_parallelism_threads = 1
        keras.backend.set_session(tf.Session(config=config))
    return fn
