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
import tempfile

from distutils.version import LooseVersion

import h5py
import tensorflow as tf
import tensorflow.keras as keras

from horovod.spark.common import constants, util
from horovod.spark.common.store import LocalStore
from horovod.spark.keras.util import decompress_row_fn, reshape_row_fn
from horovod.run.common.util import codec


PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER
TOTAL_BUFFER_MEMORY_CAP_GIB = constants.TOTAL_BUFFER_MEMORY_CAP_GIB
BYTES_PER_GIB = constants.BYTES_PER_GIB
CUSTOM_SPARSE = constants.CUSTOM_SPARSE


def RemoteTrainer(estimator, metadata, run_id, train_data, val_data):
    # Estimator parameters
    label_columns = estimator.getLabelCols()
    feature_columns = estimator.getFeatureCols()
    user_callbacks = estimator.getCallbacks()
    batch_size = estimator.getBatchSize()
    epochs = estimator.getEpochs()
    train_steps_per_epoch = estimator.getTrainStepsPerEpoch()
    validation_steps_per_epoch = estimator.getValidationStepsPerEpoch()
    sample_weight_col = estimator.getSampleWeightCol()
    custom_objects = estimator.getCustomObjects()
    should_validate = estimator.getValidation()
    user_shuffle_buffer_size = estimator.getShufflingBufferSize()
    user_verbose = estimator.getVerbose()

    # Data reader parameters
    train_reader_worker_count = estimator.getTrainReaderNumWorker()
    val_reader_worker_count = estimator.getValReaderNumWorker()

    # Dataset statistics
    train_rows, train_bytes = util.get_dataset_row_count_and_total_bytes(train_data, 'train')
    avg_row_size = train_bytes / train_rows
    val_rows = util.get_dataset_row_count_and_total_bytes(val_data, 'val')[0] if val_data else 0

    # Model parameters
    input_shapes, output_shapes = estimator.get_model_shapes()
    output_names = estimator.getModel().output_names

    # Keras implementation
    floatx = keras.backend.floatx()

    # Utility functions
    deserialize_keras_model = _deserialize_keras_model_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn()
    pin_gpu = _pin_gpu_fn()

    # Store
    store = estimator.getStore() or LocalStore(tempfile.mkdtemp())
    remote_store = store.to_remote(run_id)

    has_sparse_col = any(metadata[col]['is_sparse_vector_only']
                         for col in label_columns + feature_columns)

    # Petastorm
    transformation = estimator.getTransformationFn()
    schema_fields = feature_columns + label_columns
    if sample_weight_col:
        schema_fields.append(sample_weight_col)

    def SyncCallback(root_path, sync_to_store_fn):
        class _SyncCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                sync_to_store_fn(root_path)

        return _SyncCallback()

    decompress_row = decompress_row_fn(metadata, feature_columns, label_columns, sample_weight_col)
    reshape_row = reshape_row_fn(feature_columns, label_columns, sample_weight_col,
                                 input_shapes, output_shapes, output_names)

    @contextlib.contextmanager
    def make_dataset(converter, rank, size, workers_count, shuffle_buffer_size=None):
        from petastorm import TransformSpec, make_reader, make_batch_reader

        if converter is None:
            yield None
        else:
            petastorm_reader_kwargs = {
                'cur_shard': rank,
                'shard_count': size,
                'reader_pool_type': 'process',
                'hdfs_driver': PETASTORM_HDFS_DRIVER,
                'schema_fields': schema_fields,
                'transform_spec': TransformSpec(transformation) if transformation else None
            }

            # In general, make_batch_reader is faster than make_reader for reading the dataset.
            # However, we found out that make_reader performs data transformations much faster than
            # make_batch_reader with parallel worker processes. Therefore, the default reader
            # we choose is make_batch_reader unless there are data transformations.
            make_reader_fn = make_batch_reader
            if transformation is not None:
                make_reader_fn = make_reader
                petastorm_reader_kwargs['pyarrow_serialize'] = True

            with converter.make_tf_dataset(batch_size=batch_size if not has_sparse_col else 1,
                                           workers_count=workers_count,
                                           shuffle_buffer_size=shuffle_buffer_size,
                                           make_reader_fn=make_reader_fn,
                                           **petastorm_reader_kwargs) as dataset:
                if has_sparse_col:
                    dataset = dataset.map(decompress_row).batch(batch_size)
                dataset = dataset.map(lambda row: reshape_row(row, has_sparse_col))
                yield dataset

    def train(serialized_model):
        import horovod.tensorflow.keras as hvd

        keras.backend.set_floatx(floatx)

        hvd.init()
        pin_gpu(hvd, tf, keras)

        if not user_shuffle_buffer_size:
            shuffle_buffer_size = calculate_shuffle_buffer_size(hvd, avg_row_size, train_rows / hvd.size())
        else:
            shuffle_buffer_size = user_shuffle_buffer_size

        # needs to be deserialized in the with scope
        with keras.utils.custom_object_scope(custom_objects):
            model = deserialize_keras_model(
                serialized_model, lambda x: hvd.load_model(x))

        # Horovod: adjust learning rate based on number of processes.
        keras.backend.set_value(model.optimizer.lr,
                            keras.backend.get_value(model.optimizer.lr) * hvd.size())

        # Verbose mode 1 will print a progress bar
        verbose = user_verbose if hvd.rank() == 0 else 0

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

                callbacks.append(keras.callbacks.ModelCheckpoint(ckpt_file))
                if remote_store.saving_runs:
                    callbacks.append(keras.callbacks.TensorBoard(logs_dir))
                    callbacks.append(SyncCallback(run_output_dir, remote_store.sync))

            if train_steps_per_epoch is None:
                steps_per_epoch = int(math.ceil(train_rows / batch_size / hvd.size()))
            else:
                steps_per_epoch = train_steps_per_epoch

            if validation_steps_per_epoch is None:
                # math.ceil because if val_rows is smaller than batch_size we still get the at least
                # one step. float(val_rows) because val_rows/batch_size evaluates to zero before
                # math.ceil
                validation_steps = int(math.ceil(float(val_rows) / batch_size / hvd.size())) \
                    if should_validate else None
            else:
                validation_steps = validation_steps_per_epoch

            with make_dataset(train_data, hvd.rank(), hvd.size(),
                              workers_count=train_reader_worker_count,
                              shuffle_buffer_size=shuffle_buffer_size) as train_dataset:
                with make_dataset(val_data, hvd.rank(), hvd.size(),
                                  workers_count=val_reader_worker_count,
                                  shuffle_buffer_size=shuffle_buffer_size) as val_dataset:
                    history = model.fit(
                        train_dataset,
                        validation_data=val_dataset,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks,
                        verbose=verbose,
                        epochs=epochs)

            # Dataset API usage currently displays a wall of errors upon termination.
            # This global model registration ensures clean termination.
            # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
            globals()['_DATASET_FINALIZATION_HACK'] = model

            if hvd.rank() == 0:
                with open(ckpt_file, 'rb') as f:
                    return history.history, codec.dumps_base64(f.read()), hvd.size()
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
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    return fn


def _pin_gpu_tensorflow1_fn():
    def fn(hvd, tf, keras):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
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
