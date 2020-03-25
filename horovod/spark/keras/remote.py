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

import h5py
import tensorflow as tf

from distutils.version import LooseVersion

from horovod.spark.common import constants
from horovod.run.common.util import codec


PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER
TOTAL_BUFFER_MEMORY_CAP_GIB = constants.TOTAL_BUFFER_MEMORY_CAP_GIB
BYTES_PER_GIB = constants.BYTES_PER_GIB


def RemoteTrainer(estimator, metadata, keras_utils, run_id, dataset_idx):
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

    # Model parameters
    input_shapes, output_shapes = estimator.get_model_shapes()
    output_names = estimator.getModel().output_names

    # Keras implementation
    keras_module = keras_utils.keras()
    floatx = keras_module.backend.floatx()
    get_horovod = keras_utils.horovod_fn()
    get_keras = keras_utils.keras_fn()
    make_dataset = keras_utils.make_dataset_fn(
        feature_columns, label_columns, sample_weight_col, metadata,
        input_shapes, output_shapes, output_names, batch_size)
    fit = keras_utils.fit_fn(epochs)
    transformation_fn = estimator.getTransformationFn()
    transformation = transformation_fn if transformation_fn else None

    # Utility functions
    deserialize_keras_model = _deserialize_keras_model_fn()
    calculate_shuffle_buffer_size = _calculate_shuffle_buffer_size_fn()
    pin_gpu = _pin_gpu_fn()

    # Storage
    store = estimator.getStore()
    remote_store = store.to_remote(run_id, dataset_idx)

    # Petastorm
    make_reader = _make_reader_fn()

    def SyncCallback(root_path, sync_to_store_fn, keras):
        class _SyncCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                sync_to_store_fn(root_path)

        return _SyncCallback()

    @contextlib.contextmanager
    def empty_batch_reader():
        yield None

    def train(serialized_model, train_rows, val_rows, avg_row_size):
        from petastorm import TransformSpec

        k = get_keras()
        k.backend.set_floatx(floatx)

        hvd = get_horovod()
        hvd.init()
        pin_gpu(hvd, tf, k)

        if not user_shuffle_buffer_size:
            shuffle_buffer_size = calculate_shuffle_buffer_size(
                hvd, avg_row_size, train_rows / hvd.size())
        else:
            shuffle_buffer_size = user_shuffle_buffer_size

        # needs to be deserialized in the with scope
        with k.utils.custom_object_scope(custom_objects):
            model = deserialize_keras_model(
                serialized_model, lambda x: hvd.load_model(x))

        # Horovod: adjust learning rate based on number of processes.
        k.backend.set_value(model.optimizer.lr,
                            k.backend.get_value(model.optimizer.lr) * hvd.size())

        # Verbose mode 1 will print a progress bar
        verbose = user_verbose if hvd.rank() == 0 else 0

        transform_spec = None
        if transformation:
            transform_spec = TransformSpec(transformation)

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

                callbacks.append(k.callbacks.ModelCheckpoint(ckpt_file))
                if remote_store.saving_runs:
                    callbacks.append(k.callbacks.TensorBoard(logs_dir))
                    callbacks.append(SyncCallback(run_output_dir, remote_store.sync, k))

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

            schema_fields = feature_columns + label_columns
            if sample_weight_col:
                schema_fields.append(sample_weight_col)

            # Petastorm: read data from the store with the correct shard for this rank
            # setting num_epochs=None will cause an infinite iterator and enables
            # ranks to perform training and validation with unequal number of
            # samples
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

                    train_data = make_dataset(train_reader, shuffle_buffer_size, shuffle=True)
                    val_data = make_dataset(val_reader, shuffle_buffer_size, shuffle=False) \
                        if val_reader else None

                    history = fit(model, train_data, val_data, steps_per_epoch,
                                  validation_steps, callbacks, verbose)

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
        max_local_size = max(local_sizes)

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

def _make_reader_fn():
    from petastorm.reader import normalize_dataset_url, NullCache, LocalDiskCache, \
        logger, dataset_metadata, get_filesystem_and_path_or_paths, PetastormMetadataError, \
        ThreadPool, PyArrowSerializer, PickleSerializer, ProcessPool, DummyPool, Reader, \
        PyDictReaderWorker
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
        dataset_url = normalize_dataset_url(dataset_url)

        filesystem, dataset_path = get_filesystem_and_path_or_paths(dataset_url, hdfs_driver)

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