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
import errno
import os
import pathlib
import re
import shutil
import tempfile
import warnings

from distutils.version import LooseVersion

import pyarrow as pa
import pyarrow.parquet as pq

import fsspec
from fsspec.core import split_protocol
from fsspec.utils import update_storage_options
from fsspec.callbacks import _DEFAULT_CALLBACK

from horovod.spark.common.util import is_databricks, host_hash


class Store(object):
    """
    Storage layer for intermediate files (materialized DataFrames) and training artifacts (checkpoints, logs).

    Store provides an abstraction over a filesystem (e.g., local vs HDFS) or blob storage database. It provides the
    basic semantics for reading and writing objects, and how to access objects with certain definitions.

    The store exposes a generic interface that is not coupled to a specific DataFrame, model, or runtime. Every run
    of an Estimator should result in a separate run directory containing checkpoints and logs, and every variation
    in dataset should produce a separate intermediate data path.

    In order to allow for caching but to prevent overuse of disk space on intermediate data, intermediate datasets
    are named in a deterministic sequence. When a dataset is done being used for training, the intermediate files
    can be reclaimed to free up disk space, but will not be automatically removed so that they can be reused as
    needed. This is to support both parallel training processes using the same store on multiple DataFrames, as well
    as iterative training using the same DataFrame on different model variations.
    """
    def __init__(self):
        self._train_data_to_key = {}
        self._val_data_to_key = {}

    def is_parquet_dataset(self, path):
        """Returns True if the path is the root of a Parquet dataset."""
        raise NotImplementedError()

    def get_parquet_dataset(self, path):
        """Returns a :py:class:`pyarrow.parquet.ParquetDataset` from the path."""
        raise NotImplementedError()

    def get_train_data_path(self, idx=None):
        """Returns the path to the training dataset."""
        raise NotImplementedError()

    def get_val_data_path(self, idx=None):
        """Returns the path to the validation dataset."""
        raise NotImplementedError()

    def get_test_data_path(self, idx=None):
        """Returns the path to the test dataset."""
        raise NotImplementedError()

    def saving_runs(self):
        """Returns True if run output should be saved during training."""
        raise NotImplementedError()

    def get_runs_path(self):
        """Returns the base path for all runs."""
        raise NotImplementedError()

    def get_run_path(self, run_id):
        """Returns the path to the run with the given ID."""
        raise NotImplementedError()

    def get_checkpoint_path(self, run_id):
        """Returns the path to the checkpoint file(s) for the given run."""
        raise NotImplementedError()

    def get_checkpoints(self, run_id, suffix='.ckpt'):
        """Returns a list of paths for all checkpoints saved this run."""
        raise NotImplementedError()

    def get_logs_path(self, run_id):
        """Returns the path to the log directory for the given run."""
        raise NotImplementedError()

    def get_checkpoint_filename(self):
        """Returns the basename of the saved checkpoint file."""
        raise NotImplementedError()

    def get_logs_subdir(self):
        """Returns the subdirectory name for the logs directory."""
        raise NotImplementedError()

    def exists(self, path):
        """Returns True if the path exists in the store."""
        raise NotImplementedError()

    def read(self, path):
        """Returns the contents of the path as bytes."""
        raise NotImplementedError()

    def write_text(self, path, text):
        """Write text file to path."""
        raise NotImplementedError()

    def get_local_output_dir_fn(self, run_id):
        raise NotImplementedError()

    def sync_fn(self, run_id):
        """Returns a function that synchronises given path recursively into run path for `run_id`."""
        raise NotImplementedError()

    def to_remote(self, run_id, dataset_idx):
        """Returns a view of the store that can execute in a remote environment without Horoovd deps."""
        attrs = self._remote_attrs(run_id, dataset_idx)

        class RemoteStore(object):
            def __init__(self):
                for name, attr in attrs.items():
                    setattr(self, name, attr)

        return RemoteStore()

    def _remote_attrs(self, run_id, dataset_idx):
        return {
            'train_data_path': self.get_train_data_path(dataset_idx),
            'val_data_path': self.get_val_data_path(dataset_idx),
            'test_data_path': self.get_test_data_path(dataset_idx),
            'saving_runs': self.saving_runs(),
            'runs_path': self.get_runs_path(),
            'run_path': self.get_run_path(run_id),
            'checkpoint_path': self.get_checkpoint_path(run_id),
            'logs_path': self.get_logs_path(run_id),
            'checkpoint_filename': self.get_checkpoint_filename(),
            'logs_subdir': self.get_logs_subdir(),
            'get_local_output_dir': self.get_local_output_dir_fn(run_id),
            'sync': self.sync_fn(run_id)
        }

    @staticmethod
    def create(prefix_path, *args, **kwargs):
        if HDFSStore.matches(prefix_path):
            return HDFSStore(prefix_path, *args, **kwargs)
        elif is_databricks() and DBFSLocalStore.matches_dbfs(prefix_path):
            return DBFSLocalStore(prefix_path, *args, **kwargs)
        else:
            return FilesystemStore(prefix_path, *args, **kwargs)


class AbstractFilesystemStore(Store):
    """Abstract class for stores that use a filesystem for underlying storage."""

    def __init__(self, prefix_path, train_path=None, val_path=None, test_path=None,
            runs_path=None, save_runs=True, storage_options=None, checkpoint_filename=None,
            **kwargs):
        self.prefix_path = self.get_full_path(prefix_path)
        self._train_path = self._get_full_path_or_default(train_path, 'intermediate_train_data')
        self._val_path = self._get_full_path_or_default(val_path, 'intermediate_val_data')
        self._test_path = self._get_full_path_or_default(test_path, 'intermediate_test_data')
        self._runs_path = self._get_full_path_or_default(runs_path, 'runs')
        self._save_runs = save_runs
        self.storage_options = storage_options
        self.checkpoint_filename = checkpoint_filename if checkpoint_filename else 'checkpoint'
        super().__init__()

    def exists(self, path):
        return self.fs.exists(self.get_localized_path(path)) or self.fs.isdir(path)

    def read(self, path):
        with self.fs.open(self.get_localized_path(path), 'rb') as f:
            return f.read()

    def read_serialized_keras_model(self, ckpt_path, model, custom_objects):
        """Reads the checkpoint file of the keras model into model bytes and returns the base 64
        encoded model bytes.
        :param ckpt_path: A string of path to the checkpoint file.
        :param model: A keras model. This parameter will be used in DBFSLocalStore\
            .read_serialized_keras_model() when the ckpt_path only contains model weights.
        :param custom_objects: This parameter will be used in DBFSLocalStore\
            .read_serialized_keras_model() when loading the keras model.
        :return: the base 64 encoded model bytes of the checkpoint model.
        """
        from horovod.runner.common.util import codec
        import tensorflow
        from tensorflow import keras
        from horovod.spark.keras.util import TFKerasUtil

        if LooseVersion(tensorflow.__version__) < LooseVersion("2.0.0"):
            model_bytes = self.read(ckpt_path)
            return codec.dumps_base64(model_bytes)
        else:
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(ckpt_path)
            return TFKerasUtil.serialize_model(model)

    def write_text(self, path, text):
        with self.fs.open(self.get_localized_path(path), 'w') as f:
            f.write(text)

    def is_parquet_dataset(self, path):
        try:
            dataset = self.get_parquet_dataset(path)
            return dataset is not None
        except:
            return False

    def get_parquet_dataset(self, path):
        return pq.ParquetDataset(self.get_localized_path(path), filesystem=self.fs)

    def get_train_data_path(self, idx=None):
        return '{}.{}'.format(self._train_path, idx) if idx is not None else self._train_path

    def get_val_data_path(self, idx=None):
        return '{}.{}'.format(self._val_path, idx) if idx is not None else self._val_path

    def get_test_data_path(self, idx=None):
        return '{}.{}'.format(self._test_path, idx) if idx is not None else self._test_path

    def get_data_metadata_path(self, path):
        localized_path = self.get_localized_path(path)
        if localized_path.endswith('/'):
            localized_path = localized_path[:-1] # Remove the slash at the end if there is one
        file_hash = host_hash()
        metadata_cache = localized_path+"_"+file_hash+"_cached_metadata.pkl"
        return metadata_cache

    def saving_runs(self):
        return self._save_runs

    def get_runs_path(self):
        return self._runs_path

    def get_run_path(self, run_id):
        return os.path.join(self.get_runs_path(), run_id)

    def get_checkpoint_path(self, run_id):
        return self.get_run_path(run_id) \
            if self._save_runs else None

    def get_checkpoints(self, run_id, suffix='.ckpt'):
        checkpoint_dir = self.get_localized_path(self.get_checkpoint_path(run_id))
        filenames = self.fs.ls(checkpoint_dir)
        return sorted([name for name in filenames if name.endswith(suffix)])

    def get_logs_path(self, run_id):
        return os.path.join(self.get_run_path(run_id), self.get_logs_subdir()) \
            if self._save_runs else None

    def get_checkpoint_filename(self):
        return self.checkpoint_filename

    def get_logs_subdir(self):
        return 'logs'

    def _get_full_path_or_default(self, path, default_key):
        if path is not None:
            return self.get_full_path(path)
        return self._get_path(default_key)

    def _get_path(self, key):
        return os.path.join(self.prefix_path, key)

    def get_local_output_dir_fn(self, run_id):
        @contextlib.contextmanager
        def local_run_path():
            with tempfile.TemporaryDirectory() as tmpdir:
                yield tmpdir
        return local_run_path

    def get_localized_path(self, path):
        raise NotImplementedError()

    def get_full_path(self, path):
        raise NotImplementedError()

    def get_full_path_fn(self):
        raise NotImplementedError()

    @property
    def fs(self):
        raise NotImplementedError()


class FilesystemStore(AbstractFilesystemStore):
    """Concrete filesystems store that delegates to `fsspec`."""

    def __init__(self, prefix_path, *args, **kwargs):
        self.storage_options = kwargs['storage_options'] if 'storage_options' in kwargs else {}
        self.prefix_path = prefix_path
        self._fs, self.protocol = self._get_fs_and_protocol()
        std_params = ['train_path', 'val_path', 'test_path', 'runs_path', 'save_runs', 'storage_options']
        params = dict((k, kwargs[k]) for k in std_params if k in kwargs)
        super().__init__(prefix_path, *args, **params)

    def sync_fn(self, run_id):
        run_path = self.get_run_path(run_id)

        def fn(local_run_path):
            print(f"Syncing dir {local_run_path} to dir {run_path}")
            self.copy(local_run_path, run_path, recursive=True, overwrite=True)

        return fn

    def copy(self, lpath, rpath, recursive=False, callback=_DEFAULT_CALLBACK,**kwargs):
        """
        This method copies the contents of the local source directory to the target directory.
        This is different from the fsspec's put() because it does not copy the source folder
        to the target directory in the case when target directory already exists.
        """

        from fsspec.implementations.local import LocalFileSystem, make_path_posix
        from fsspec.utils import other_paths

        rpath = (
            self.fs._strip_protocol(rpath)
            if isinstance(rpath, str)
            else [self.fs._strip_protocol(p) for p in rpath]
        )
        if isinstance(lpath, str):
            lpath = make_path_posix(lpath)
        fs = LocalFileSystem()
        lpaths = fs.expand_path(lpath, recursive=recursive)
        rpaths = other_paths(
            lpaths, rpath
        )

        callback.set_size(len(rpaths))
        for lpath, rpath in callback.wrap(zip(lpaths, rpaths)):
            callback.branch(lpath, rpath, kwargs)
            self.fs.put_file(lpath, rpath, **kwargs)

    def get_filesystem(self):
        return self.fs

    def get_localized_path(self, path):
        _, lpath = split_protocol(path)
        return lpath

    def get_full_path(self, path):
        return self.get_full_path_fn()(path)

    def get_full_path_fn(self):
        def get_path(path):
            protocol, _ = split_protocol(path)
            if protocol is not None:
                return path
            return pathlib.Path(os.path.abspath(path)).as_uri()
        return get_path

    @property
    def fs(self):
        return self._fs

    #@staticmethod
    def _get_fs_and_protocol(self):
        storage_options = self.storage_options or {}
        protocol, path = split_protocol(self.prefix_path)
        cls = fsspec.get_filesystem_class(protocol)
        options = cls._get_kwargs_from_urls(self.prefix_path)
        update_storage_options(options, storage_options)
        fs = cls(**options)
        return fs, protocol

    @classmethod
    def matches(cls, path):
        return True


class LocalStore(FilesystemStore):
    """Uses the local filesystem as a store of intermediate data and training artifacts.

    This class is deprecated and now just resolves to FilesystemStore.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class HDFSStore(AbstractFilesystemStore):
    """Uses HDFS as a store of intermediate data and training artifacts.

    Initialized from a `prefix_path` that can take one of the following forms:

    1. "hdfs://namenode01:8020/user/test/horovod"
    2. "hdfs:///user/test/horovod"
    3. "/user/test/horovod"

    The full path (including prefix, host, and port) will be used for all reads and writes to HDFS through Spark. If
    host and port are not provided, they will be omitted. If prefix is not provided (case 3), it will be prefixed to
    the full path regardless.

    The localized path (without prefix, host, and port) will be used for interaction with PyArrow. Parsed host and port
    information will be used to initialize PyArrow `HadoopFilesystem` if they are not provided through the `host` and
    `port` arguments to this initializer. These parameters will default to `default` and `0` if neither the path URL
    nor the arguments provide this information.
    """

    FS_PREFIX = 'hdfs://'
    URL_PATTERN = '^(?:(.+://))?(?:([^/:]+))?(?:[:]([0-9]+))?(?:(.+))?$'

    def __init__(self, prefix_path,
                 host=None, port=None, user=None, kerb_ticket=None,
                 driver='libhdfs', extra_conf=None, *args, **kwargs):
        prefix, url_host, url_port, path, path_offset = self.parse_url(prefix_path)
        self._check_url(prefix_path, prefix, path)
        self._url_prefix = prefix_path[:path_offset] if prefix else self.FS_PREFIX

        host = host or url_host or 'default'
        port = port or url_port or 0
        self._hdfs_kwargs = dict(host=host,
                                 port=port,
                                 user=user,
                                 kerb_ticket=kerb_ticket,
                                 extra_conf=extra_conf)
        if LooseVersion(pa.__version__) < LooseVersion('0.17.0'):
            self._hdfs_kwargs['driver'] = driver
        self._hdfs = self._get_filesystem_fn()()

        super(HDFSStore, self).__init__(prefix_path, *args, **kwargs)

    def parse_url(self, url):
        match = re.search(self.URL_PATTERN, url)
        prefix = match.group(1)
        host = match.group(2)

        port = match.group(3)
        if port is not None:
            port = int(port)

        path = match.group(4)
        path_offset = match.start(4)
        return prefix, host, port, path, path_offset

    def get_full_path(self, path):
        if not self.matches(path):
            return self._url_prefix + path
        return path

    def get_full_path_fn(self):
        prefix = self._url_prefix

        def get_path(path):
            return prefix + path
        return get_path

    @property
    def fs(self):
        return self._hdfs

    def sync_fn(self, run_id):
        class SyncState(object):
            def __init__(self):
                self.fs = None
                self.uploaded = {}

        state = SyncState()
        get_filesystem = self._get_filesystem_fn()
        hdfs_root_path = self.get_run_path(run_id)

        def fn(local_run_path):
            print(f"Syncing local dir {local_run_path} to hdfs dir {hdfs_root_path}")

            if state.fs is None:
                state.fs = get_filesystem()

            hdfs = state.fs
            uploaded = state.uploaded

            # We need to swap this prefix from the local path with the absolute path, +1 due to
            # including the trailing slash
            prefix = len(local_run_path) + 1

            for local_dir, dirs, files in os.walk(local_run_path):
                hdfs_dir = os.path.join(hdfs_root_path, local_dir[prefix:])

                for file in files:
                    local_path = os.path.join(local_dir, file)
                    modified_ts = int(os.path.getmtime(local_path))

                    if local_path in uploaded:
                        last_modified_ts = uploaded.get(local_path)
                        if modified_ts <= last_modified_ts:
                            continue

                    hdfs_path = os.path.join(hdfs_dir, file)
                    with open(local_path, 'rb') as f:
                        hdfs.upload(hdfs_path, f)
                    uploaded[local_path] = modified_ts

        return fn

    def _get_filesystem_fn(self):
        hdfs_kwargs = self._hdfs_kwargs

        def fn():
            return pa.hdfs.connect(**hdfs_kwargs)
        return fn

    def _check_url(self, url, prefix, path):
        print('_check_url: {}'.format(prefix))
        if prefix is not None and prefix != self.FS_PREFIX:
            raise ValueError('Mismatched HDFS namespace for URL: {}. Found {} but expected {}'
                             .format(url, prefix, self.FS_PREFIX))

        if not path:
            raise ValueError('Failed to parse path from URL: {}'.format(url))

    def get_localized_path(self, path):
        if self.matches(path):
            return path[len(self._url_prefix):]
        return path

    @classmethod
    def matches(cls, path):
        return path.startswith(cls.FS_PREFIX)


# If `_DBFS_PREFIX_MAPPING` is not None, map `/dbfs/...` path to `{_DBFS_PREFIX_MAPPING}/...`
# This is used in testing, and this mapping only applies to `DBFSLocalStore.get_localized_path`
_DBFS_PREFIX_MAPPING = None


class DBFSLocalStore(FilesystemStore):
    """Uses Databricks File System (DBFS) local file APIs as a store of intermediate data and
    training artifacts.

    Initialized from a `prefix_path` starts with `/dbfs/...`, `file:///dbfs/...`, `file:/dbfs/...`
    or `dbfs:/...`, see
    https://docs.databricks.com/data/databricks-file-system.html#local-file-apis.
    """

    DBFS_PATH_FORMAT_ERROR = "The provided path is not a DBFS path: {}, Please provide a path " \
                             "starting with `/dbfs/...` or `dbfs:/...` or `file:/dbfs/...` or " \
                             "`file:///dbfs/...`."

    def __init__(self, prefix_path, *args, **kwargs):
        prefix_path = self.normalize_path(prefix_path)
        if not DBFSLocalStore.matches_dbfs(prefix_path):
            raise ValueError(DBFSLocalStore.DBFS_PATH_FORMAT_ERROR.format(prefix_path))
        super(DBFSLocalStore, self).__init__(prefix_path, *args, **kwargs)

    @classmethod
    def matches_dbfs(cls, path):
        return (path.startswith("dbfs:/") and not path.startswith("dbfs://")) or \
               path.startswith("/dbfs/") or \
               path.startswith("file:///dbfs/") or \
               path.startswith("file:/dbfs/")

    @staticmethod
    def normalize_path(path):
        """
        Normalize the path to the form `/dbfs/...`
        """
        if path.startswith("dbfs:/") and not path.startswith("dbfs://"):
            return "/dbfs" + path[5:]
        elif path.startswith("/dbfs/"):
            return path
        elif path.startswith("file:///dbfs/"):
            return path[7:]
        elif path.startswith("file:/dbfs/"):
            return path[5:]
        else:
            raise ValueError(DBFSLocalStore.DBFS_PATH_FORMAT_ERROR.format(path))

    def exists(self, path):
        localized_path = self.get_localized_path(path)
        return self.fs.exists(localized_path)

    def get_localized_path(self, path):
        local_path = DBFSLocalStore.normalize_path(path)
        if _DBFS_PREFIX_MAPPING:
            # this is for testing.
            return os.path.join(_DBFS_PREFIX_MAPPING, path[6:])
        else:
            return local_path

    def get_full_path(self, path):
        return "file://" + DBFSLocalStore.normalize_path(path)

    def get_checkpoint_filename(self):
        # Use the default Tensorflow SavedModel format in TF 2.x. In TF 1.x, the SavedModel format
        # is used by providing `save_weights_only=True` to the ModelCheckpoint() callback.
        return 'checkpoint.tf'

    def read_serialized_keras_model(self, ckpt_path, model, custom_objects):
        """
        Returns serialized keras model.
        The parameter `model` is for providing the model structure when the checkpoint file only
        contains model weights.
        """
        import tensorflow
        from tensorflow import keras
        from horovod.spark.keras.util import TFKerasUtil

        if LooseVersion(tensorflow.__version__) < LooseVersion("2.0.0"):
            model.load_weights(ckpt_path)
        else:
            with keras.utils.custom_object_scope(custom_objects):
                model = keras.models.load_model(ckpt_path)
        return TFKerasUtil.serialize_model(model)
