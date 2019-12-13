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
from __future__ import print_function

import contextlib
import os
import re
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


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

    def get_parquet_dataset(self, path):
        raise NotImplementedError()

    def get_train_data_path(self, idx=None):
        raise NotImplementedError()

    def get_val_data_path(self, idx=None):
        raise NotImplementedError()

    def get_test_data_path(self, idx=None):
        raise NotImplementedError()

    def saving_runs(self):
        raise NotImplementedError()

    def get_runs_path(self):
        raise NotImplementedError()

    def get_run_path(self, run_id):
        raise NotImplementedError()

    def get_checkpoint_path(self, run_id):
        raise NotImplementedError()

    def get_logs_path(self, run_id):
        raise NotImplementedError()

    def get_checkpoint_filename(self):
        raise NotImplementedError()

    def get_logs_subdir(self):
        raise NotImplementedError()

    def exists(self, path):
        raise NotImplementedError()

    def read(self, path):
        raise NotImplementedError()

    def get_local_output_dir_fn(self, run_id):
        raise NotImplementedError()

    def sync_fn(self, run_id):
        raise NotImplementedError()

    def to_remote(self, run_id, dataset_idx):
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
        else:
            return LocalStore(prefix_path, *args, **kwargs)


class PrefixStore(Store):
    def __init__(self, prefix_path, train_path=None, val_path=None, test_path=None, runs_path=None, save_runs=True):
        self.prefix_path = self.get_full_path(prefix_path)
        self._train_path = train_path or self._get_path('intermediate_train_data')
        self._val_path = val_path or self._get_path('intermediate_val_data')
        self._test_path = test_path or self._get_path('intermediate_test_data')
        self._runs_path = runs_path or self._get_path('runs')
        self._save_runs = save_runs
        super(PrefixStore, self).__init__()

    def exists(self, path):
        return self.get_filesystem().exists(self.get_localized_path(path))

    def read(self, path):
        with self.get_filesystem().open(self.get_localized_path(path), 'rb') as f:
            return f.read()

    def get_parquet_dataset(self, path):
        return pq.ParquetDataset(self.get_localized_path(path), filesystem=self.get_filesystem())

    def get_train_data_path(self, idx=None):
        return '{}.{}'.format(self._train_path, idx) if idx is not None else self._train_path

    def get_val_data_path(self, idx=None):
        return '{}.{}'.format(self._val_path, idx) if idx is not None else self._val_path

    def get_test_data_path(self, idx=None):
        return '{}.{}'.format(self._test_path, idx) if idx is not None else self._test_path

    def saving_runs(self):
        return self._save_runs

    def get_runs_path(self):
        return self._runs_path

    def get_run_path(self, run_id):
        return os.path.join(self.get_runs_path(), run_id)

    def get_checkpoint_path(self, run_id):
        return os.path.join(self.get_run_path(run_id), self.get_checkpoint_filename()) \
            if self._save_runs else None

    def get_logs_path(self, run_id):
        return os.path.join(self.get_run_path(run_id), self.get_logs_subdir()) \
            if self._save_runs else None

    def get_checkpoint_filename(self):
        return 'checkpoint.h5'

    def get_logs_subdir(self):
        return 'logs'

    def get_full_path(self, path):
        if not self.matches(path):
            return self.path_prefix() + path
        return path

    def get_localized_path(self, path):
        if self.matches(path):
            return path[len(self.path_prefix()):]
        return path

    def get_full_path_fn(self):
        prefix = self.path_prefix()

        def get_path(path):
            return prefix + path
        return get_path

    def _get_path(self, key):
        return os.path.join(self.prefix_path, key)

    def path_prefix(self):
        raise NotImplementedError()

    def get_filesystem(self):
        raise NotImplementedError()

    @classmethod
    def matches(cls, path):
        return path.startswith(cls.filesystem_prefix())

    @classmethod
    def filesystem_prefix(cls):
        raise NotImplementedError()


class LocalStore(PrefixStore):
    FS_PREFIX = 'file://'

    def __init__(self, prefix_path, *args, **kwargs):
        self._fs = pa.LocalFileSystem()
        super(LocalStore, self).__init__(prefix_path, *args, **kwargs)

    def path_prefix(self):
        return self.FS_PREFIX

    def get_filesystem(self):
        return self._fs

    def get_local_output_dir_fn(self, run_id):
        run_path = self.get_localized_path(self.get_run_path(run_id))

        @contextlib.contextmanager
        def local_run_path():
            if not os.path.exists(run_path):
                try:
                    os.makedirs(run_path)
                except OSError:
                    # Race condition from workers on the same host: ignore
                    pass
            yield run_path

        return local_run_path

    def sync_fn(self, run_id):
        def fn(root_path):
            pass
        return fn

    @classmethod
    def filesystem_prefix(cls):
        return cls.FS_PREFIX


class HDFSStore(PrefixStore):
    FS_PREFIX = 'hdfs://'
    HDFS_URL_PATTERN = '^(?:(.+://))?(?:([^/:]+))?(?:[:]([0-9]+))?(?:(.+))?$'

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
    def __init__(self, prefix_path,
                 host=None, port=None, user=None, kerb_ticket=None,
                 driver='libhdfs', extra_conf=None, temp_dir=None, *args, **kwargs):
        self._temp_dir = temp_dir

        prefix, url_host, url_port, path, path_offset = self.parse_url(prefix_path)
        self._check_url(prefix_path, prefix, path)
        self._url_prefix = prefix_path[:path_offset] if prefix else self.FS_PREFIX

        host = host or url_host or 'default'
        port = port or url_port or 0
        self._hdfs_kwargs = dict(host=host,
                                 port=port,
                                 user=user,
                                 kerb_ticket=kerb_ticket,
                                 driver=driver,
                                 extra_conf=extra_conf)
        self._hdfs = self._get_filesystem_fn()()

        super(HDFSStore, self).__init__(prefix_path, *args, **kwargs)

    def parse_url(self, url):
        match = re.search(self.HDFS_URL_PATTERN, url)
        prefix = match.group(1)
        host = match.group(2)

        port = match.group(3)
        if port is not None:
            port = int(port)

        path = match.group(4)
        path_offset = match.start(4)
        return prefix, host, port, path, path_offset

    def path_prefix(self):
        return self._url_prefix

    def get_filesystem(self):
        return self._hdfs

    def get_local_output_dir_fn(self, run_id):
        temp_dir = self._temp_dir

        @contextlib.contextmanager
        def local_run_path():
            dirpath = tempfile.mkdtemp(dir=temp_dir)
            try:
                yield dirpath
            finally:
                shutil.rmtree(dirpath)

        return local_run_path

    def sync_fn(self, run_id):
        class SyncState(object):
            def __init__(self):
                self.fs = None
                self.uploaded = {}

        state = SyncState()
        get_filesystem = self._get_filesystem_fn()
        hdfs_root_path = self.get_run_path(run_id)

        def fn(root_path):
            if state.fs is None:
                state.fs = get_filesystem()

            hdfs = state.fs
            uploaded = state.uploaded

            # We need to swap this prefix from the local path with the absolute path, +1 due to
            # including the trailing slash
            prefix = len(root_path) + 1

            for local_dir, dirs, files in os.walk(root_path):
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

    @classmethod
    def filesystem_prefix(cls):
        return cls.FS_PREFIX
