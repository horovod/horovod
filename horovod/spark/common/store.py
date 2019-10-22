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
import shutil
import tempfile

import pyarrow as pa


class Store(object):
    def get_filesystem(self):
        raise NotImplementedError()

    def get_train_data_path(self):
        raise NotImplementedError()

    def get_val_data_path(self):
        raise NotImplementedError()

    def get_test_data_path(self):
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

    def get_petastorm_path_fn(self):
        raise NotImplementedError()

    def get_local_output_dir_fn(self, run_id):
        raise NotImplementedError()

    def sync_fn(self, run_id):
        raise NotImplementedError()

    def to_remote(self, run_id):
        attrs = self._remote_attrs(run_id)

        class RemoteStore(object):
            def __init__(self):
                for name, attr in attrs.items():
                    setattr(self, name, attr)

        return RemoteStore()

    def _remote_attrs(self, run_id):
        return {
            'train_data_path': self.get_train_data_path(),
            'val_data_path': self.get_val_data_path(),
            'test_data_path': self.get_test_data_path(),
            'saving_runs': self.saving_runs(),
            'runs_path': self.get_runs_path(),
            'run_path': self.get_run_path(run_id),
            'checkpoint_path': self.get_checkpoint_path(run_id),
            'logs_path': self.get_logs_path(run_id),
            'checkpoint_filename': self.get_checkpoint_filename(),
            'logs_subdir': self.get_logs_subdir(),
            'get_petastorm_path': self.get_petastorm_path_fn(),
            'get_local_output_dir': self.get_local_output_dir_fn(run_id),
            'sync': self.sync_fn(run_id)
        }


class PrefixStore(Store):
    def __init__(self, prefix_path, train_path=None, val_path=None, test_path=None, save_runs=True):
        self.prefix_path = prefix_path
        self._train_path = train_path or self._get_path('train_data')
        self._val_path = val_path or self._get_path('val_data')
        self._test_path = test_path or self._get_path('test_path')
        self._save_runs = save_runs

    def get_train_data_path(self):
        return self._train_path

    def get_val_data_path(self):
        return self._val_path

    def get_test_data_path(self):
        return self._test_path

    def saving_runs(self):
        return self._save_runs

    def get_runs_path(self):
        return self._get_path('runs')

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

    def _get_path(self, key):
        return os.path.join(self.prefix_path, key)


class LocalStore(PrefixStore):
    def __init__(self, prefix_path, *args, **kwargs):
        super(LocalStore, self).__init__(prefix_path, *args, **kwargs)
        self._fs = pa.LocalFileSystem()

    def get_filesystem(self):
        return self._fs

    def exists(self, path):
        return os.path.exists(path)

    def read(self, path):
        with open(path, 'rb') as f:
            return f.read()

    def get_petastorm_path_fn(self):
        def get_path(path):
            return 'file://' + path
        return get_path

    def get_local_output_dir_fn(self, run_id):
        run_path = self.get_run_path(run_id)

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


class HDFSStore(PrefixStore):
    def __init__(self, prefix_path,
                 host='default', port=0, user=None, kerb_ticket=None,
                 driver='libhdfs', extra_conf=None, *args, **kwargs):
        super(HDFSStore, self).__init__(prefix_path, *args, **kwargs)

        self._host = host
        self._port = port
        self._user = user
        self._kerb_ticket = kerb_ticket
        self._driver = driver
        self._extra_conf = extra_conf
        self._hdfs = self._get_filesystem_fn()()

    def get_filesystem(self):
        return self._hdfs

    def exists(self, path):
        return self._hdfs.exists(path)

    def read(self, path):
        with self._hdfs.open(path, 'rb') as f:
            return f.read()

    def get_petastorm_path_fn(self):
        def get_path(path):
            return 'hdfs://' + path
        return get_path

    def get_local_output_dir_fn(self, run_id):
        @contextlib.contextmanager
        def local_run_path():
            dirpath = tempfile.mkdtemp()
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
        host = self._host
        port = self._port
        user = self._user
        kerb_ticket = self._kerb_ticket
        driver = self._driver
        extra_conf = self._extra_conf

        def fn():
            return pa.hdfs.connect(host=host,
                                   port=port,
                                   user=user,
                                   kerb_ticket=kerb_ticket,
                                   driver=driver,
                                   extra_conf=extra_conf)
        return fn
