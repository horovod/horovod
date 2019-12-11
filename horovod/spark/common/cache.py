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

import collections
import threading


class TrainingDataCache(object):
    def __init__(self):
        self.lock = threading.Lock()
        self._reset()

    def create_key(self, df, store, validation):
        return df.__hash__(), store.get_train_data_path(), store.get_val_data_path(), validation

    def set_in_use(self, key, in_use):
        if in_use:
            self._keys_in_use[key] += 1
        else:
            self._keys_in_use[key] -= 1

    def create_data_paths(self, key, store):
        _, _, _, validation = key
        idx = 0
        while True:
            train_data = store.get_train_data_path(idx)
            val_data = store.get_val_data_path(idx)

            last_key = self._data_to_key.get((train_data, val_data))
            if self._keys_in_use[last_key] > 0 and \
                    store.exists(train_data) and \
                    (not validation or store.exists(val_data)):
                # Paths are in use, try the next index
                idx += 1
                continue

            self._data_to_key[(train_data, val_data)] = key
            return train_data, val_data, idx

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        self._entries[key] = value

    def is_cached(self, key, store):
        if key not in self._entries:
            return False

        _, _, _, validation = key
        _, _, _, _, dataset_idx = self._entries.get(key)
        train_data_path = store.get_train_data_path(dataset_idx)
        val_data_path = store.get_val_data_path(dataset_idx)

        return self._keys_in_use[key] > 0 and \
            self._data_to_key.get((train_data_path, val_data_path)) == key and \
            store.exists(train_data_path) and \
            (not validation or store.exists(val_data_path))

    def clear(self):
        self._reset()

    def _reset(self):
        with self.lock:
            self._entries = {}
            self._keys_in_use = collections.Counter()
            self._data_to_key = {}
