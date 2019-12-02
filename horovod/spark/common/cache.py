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

import threading


class TrainingDataCache(object):
    def __init__(self):
        self.lock = threading.Lock()
        self._reset()

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        self._entries[key] = value

    def is_cached(self, key):
        dataframe_hash, validation_split, validation_col, train_data_path, val_data_path = key

        content = self._entries.get(key, None)
        if not content:
            return False

        return \
            content.dataframe_hash == dataframe_hash and \
            content.validation_split == validation_split and \
            content.validation_col == validation_col and \
            content.store.exists(train_data_path) and \
            (validation_split == 0.0 or content.store.exists(val_data_path))

    def clear(self):
        self._reset()

    def _reset(self):
        with self.lock:
            self._entries = {}


class CacheEntry(object):
    def __init__(self, store, dataframe_hash, validation_split, validation_col,
                 train_rows, val_rows, metadata, avg_row_size):
        self.store = store
        self.dataframe_hash = dataframe_hash
        self.validation_split = validation_split
        self.validation_col = validation_col
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.metadata = metadata
        self.avg_row_size = avg_row_size
