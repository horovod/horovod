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

    def create_key(self, df, store, validation):
        return df.__hash__(), store.get_train_data_path(), store.get_val_data_path(), validation

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, value):
        self._entries[key] = value

    def is_cached(self, key):
        dataframe_hash, train_data_path, val_data_path, validation = key

        content = self._entries.get(key, None)
        if not content:
            return False

        return \
            content.dataframe_hash == dataframe_hash and \
            content.validation == validation and \
            content.store.exists(train_data_path) and \
            (not validation or content.store.exists(val_data_path))

    def clear(self):
        self._reset()

    def _reset(self):
        with self.lock:
            self._entries = {}
