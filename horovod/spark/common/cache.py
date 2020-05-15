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

import collections
import contextlib
import threading


class TrainingDataCache(object):
    def __init__(self):
        self.lock = threading.Lock()
        self._reset()

    def create_key(self, df, store, validation):
        return df.__hash__(), store.get_train_data_path(), store.get_val_data_path(), validation

    @contextlib.contextmanager
    def use_key(self, key):
        self._keys_in_use[key] += 1
        try:
            yield
        finally:
            self._keys_in_use[key] -= 1

    def next_dataset_index(self, key):
        """Finds the next available `dataset_idx` given a key.

        Indices start a 0 and go up until the first unused index is found.

        Will attempt to reuse earlier indices if they are no longer in use. This balances between
        supporting multiple concurrent datasets being trained at once (multiple dataset indices),
        and avoiding overuse of disk space (reclaiming unused datasets when no longer needed).

        NOTE: this method is not thread-safe. You must wrap usage with `cache.lock` if using
        in a multi-threaded setting (see `prepare_data`).
        """
        idx = 0
        while True:
            last_key = self._dataset_to_key.get(idx)
            if self._keys_in_use[last_key] > 0:
                # Paths are in use, try the next index
                idx += 1
                continue

            self._dataset_to_key[idx] = key
            self._key_to_dataset[key] = idx
            return idx

    def get_dataset(self, key):
        return self._key_to_dataset[key]

    def get_dataset_properties(self, dataset_idx):
        return self._dataset_properties[dataset_idx]

    def set_dataset_properties(self, dataset_idx, props):
        self._dataset_properties[dataset_idx] = props

    def is_cached(self, key, store):
        """Returns true if the key is in the cache and its paths exist in the store already."""
        if key not in self._key_to_dataset:
            return False

        dataset_idx = self._key_to_dataset[key]
        _, _, _, validation = key
        train_data_path = store.get_train_data_path(dataset_idx)
        val_data_path = store.get_val_data_path(dataset_idx)

        return self._keys_in_use[key] > 0 and \
            self._dataset_to_key.get(dataset_idx) == key and \
            store.is_parquet_dataset(train_data_path) and \
            (not validation or store.is_parquet_dataset(val_data_path))

    def clear(self):
        self._reset()

    def _reset(self):
        with self.lock:
            self._keys_in_use = collections.Counter()
            self._key_to_dataset = {}
            self._dataset_to_key = {}
            self._dataset_properties = {}
