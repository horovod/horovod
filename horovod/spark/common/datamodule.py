# Copyright (C) 2022, NVIDIA CORPORATION. All rights reserved.
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
from abc import ABC, abstractmethod


class DataModule(ABC):
    """Context manager base class for data module/loader implementations."""
    short_name = None     # implementations should provide a short name for easy reference, e.g. 'petastorm', 'nvtabular', etc.

    def __init__(self, train_dir: str, val_dir: str, num_train_epochs: int=1, has_val: bool=True,
                 train_batch_size: int=32, val_batch_size: int=32, shuffle: bool=True,
                 transform_fn=None, inmemory_cache_all=False,
                 cur_shard: int=0, shard_count: int=1, schema_fields=None, storage_options=None,
                 steps_per_epoch_train: int=1, steps_per_epoch_val: int=1, verbose=True, **kwargs):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_train_epochs = num_train_epochs
        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.transform_fn = transform_fn
        self.inmemory_cache_all = inmemory_cache_all
        self.cur_shard = cur_shard
        self.shard_count = shard_count
        self.schema_fields = schema_fields
        self.storage_options = storage_options
        self.steps_per_epoch_train = steps_per_epoch_train
        self.steps_per_epoch_val = steps_per_epoch_val
        self.verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    def train_data(self):
        """Returns the training data in a form required by the target DL framework."""
        pass

    @abstractmethod
    def val_data(self):
        """Returns the validation data in a form required by the target DL framework."""
        pass