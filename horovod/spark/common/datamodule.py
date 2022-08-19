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
import importlib
from abc import ABC, abstractmethod


_data_modules = {}


class DataModule(ABC):
    """Context manager base class for data module/loader implementations."""
    short_name = None     # implementations should provide a short name for easy reference, e.g. 'petastorm', 'nvtabular', etc.

    def __init__(self, train_dir: str, val_dir: str, num_train_epochs: int=1, has_val: bool=True,
                 train_batch_size: int=32, val_batch_size: int=32, shuffle_size: int=1000,
                 transform_fn=None, inmemory_cache_all=False,
                 cur_shard: int=0, shard_count: int=1, schema_fields=None, storage_options=None,
                 steps_per_epoch_train: int=1, steps_per_epoch_val: int=1, verbose=True,
                 debug_data_loader: bool=False, train_async_data_loader_queue_size: int=None,
                 val_async_data_loader_queue_size: int=None, **kwargs):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.num_train_epochs = num_train_epochs
        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle_size = shuffle_size
        self.transform_fn = transform_fn
        self.inmemory_cache_all = inmemory_cache_all
        self.cur_shard = cur_shard
        self.shard_count = shard_count
        self.schema_fields = schema_fields
        self.storage_options = storage_options
        self.steps_per_epoch_train = steps_per_epoch_train
        self.steps_per_epoch_val = steps_per_epoch_val
        self.verbose = verbose
        self.debug_data_loader = debug_data_loader
        self.train_async_data_loader_queue_size = train_async_data_loader_queue_size
        self.val_async_data_loader_queue_size = val_async_data_loader_queue_size

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    def train_data(self, reader=None):
        """Returns the training data in a form required by the target DL framework."""
        pass

    @abstractmethod
    def val_data(self, reader=None):
        """Returns the validation data in a form required by the target DL framework."""
        pass

    @classmethod
    def register(cls):
        """Adds this DataModule implementation to a global registry keyed by short name."""
        _data_modules[cls.short_name] = cls


def datamodule_from_name(module_name):
    """Returns a DataModule implementation (the class) associated with a name.

    Alternatively, implementations of DataModule can be referenced by their fully qualified class names, e.g. `horovod.spark.keras.datamodule.PetastormDataModule`
    """
    if module_name in _data_modules:
        # return data module class from registry
        return _data_modules[module_name]
    else:
        # otherwise, try to dynamically import data module
        try:
            splits = module_name.split('.')
            module_name, class_name = '.'.join(splits[:-1]), splits[-1]
            module = importlib.import_module(m)
            return getattr(module, c)
        except Exception as e:
            print("Unable to dynamically load data module: {}".format(module_name))
            raise(e)
