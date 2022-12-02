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
import contextlib
import glob
import horovod.torch as hvd
import os

from horovod.spark.common import constants
from horovod.spark.common.datamodule import DataModule


class PetastormDataModule(DataModule):
    """Default Petastorm-based DataModule for KerasEstimator."""

    def __init__(self,
                 reader_pool_type="thread",
                 train_reader_worker_count=10,
                 val_reader_worker_count=10,
                 random_seed=0,
                 **kwargs):
        from petastorm import TransformSpec, make_reader, make_batch_reader

        super().__init__(**kwargs)
        self.reader_pool_type = reader_pool_type
        self.train_reader_worker_count = train_reader_worker_count
        self.val_reader_worker_count = val_reader_worker_count
        self.random_seed = random_seed

        # In general, make_batch_reader is faster than make_reader for reading the dataset.
        # However, we found out that make_reader performs data transformations much faster than
        # make_batch_reader with parallel worker processes. Therefore, the default reader
        # we choose is make_batch_reader unless there are data transformations.
        self.transform_spec = TransformSpec(self.transform_fn) if self.transform_fn else None
        self.reader_factory_kwargs = dict()
        if self.transform_spec:
            self.reader_factory = make_reader
            self.reader_factory_kwargs['pyarrow_serialize'] = True
        else:
            self.reader_factory = make_batch_reader

    def __enter__(self):
        super().__enter__()
        # Petastorm: read data from the store with the correct shard for this rank
        # setting num_epochs=None will cause an infinite iterator
        # and enables ranks to perform training and validation with
        # unequal number of samples
        self.train_reader = self.reader_factory(
            self.train_dir,
            num_epochs=None,
            cur_shard=hvd.rank(),
            reader_pool_type=self.reader_pool_type,
            workers_count=self.train_reader_worker_count,
            shard_count=hvd.size(),
            hdfs_driver=constants.PETASTORM_HDFS_DRIVER,
            schema_fields=self.schema_fields,
            transform_spec=self.transform_spec,
            storage_options=self.storage_options,
            shuffle_rows=self.shuffle,
            shuffle_row_groups=self.shuffle,
            seed=self.random_seed,
            **self.reader_factory_kwargs
        )

        self.val_reader = self.reader_factory(
            self.val_dir,
            num_epochs=None,
            cur_shard=hvd.rank(),
            reader_pool_type=self.reader_pool_type,
            workers_count=self.val_reader_worker_count,
            shard_count=hvd.size(),
            hdfs_driver=constants.PETASTORM_HDFS_DRIVER,
            schema_fields=self.schema_fields,
            transform_spec=self.transform_spec,
            storage_options=self.storage_options,
            shuffle_rows=False,
            shuffle_row_groups=False,
            seed=self.random_seed,
            **self.reader_factory_kwargs
        ) if self.has_val else self.empty_batch_reader()

        return self

    def __exit__(self, type, value, traceback):
        if self.has_val and self.val_reader:
            self.val_reader.__exit__(type, value, traceback)
        if self.train_reader:
            self.train_reader.__exit__(type, value, traceback)
        super().__exit__(type, value, traceback)

    @contextlib.contextmanager
    def empty_batch_reader(self):
        yield None

    def train_data(self):
        from petastorm.pytorch import BatchedDataLoader, InMemBatchedDataLoader

        if self.inmemory_cache_all:
            train_loader = InMemBatchedDataLoader(self.train_reader,
                                                  batch_size=self.train_batch_size,
                                                  num_epochs=self.num_train_epochs,
                                                  rows_capacity=self.steps_per_epoch_train*self.train_batch_size,
                                                  shuffle=self.shuffle)
        else:
            train_loader = BatchedDataLoader(self.train_reader,
                                             batch_size=self.train_batch_size,
                                             # No need to shuffle again in dataloader level
                                             shuffling_queue_capacity=0)
        return train_loader

    def val_data(self):
        from petastorm.pytorch import BatchedDataLoader, InMemBatchedDataLoader

        if self.inmemory_cache_all:
            val_loader = InMemBatchedDataLoader(self.val_reader,
                                                batch_size=self.val_batch_size,
                                                num_epochs=self.num_train_epochs,
                                                rows_capacity=self.steps_per_epoch_val*self.val_batch_size,
                                                shuffle=False)
        else:
            val_loader = BatchedDataLoader(self.val_reader,
                                           batch_size=self.val_batch_size,
                                           shuffling_queue_capacity=0)
        return val_loader


class MapIterable():
    """Wraps an iterable with a user-defined map function for N epochs."""

    def __init__(self, data, epochs=None, map_fn=lambda x: x):
        self.data = data
        self.epochs = epochs
        self.map_fn = map_fn

    def __iter__(self):
        if self.epochs:
            for _ in range(self.epochs):
                for x in self.data:
                    yield self.map_fn(x)
        else:
            while True:
                for x in self.data:
                    yield self.map_fn(x)


class NVTabularDataModule(DataModule):
    """NVTabular-based DataModule for TorchEstimator for GPU-accelerated data loading of tabular datasets.

    Note: requires `label_cols`, `categorical_cols`, and `continuous_cols` to be explicitly provided."""

    def __init__(self, label_cols=[], categorical_cols=[], continuous_cols=[], **kwargs):
        super().__init__(**kwargs)
        self.label_cols = label_cols
        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.kwargs = kwargs

    @staticmethod
    def seed_fn():
        """
        Generate consistent dataloader shuffle seeds across workers
        Reseeds each worker's dataloader each epoch to get fresh a shuffle
        that's consistent across workers.
        """
        import numpy as np
        import torch
        hvd.init()
        seed = np.random.randint(0, torch.iinfo(torch.int32).max)
        seed_tensor = torch.tensor(seed)
        root_seed = hvd.broadcast(seed_tensor, name="shuffle_seed", root_rank=0)
        return root_seed

    def _transform(self, features_and_label):
        """Transform NVTabular value-and-offsets arrays into torch arrays."""
        import torch
        features, label = features_and_label
        for k, v in features.items():
            if isinstance(v, tuple):  # values and offsets
                indices = v[1].flatten().tolist()
                features[k] = torch.vstack(v[0].tensor_split(indices[1:]))
        return features, label

    def train_data(self):
        import nvtabular as nvt
        from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader

        train_dataset = TorchAsyncItr(
            nvt.Dataset(self.train_dir, engine='parquet', calculate_divisions=True, **self.kwargs),
            batch_size=self.train_batch_size,
            cats=self.categorical_cols,
            conts=self.continuous_cols,
            labels=self.label_cols,
            shuffle=self.shuffle,
            parts_per_chunk=1,
            global_size=hvd.size(),
            global_rank=hvd.rank(),
            seed_fn=self.seed_fn)

        train_dataloader = DLDataLoader(train_dataset, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0)
        return MapIterable(train_dataloader, epochs=self.num_train_epochs, map_fn=self._transform)

    def val_data(self):
        import nvtabular as nvt
        from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader

        val_dataset = TorchAsyncItr(
            nvt.Dataset(self.val_dir, engine='parquet', calculate_divisions=True, **self.kwargs),
            batch_size=self.val_batch_size,
            cats=self.categorical_cols,
            conts=self.continuous_cols,
            labels=self.label_cols,
            shuffle=False,
            parts_per_chunk=1,
            global_size=hvd.size(),
            global_rank=hvd.rank()) if self.has_val else None

        val_dataloader = DLDataLoader(val_dataset, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0)
        return MapIterable(val_dataloader, epochs=self.num_train_epochs, map_fn=self._transform)
