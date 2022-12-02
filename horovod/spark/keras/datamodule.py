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
import horovod.tensorflow.keras as hvd

from horovod.spark.common import constants
from horovod.spark.common.datamodule import DataModule


class PetastormDataModule(DataModule):
    """Default Petastorm-based DataModule for KerasEstimator."""
    def __init__(self, reader_pool_type: str='thread',
                       train_reader_worker_count: int=2,
                       val_reader_worker_count: int=2,
                       make_dataset=None,
                       random_seed=0,
                       **kwargs):
        from petastorm import TransformSpec, make_reader, make_batch_reader

        super().__init__(**kwargs)
        self.reader_pool_type = reader_pool_type
        self.train_reader_worker_count = train_reader_worker_count
        self.val_reader_worker_count = val_reader_worker_count
        self.make_dataset = make_dataset
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
            self.is_batch_reader = False
        else:
            self.reader_factory = make_batch_reader
            self.is_batch_reader = True

    def __enter__(self):
        super().__enter__()
        self.train_reader = self.reader_factory(self.train_dir,
                                num_epochs=1,
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
                                **self.reader_factory_kwargs)
        self.val_reader = self.reader_factory(self.val_dir,
                                num_epochs=1,
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
                                **self.reader_factory_kwargs) if self.has_val else self.empty_batch_reader()
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
        return self.make_dataset(self.train_reader, self.train_batch_size,
                                 self.is_batch_reader, shuffle=self.shuffle, cache=self.inmemory_cache_all)

    def val_data(self):
        return self.make_dataset(self.val_reader, self.val_batch_size,
                                 self.is_batch_reader, shuffle=False, cache=self.inmemory_cache_all) if self.val_reader else None


class NVTabularDataModule(DataModule):
    """NVTabular-based DataModule for KerasEstimator for GPU-accelerated data loading of tabular datasets.

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
        import horovod.tensorflow.keras as hvd
        import numpy as np
        import tensorflow as tf
        hvd.init()
        seed = np.random.randint(0, tf.int32.max)
        seed_tensor = tf.constant(seed)
        root_seed = hvd.broadcast(seed_tensor, name="shuffle_seed", root_rank=0)
        return root_seed

    @staticmethod
    def to_dense(X, labels):
        """Convert NVTabular's ragged representation of array types to dense arrays."""
        import tensorflow as tf
        retX = {}
        for col, tensor in X.items():
            if isinstance(tensor, tuple):
                retX[col] = tf.RaggedTensor.from_row_lengths(values=tensor[0][:,0], row_lengths=tensor[1][:,0]).to_tensor()
            else:
                retX[col] = tensor
        return retX, labels

    def train_data(self):
        import horovod.tensorflow.keras as hvd
        from nvtabular.loader.tensorflow import KerasSequenceLoader, Dataset
        return KerasSequenceLoader(Dataset(self.train_dir, engine="parquet", calculate_divisions=True, **self.kwargs),
                                   batch_size=self.train_batch_size,
                                   label_names=self.label_cols,
                                   cat_names=self.categorical_cols,
                                   cont_names=self.continuous_cols,
                                   engine="parquet",
                                   shuffle=self.shuffle,
                                   buffer_size=0.1,  # how many batches to load at once
                                   parts_per_chunk=1,
                                   global_size=hvd.size(),
                                   global_rank=hvd.rank(),
                                   seed_fn=self.seed_fn).map(self.to_dense)

    def val_data(self):
        import horovod.tensorflow.keras as hvd
        from nvtabular.loader.tensorflow import KerasSequenceLoader, Dataset
        return KerasSequenceLoader(Dataset(self.train_dir, engine="parquet", calculate_divisions=True, **self.kwargs),
                                   batch_size=self.val_batch_size,
                                   label_names=self.label_cols,
                                   cat_names=self.categorical_cols,
                                   cont_names=self.continuous_cols,
                                   engine="parquet",
                                   shuffle=False,
                                   buffer_size=0.06,  # how many batches to load at once
                                   parts_per_chunk=1,
                                   global_size=hvd.size(),
                                   global_rank=hvd.rank()).map(self.to_dense) if self.has_val else self.empty_batch_reader()
