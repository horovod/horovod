import pytorch_lightning as pl

from horovod.spark.common import constants
from horovod.spark.data_loaders.pytorch_data_loaders import (
    PytorchInfiniteAsyncDataLoader,
    PytorchInmemAsyncDataLoader)
from petastorm import TransformSpec, make_reader, make_batch_reader

PETASTORM_HDFS_DRIVER = constants.PETASTORM_HDFS_DRIVER

class PetastormDataModule(pl.LightningDataModule):
    """Default DataModule for Lightning Estimator"""
    def __init__(self, train_dir: str, val_dir: str, num_train_epochs: int=1, has_val: bool=True,
                 train_batch_size: int=32, val_batch_size: int=32, shuffle_size: int=1000,
                 num_reader_epochs=None, reader_pool_type: str="process",
                 reader_worker_count: int=2, transform_spec=None, inmemory_cache_all=False,
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
        self.num_reader_epochs = num_reader_epochs
        self.reader_pool_type = reader_pool_type
        self.reader_worker_count = reader_worker_count
        self.transform_spec = transform_spec
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

        if debug_data_loader:
            print("Creating data_module")


    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            transform_spec = TransformSpec(self.transform_spec) if self.transform_spec else None
            # In general, make_batch_reader is faster than make_reader for reading the dataset.
            # However, we found out that make_reader performs data transformations much faster than
            # make_batch_reader with parallel worker processes. Therefore, the default reader
            # we choose is make_batch_reader unless there are data transformations.
            if transform_spec:
                reader_factory = make_reader
            else:
                reader_factory = make_batch_reader

            self.train_reader = reader_factory(self.train_dir, num_epochs=self.num_reader_epochs,
                                               cur_shard=self.cur_shard, shard_count=self.shard_count,
                                               hdfs_driver=PETASTORM_HDFS_DRIVER,
                                               schema_fields=self.schema_fields,
                                               storage_options=self.storage_options,
                                               # Don't shuffle row groups without shuffling.
                                               shuffle_row_groups=True if self.shuffle_size > 0 else False)
            if self.has_val:
                self.val_reader = reader_factory(self.val_dir, num_epochs=self.num_reader_epochs,
                                                 cur_shard=self.cur_shard, shard_count=self.shard_count,
                                                 hdfs_driver=PETASTORM_HDFS_DRIVER,
                                                 schema_fields=self.schema_fields,
                                                 storage_options=self.storage_options,
                                                 shuffle_row_groups=False)

    def teardown(self, stage=None):
        if stage == "fit" or stage is None:
            if self.verbose:
                print("Tear down: closing async dataloaders")
            self.train_dl.close_async_loader()
            if self.has_val:
                self.val_dl.close_async_loader()
            if not self.inmemory_cache_all:
                # Reader was loaded once and stopped for inmemory datalaoder.
                if self.verbose:
                    print("Tear down: closing petastorm readers")
                self.train_reader.stop()
                self.train_reader.join()
                if self.has_val:
                    self.val_reader.stop()
                    self.val_reader.join()
            if self.verbose:
                print("Tear down: async dataloaders closed.")

    def train_dataloader(self):
        if self.verbose:
            print("Setup train dataloader")
        kwargs = dict(reader=self.train_reader, batch_size=self.train_batch_size,
                      name="train dataloader",
                      limit_step_per_epoch=self.steps_per_epoch_train,
                      verbose=self.verbose)
        if self.inmemory_cache_all:
            # Use inmem dataloader
            dataloader_class = PytorchInmemAsyncDataLoader
            kwargs['shuffle'] = self.shuffle_size > 0
            kwargs['num_epochs'] = self.num_train_epochs
        else:
            dataloader_class = PytorchInfiniteAsyncDataLoader
            kwargs['shuffling_queue_capacity'] = self.shuffle_size

            if self.debug_data_loader:
                kwargs['debug_data_loader'] = self.debug_data_loader

            if self.train_async_data_loader_queue_size is not None:
                if isinstance(self.train_async_data_loader_queue_size, int):
                    kwargs['async_loader_queue_size'] = self.train_async_data_loader_queue_size
                elif isinstance(self.train_async_data_loader_queue_size, float):
                    # use async data loader queue size as ratio of total steps.
                    kwargs['async_loader_queue_size'] = int(kwargs['limit_step_per_epoch'] * self.train_async_data_loader_queue_size)
                else:
                    raise RuntimeError(f"Unsupported type for train_async_data_loader_queue_size={self.train_async_data_loader_queue_size}")

        self.train_dl = dataloader_class(**kwargs)
        return self.train_dl

    def val_dataloader(self):
        if not self.has_val:
            return None
        if self.verbose:
            print("setup val dataloader")
        kwargs = dict(reader=self.val_reader, batch_size=self.val_batch_size,
                      name="val dataloader",
                      limit_step_per_epoch=self.steps_per_epoch_val,
                      verbose=self.verbose)
        if self.inmemory_cache_all:
            # Use inmem dataloader
            dataloader_class = PytorchInmemAsyncDataLoader
            kwargs['shuffle'] = False
            kwargs['num_epochs'] = self.num_train_epochs
        else:
            dataloader_class = PytorchInfiniteAsyncDataLoader
            kwargs['shuffling_queue_capacity'] = 0

            if self.debug_data_loader:
                kwargs['debug_data_loader'] = self.debug_data_loader

            if self.val_async_data_loader_queue_size is not None:
                if isinstance(self.val_async_data_loader_queue_size, int):
                    kwargs['async_loader_queue_size'] = self.val_async_data_loader_queue_size
                elif isinstance(self.val_async_data_loader_queue_size, float):
                    # use async data loader queue size as ratio of total steps.
                    kwargs['async_loader_queue_size'] = int(kwargs['limit_step_per_epoch'] * self.val_async_data_loader_queue_size)
                else:
                    raise RuntimeError(f"Unsupported type for val_async_data_loader_queue_size={self.val_async_data_loader_queue_size}")

        self.val_dl = dataloader_class(**kwargs)
        return self.val_dl
