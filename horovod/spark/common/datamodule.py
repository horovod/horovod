import importlib


_data_modules = {}
class DataModule:
    """Context manager base class for data module/loader implementation.s"""
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

    def train_data(self, reader=None):
        raise NotImplementedError()

    def val_data(self, reader=None):
        raise NotImplementedError()


def register_datamodule(module_name, data_module):
    """Registers a DataModule implementation with a string key, e.g. 'petastorm', 'nvtabular'."""
    _data_modules[module_name] = data_module

def datamodule_from_name(module_name):
    """Returns a DataModule implementation associated with a string key."""
    if module_name in _data_modules:
        return _data_modules[module_name]
    else:
        # try dynamic import
        module = importlib.import_module(module_name)
        module.register()
        return _data_modules[module_name]
