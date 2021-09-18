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

from petastorm.pytorch import BatchedDataLoader, InMemBatchedDataLoader
from horovod.data import BaseDataLoader, AsyncDataLoaderMixin


class PytorchDataLoader(BaseDataLoader):
    def __init__(self, reader, batch_size, shuffling_queue_capacity, name="",
                 limit_step_per_epoch=-1, verbose=False):
        self.reader = reader
        self.batch_size = batch_size
        self.shuffling_queue_capacity = shuffling_queue_capacity
        self.limit_step_per_epoch = limit_step_per_epoch
        self.name = name
        self.verbose = verbose

        print(f"[{self.name}]: Initializing petastorm dataloader with batch_size={batch_size}"
              f"shuffling_queue_capacity={shuffling_queue_capacity}, "
              f"limit_step_per_epoch={limit_step_per_epoch}")

    def __len__(self):
        # We cannot infer length from reader.
        return self.limit_step_per_epoch if self.limit_step_per_epoch != -1 else 0

    def _iterate(self):
        # Reset the reader if needed.
        if self.reader.last_row_consumed:
            self._print_verbose(f"[{self.name}]: Resetting Petastorm reader for {self.reader.dataset.paths}")
            self.reader.reset()

        # Re-create the data loader for each iteration. This is needed becasue there may be
        # some left-over data from last epoch which can cause petastorm's BatchedDataLoader
        # fail to start new iteration. To workaround the issue, we have to re-create the data
        # loader at each new iterration starts.
        data_loader = BatchedDataLoader(
            self.reader,
            batch_size=self.batch_size,
            shuffling_queue_capacity=self.shuffling_queue_capacity,
        )

        num_steps = 0

        self._print_verbose(f"[{self.name}]: Start to generate batch data. limit_step_per_epoch={self.limit_step_per_epoch}")

        for batch in data_loader:
            if num_steps == self.limit_step_per_epoch:
                self._print_verbose(f"[{self.name}]: Reach limit_step_per_epoch. Stop at step {num_steps}.")
                break

            num_steps += 1
            yield batch

    def _print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class PytorchAsyncDataLoader(AsyncDataLoaderMixin, PytorchDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PytorchInfiniteDataLoader(PytorchDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.reader.num_epochs is not None:
            raise ValueError("Need to set num_epochs as None in reader.")

        self.data_loader = BatchedDataLoader(
            self.reader,
            batch_size=self.batch_size,
            shuffling_queue_capacity=self.shuffling_queue_capacity)
        self.iterator = iter(self.data_loader)

    def _iterate(self):
        num_steps = 0
        self._print_verbose(f"[{self.name}]: Start to generate batch data. limit_step_per_epoch={self.limit_step_per_epoch}")

        while True:
            if num_steps == self.limit_step_per_epoch:
                self._print_verbose(f"[{self.name}]: Reach limit_step_per_epoch. Stop at step {num_steps}.")
                break
            num_steps += 1

            yield next(self.iterator)


class PytorchInfiniteAsyncDataLoader(AsyncDataLoaderMixin, PytorchInfiniteDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PytorchInmemDataLoader(BaseDataLoader):
    def __init__(self, reader, batch_size, num_epochs, name="",
                 shuffle=False, limit_step_per_epoch=-1, verbose=False):
        self.batch_size = batch_size
        self.limit_step_per_epoch = limit_step_per_epoch
        self.name = name
        self.verbose = verbose

        if limit_step_per_epoch == -1:
            raise ValueError('limit_step_per_epoch cannot be -1 for inmem dataloader')

        print(f"[{self.name}]: Initializing petastorm inmem_dataloader with batch_size={batch_size}"
              f"num_epochs={num_epochs}, "
              f"shuffle={shuffle}"
              f"limit_step_per_epoch={limit_step_per_epoch}")

        self.dataloader = InMemBatchedDataLoader(reader, batch_size=batch_size, num_epochs=num_epochs,
                                                 rows_capacity=batch_size*limit_step_per_epoch, shuffle=shuffle)
        self.iterator = iter(self.dataloader)

    def __len__(self):
        # We cannot infer length from reader.
        return self.limit_step_per_epoch

    def _iterate(self):
        num_steps = 0
        self._print_verbose(f"[{self.name}]: Start to generate batch data. limit_step_per_epoch={self.limit_step_per_epoch}")

        while True:
            if num_steps == self.limit_step_per_epoch:
                self._print_verbose(f"[{self.name}]: Reach limit_step_per_epoch. Stop at step {num_steps}.")
                break
            num_steps += 1

            yield next(self.iterator)

    def _print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class PytorchInmemAsyncDataLoader(AsyncDataLoaderMixin, PytorchInmemDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PetastormBatchedDataLoader(BatchedDataLoader):
    def __init__(self, name="", limit_step_per_epoch=-1, verbose=False, *args, **kwargs):
        print(f"[{name}]Petastorm BatchedDataLoader will ignore limit_step_per_epoch and verbose.")
        super().__init__(*args, **kwargs)
