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

from petastorm.pytorch import BatchedDataLoader
from horovod.data import BaseDataLoader, AsyncDataLoaderMixin


class PytorchDataLoader(BaseDataLoader):
    def __init__(self, reader, batch_size, shuffling_queue_capacity):
        self.reader = reader
        self.batch_size = batch_size
        self.shuffling_queue_capacity = shuffling_queue_capacity
        print(f"Initializing petastorm dataloader with batch_size {batch_size}"
              f" and shuffling_queue_capacity {shuffling_queue_capacity}")

    def __len__(self):
        return len(self.reader)

    def _iterate(self):
        # Reset the reader if needed.
        if self.reader.last_row_consumed:
            print(f"Resetting Petastorm reader for {self.reader.dataset.paths}")
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

        for batch in data_loader:
            yield batch


class PytorchAsyncDataLoader(AsyncDataLoaderMixin, PytorchDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PytorchInfiniteDataLoader(BaseDataLoader):
    def __init__(self, reader, batch_size, shuffling_queue_capacity):
        from petastorm.pytorch import BatchedDataLoader
        self.reader = reader
        self.batch_size = batch_size
        self.shuffling_queue_capacity = shuffling_queue_capacity

        self.data_loader = BatchedDataLoader(
            self.reader,
            batch_size=self.batch_size,
            shuffling_queue_capacity=self.shuffling_queue_capacity)
        self.iterater = iter(self.data_loader)

        print(f"Initializing petastorm dataloader with batch_size {batch_size}"
              f" and shuffling_queue_capacity {shuffling_queue_capacity}")

    def __len__(self):
        return len(self.reader)

    def _iterate(self):
        while True:
            yield next(self.iterater)


class PytorchInfiniteAsyncDataLoader(AsyncDataLoaderMixin, PytorchInfiniteDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

