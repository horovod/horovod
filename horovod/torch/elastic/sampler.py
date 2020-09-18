# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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

import math
import random

import torch.utils.data.distributed

import horovod.torch as hvd


class ElasticSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed

        self.epoch = 0
        self.processed_indices = set()

        self.num_replicas = 0
        self.rank = 0
        self.remaining_indices = []
        self.num_samples = 0
        self.total_size = 0

        self.reset()

    def reset(self):
        self.num_replicas = hvd.size()
        self.rank = hvd.rank()

        # Exclude any samples we have already processed this epoch
        self.remaining_indices = [idx for idx in range(len(self.dataset))
                                  if idx not in self.processed_indices]

        self.num_samples = int(math.ceil(len(self.remaining_indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.processed_indices = set()
        self.reset()

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.processed_indices = state_dict['processed_indices']
        self.reset()

    def state_dict(self):
        return dict(
            epoch=self.epoch,
            processed_indices=self.processed_indices
        )

    def get_indices(self, offset, length):
        start_idx = offset * length
        end_idx = min(start_idx + length, len(self.indices))
        return self.indices[start_idx:end_idx]

    def record_indices(self, batch_idx, batch_size):
        processed = set(self.get_indices(batch_idx, batch_size))
        self.processed_indices.update(processed)

    def __iter__(self):
        self.indices = self.remaining_indices[:]
        if self.shuffle:
            # Shuffle indices across workers deterministically in place
            seed = self.seed + self.epoch
            random.Random(seed).shuffle(self.indices)

        # add extra samples to make it evenly divisible
        self.indices += self.indices[:(self.total_size - len(self.indices))]
        assert len(self.indices) == self.total_size

        # subsample
        self.indices = self.indices[self.rank:self.total_size:self.num_replicas]
        assert len(self.indices) == self.num_samples

        return iter(self.indices)

    def __len__(self):
        return self.num_samples
