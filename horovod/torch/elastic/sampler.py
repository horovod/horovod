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

from horovod.torch.mpi_ops import rank, size


class ElasticSampler(torch.utils.data.Sampler):
    """Sampler that partitions dataset across ranks and repartitions after reset events.

    Works similar to `DistributedSampler`, but with an optional capability to record
    which dataset indices have been processed each batch. When tracked by a `TorchState`
    object, the sampler will automatically repartition the unprocessed indices among the
    new set of workers.

    In order to use this object successfully it is recommended that the user:

    1. Include this object in the `TorchState`.
    2. Call `record_batch` after processing a set of samples.
    3. Call `set_epoch` at the end of each epoch to clear the processed indices.

    Args:
        dataset: Dataset used for sampling (assumed to be of constant size).
        shuffle: If `True` (default), shuffle the indices.
        seed: Random seed used to shuffle the sampler when `shuffle=True`.
              This number should be identical across all ranks (default: 0).
    """
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
        self.processed_num = 0

        self.reset()

    def set_epoch(self, epoch):
        """Sets the epoch for this sampler.

        When `shuffle=True`, this ensures all replicas use a different random ordering
        for each epoch.

        Will clear and reset the `processed_indices` for the next epoch. It is important
        that this is called at the end of the epoch (not the beginning) to ensure that
        partially completed epochs do not reprocess samples.

        Args:
            epoch: Epoch number.
        """
        self.epoch = epoch
        self.processed_num = 0
        self.reset()

    def record_batch(self, batch_idx, batch_size):
        """Record the number of processed samples."""
        self.processed_num += (batch_size * self.num_replicas)

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        self.processed_num = state_dict["processed_num"]
        self.reset()

    def state_dict(self):
        return dict(
            epoch=self.epoch,
            processed_num=self.processed_num
        )

    def reset(self):
        self.num_replicas = size()
        self.rank = rank()

        # Exclude any samples we have already processed this epoch
        all_indices = [idx for idx in range(len(self.dataset))]
        if self.shuffle:
            # Shuffle indices across workers deterministically in place
            seed = self.seed + self.epoch
            random.Random(seed).shuffle(all_indices)
        self.remaining_indices = all_indices[self.processed_num:]

        self.num_samples = int(math.ceil(len(self.remaining_indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        self.indices = self.remaining_indices[:]

        # add extra samples to make it evenly divisible
        self.indices += self.indices[:(self.total_size - len(self.indices))]
        assert len(self.indices) == self.total_size

        # subsample
        self.indices = self.indices[self.rank:self.total_size:self.num_replicas]
        assert len(self.indices) == self.num_samples

        return iter(self.indices)

    def __len__(self):
        return self.num_samples
