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

import copy
import unittest
import warnings

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

import horovod.torch as hvd


class TorchElasticTests(unittest.TestCase):
    """
    Tests for utilities in horovod.torch.elastic.
    """
    def __init__(self, *args, **kwargs):
        super(TorchElasticTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_elastic_state(self):
        hvd.init()

        v = 1.0 if hvd.rank() == 0 else 2.0
        model1 = torch.nn.Sequential(torch.nn.Linear(2, 2))
        model1.load_state_dict({
            '0.weight': torch.tensor([[v, v], [v, v]]),
            '0.bias': torch.tensor([v, v])
        })

        model2 = torch.nn.Sequential(torch.nn.Linear(2, 2))
        model2.load_state_dict({
            '0.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            '0.bias': torch.tensor([0.0, 0.0])
        })

        optimizer = torch.optim.SGD(model1.parameters(), lr=0.001 * hvd.size())

        state = hvd.elastic.TorchState(model1, optimizer, batch=20 + hvd.rank(), epoch=10 + hvd.rank())
        state.sync()

        model1_weights = model1.state_dict().values()
        model2_weights = model2.state_dict().values()

        # After sync, all values should match the root rank
        for w in state.model.state_dict().values():
            np.testing.assert_allclose(w, np.ones_like(w))
        assert state.batch == 20
        assert state.epoch == 10

        # Partially modify then restore
        model1.load_state_dict(model2.state_dict())
        state.batch = 21
        state.epoch = 11

        state.restore()

        for w1, w2 in zip(model1.state_dict().values(), model1_weights):
            np.testing.assert_allclose(w1, w2)
        assert state.batch == 20
        assert state.epoch == 10

        # Partially modify then commit
        model1.load_state_dict(model2.state_dict())
        state.batch = 21
        state.epoch = 11

        state.commit()
        state.restore()

        for w1, w2 in zip(model1.state_dict().values(), model2_weights):
            np.testing.assert_allclose(w1, w2)
        assert state.batch == 21
        assert state.epoch == 11

    def test_elastic_sampler(self):
        hvd.init()

        batch_size = 2

        class ListDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, idx):
                return self.data[idx]

            def __len__(self):
                return len(self.data)

        samples_per_worker = 8
        dataset = ListDataset(list(range(samples_per_worker * hvd.size())))
        sampler = hvd.elastic.ElasticSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        state = hvd.elastic.TorchState(sampler=sampler)
        state.sync()

        assert state.sampler.epoch == 0
        assert len(state.sampler.processed_indices) == 0

        # Normal usage, no errors
        epochs = 2
        total_batches = 0
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(data_loader):
                batch_indices = sampler.get_indices(batch_idx, batch_size)
                batch_data = [dataset[idx] for idx in batch_indices]
                assert batch_data == batch.numpy().tolist()

                sampler.record_batch(batch_idx, batch_size)
                assert len(sampler.processed_indices) == batch_size * (batch_idx + 1)

                total_batches += 1
        assert total_batches == (samples_per_worker / batch_size) * epochs

        # Do not reset epoch: processed samples are retained and data loader repeats
        total_batches = 0
        for _ in enumerate(data_loader):
            assert len(sampler.processed_indices) == len(sampler)
            total_batches += 1
        assert total_batches == samples_per_worker / batch_size

        # Elastic: partial epoch + commit
        sampler.set_epoch(2)
        assert len(sampler.processed_indices) == 0

        sampler.record_batch(0, batch_size)
        sampler.record_batch(1, batch_size)
        assert len(sampler.processed_indices) == 2 * batch_size

        committed_indices = copy.copy(sampler.processed_indices)
        state.commit()

        # Elastic: partial epoch + restore
        sampler.record_batch(2, batch_size)
        sampler.record_batch(3, batch_size)
        assert len(sampler.processed_indices) == 4 * batch_size

        state.restore()

        assert len(sampler.processed_indices) == 2 * batch_size
        assert sampler.processed_indices == committed_indices

        # Elastic: sync across workers and verify non-overlap of processed samples
        sampler.record_batch(2, batch_size)
        assert len(sampler.processed_indices) == 3 * batch_size

        state.commit()
        state.sync()

        assert len(sampler.processed_indices) == 3 * batch_size * hvd.size()

        # After the sync, the remaining indices should be updated and repartitioned
        total_batches = 0
        assert len(sampler) == batch_size
        for batch_idx, batch in enumerate(data_loader):
            batch_indices = sampler.get_indices(batch_idx, batch_size)
            overlap_indices = set(batch_indices) & sampler.processed_indices
            assert overlap_indices == set()
            total_batches += 1
        assert total_batches == 1

        # Proceed to the next epoch, which should reset the state
        sampler.set_epoch(3)
        assert len(sampler) == samples_per_worker
        total_batches = 0
        for _ in enumerate(data_loader):
            total_batches += 1
        assert total_batches == samples_per_worker / batch_size
