# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import itertools
import unittest
from packaging import version

import pytest
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from base_test_mxnet import *


@pytest.mark.skipif(version.parse(mx.__version__).major != 2, reason='MXNet v2 tests')
class MX2Tests(MXTests, unittest.TestCase):
    """
    Tests for ops in horovod.mxnet. This tests MXNet 2.x specifically.
    """

    def test_horovod_broadcast_deferred_init_parameters(self):
        """Test that the deferred initialized parameters are broadcasted."""
        hvd.init()
        root_rank = 0
        rank = hvd.rank()

        # This test does not apply if there is only one worker.
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        mx.np.random.seed(rank)
        layer = mx.gluon.nn.Conv2D(10, 2)
        layer.initialize()
        hvd.broadcast_parameters(layer.collect_params(), root_rank=root_rank)

        x = mx.np.ones((5, 4, 10, 10))
        layer(x)
        tensors = [p.data() for _, p in sorted(layer.collect_params().items())]
        root_tensors = []
        for tensor in tensors:
            root_tensors.append(hvd.broadcast(tensor, root_rank=root_rank))

        for tensor, root_tensor in zip(tensors, root_tensors):
            assert same(tensor.asnumpy(), root_tensor.asnumpy()), \
                'horovod did not broadcast deferred initialized parameter correctly'

    def test_two_trainer(self):
        """Test using horovod allreduce in MXNet Gluon trainer."""
        from mxnet import gluon
        from mxnet.gluon import Block, nn, HybridBlock

        hvd.init()
        rank = hvd.rank()
        ctx = mx.cpu(rank)

        net1 = nn.Dense(20, in_units=10)
        net2 = nn.Dense(30, in_units=10)
        net1.initialize(ctx=ctx)
        net2.initialize(ctx=ctx)

        params1 = net1.collect_params()
        params2 = net2.collect_params()
        hvd.broadcast_parameters(params1, prefix="net1")
        hvd.broadcast_parameters(params2, prefix="net2")
        trainer1 = hvd.DistributedTrainer(params1, 'sgd', {'learning_rate': 0.1}, prefix="net1")
        trainer2 = hvd.DistributedTrainer(params2, 'sgd', {'learning_rate': 0.1}, prefix="net2")

        for i in range(10):
            data = mx.np.ones((5, 10), ctx=ctx)
            with mx.autograd.record():
                pred1 = net1(data).sum()
                pred2 = net2(data).sum()
            mx.autograd.backward([pred1, pred2])
            trainer1.step(1.0)
            trainer2.step(1.0)
            l = pred1.item() + pred2.item()

    @unittest.skipUnless(has_gpu, "no gpu detected")
    def test_gluon_trainer(self):
        """Test using horovod allreduce in MXNet Gluon trainer."""
        from mxnet import gluon
        from mxnet.gluon import Block, nn, HybridBlock

        hvd.init()
        rank = hvd.rank()
        np.random.seed(1000 + 10 * rank)
        mx.np.random.seed(1000 + 10 * rank)
        ctx = mx.gpu(rank)

        def gen_random_dataset(batch_size=64, dim=32, min_len=20, max_len=100,
                               size=1000):
            for _ in range(size):
                length = np.random.randint(min_len, max_len + 1)
                rand_src = mx.np.random.normal(0, 1, (length, dim))
                rand_dst = mx.np.random.normal(0, 1, (length, dim))
                yield rand_src, rand_dst

        class SimpleNet(HybridBlock):
            def __init__(self, layer_num=6, **kwargs):
                super(SimpleNet, self).__init__(**kwargs)
                self._layer_num = layer_num
                self.ln_l = nn.HybridSequential()
                self.dense_l = nn.HybridSequential()
                for i in range(layer_num):
                    self.dense_l.add(nn.Dense(units=32 + layer_num - 1 - i,
                        flatten=False))
                    self.ln_l.add(nn.LayerNorm())

            def forward(self, data):
                """

                Parameters
                ----------
                data :
                    Shape (batch_size, seq_len, fea_dim)

                Returns
                -------
                out :
                    Shape (batch_size, seq_len, fea_dim)
                """
                for i in range(self._layer_num):
                   data = self.ln_l[i](data)
                   data = self.dense_l[i](data)
                return data

        net = SimpleNet()
        net.initialize(ctx=ctx)
        net.hybridize(static_alloc=True)

        params = net.collect_params()
        cnt = 0
        lr = 1E-4
        trainer = gluon.Trainer(params, 'adam', {'learning_rate': lr},
            update_on_kvstore=False)

        data_gen = gen_random_dataset()
        for (src_data, dst_data) in data_gen:
            src_data = src_data.as_in_context(ctx).astype(np.float32)
            dst_data = dst_data.as_in_context(ctx).astype(np.float32)
            with mx.autograd.record():
                pred = net(src_data)
                loss = mx.np.abs(pred - dst_data).mean()
                loss.backward()
            # Begin to update the parameter
            trainer.step(1.0)
            cnt += 1
            l = loss.item()
            if cnt >= 10:
                for key, param in params.items():
                    hvd.allreduce_(param.list_data()[0])
                cnt = 0


if __name__ == '__main__':
    unittest.main()
