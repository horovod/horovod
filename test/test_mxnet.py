# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import horovod.mxnet as hvd
import itertools
import mxnet as mx
import os
import unittest
from mxnet.base import MXNetError
from mxnet.test_utils import same


has_gpu = mx.context.num_gpus() > 0


class MXTests(unittest.TestCase):
    """
    Tests for ops in horovod.mxnet.
    """

    def _current_context(self):
        if has_gpu:
            return mx.gpu(hvd.local_rank())
        else:
            return mx.current_context()

    def test_horovod_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            # MXNet uses gpu_id as part of the seed, so to get identical seeds
            # we must set a context.
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            summed = hvd.allreduce(tensor, average=False, name=str(count))
            multiplied = tensor * size
            max_difference = mx.nd.max(mx.nd.subtract(summed, multiplied))
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            if max_difference > threshold:
                print("allreduce", count, dtype, dim, max_difference,
                      threshold)
                print("tensor", hvd.rank(), tensor)
                print("summed", hvd.rank(), summed)
                print("multiplied", hvd.rank(), multiplied)
            assert max_difference <= threshold, 'hvd.allreduce produces \
                                                 incorrect results'

    def test_horovod_allreduce_average(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            averaged = hvd.allreduce(tensor, average=True, name=str(count))
            tensor *= size
            tensor /= size
            max_difference = mx.nd.max(mx.nd.subtract(averaged, tensor))
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 1
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            if max_difference > threshold:
                print("average", count, dtype, dim, max_difference, threshold)
                print("tensor", hvd.rank(), tensor)
                print("averaged", hvd.rank(), averaged)
            assert max_difference <= threshold, 'hvd.allreduce produces \
                                                 incorrect results for average'

    def test_horovod_allreduce_inplace(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = ['int32',   'int64',
                  'float32', 'float64'] 
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=ctx)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim],
                                          ctx=ctx)
            tensor = tensor.astype(dtype)
            multiplied = tensor * size
            hvd.allreduce_(tensor, average=False, name=str(count))
            max_difference = mx.nd.max(mx.nd.subtract(tensor, multiplied))
            count += 1

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in ['int32', 'int64']:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            if max_difference > threshold:
                print("self", count, dtype, dim, max_difference, threshold)
                print("tensor", hvd.rank(), tensor)
                print("multiplied", hvd.rank(), multiplied)
            assert max_difference <= threshold, 'hvd.allreduce produces \
                                                 incorrect results for self'

    def test_horovod_allreduce_error(self):
        """Test that the allreduce raises an error if different ranks try to
           send tensors of different rank or dimension."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # Same rank, different dimension
        ctx = self._current_context()

        shape = (17 + rank, 3)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw error'
        except (MXNetError, RuntimeError):
            pass

        # Same number of elements, different rank
        if rank == 0:
            shape = (17, 23 * 57)
        else:
            shape = (17, 23, 57)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_allreduce_type_error(self):
        """Test that the allreduce raises an error if different ranks try to
           send tensors of different type."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        ctx = self._current_context()
        shape = (17, 3)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        if rank % 2 == 0:
            tensor = tensor.astype('int32')
        else:
            tensor = tensor.astype('float32')

        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw error'
        except (MXNetError, RuntimeError):
            pass

    @unittest.skipUnless(has_gpu, "no gpu detected")
    def test_horovod_allreduce_cpu_gpu_error(self):
        """Test that the allreduce raises an error if different ranks try to
           perform reduction on CPU and GPU."""
        if os.environ.get('HOROVOD_MIXED_INSTALL'):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            return

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        shape = (17, 17, 17)
        if rank % 2 == 0:
            ctx = mx.gpu(hvd.rank())
        else:
            ctx = mx.cpu(hvd.rank())
        tensor = mx.nd.ones(shape=shape, ctx=ctx)

        try:
            output = hvd.allreduce(tensor)
            output.wait_to_read()
            assert False, 'hvd.allreduce did not throw cpu-gpu error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        dtypes = ['int32',   'int64',
                  'float32', 'float64'] 
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims,
                                                       root_ranks):
            tensor = mx.nd.ones(shapes[dim], ctx=ctx) * rank
            root_tensor = mx.nd.ones(shapes[dim], ctx=ctx) * root_rank
            tensor = tensor.astype(dtype)
            root_tensor = root_tensor.astype(dtype)

            broadcast_tensor = hvd.broadcast(tensor, root_rank=root_rank,
                                             name=str(count))
            if rank != root_rank:
                if same(tensor.asnumpy(), root_tensor.asnumpy()):
                    print("broadcast", count, dtype, dim,
                          mx.nd.max(tensor == root_tensor))
                    print("tensor", hvd.rank(), tensor)
                    print("root_tensor", hvd.rank(), root_tensor)
                    print("comparison", hvd.rank(), tensor == root_tensor)
                assert not same(tensor.asnumpy(), root_tensor.asnumpy()), \
                    'hvd.broadcast modifies source tensor'
            if not same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()):
                print("broadcast", count, dtype, dim)
                print("broadcast_tensor", hvd.rank(), broadcast_tensor)
                print("root_tensor", hvd.rank(), root_tensor)
                print("comparison", hvd.rank(),
                      broadcast_tensor == root_tensor)
            assert same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()), \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        dtypes = ['int32',   'int64',
                  'float32', 'float64'] 
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims,
                                                       root_ranks):
            tensor = mx.nd.ones(shapes[dim], ctx=ctx) * rank
            root_tensor = mx.nd.ones(shapes[dim], ctx=ctx) * root_rank
            tensor = tensor.astype(dtype)
            root_tensor = root_tensor.astype(dtype)

            # Only do broadcasting using broadcast_tensor
            broadcast_tensor = tensor.copy()
            hvd.broadcast_(broadcast_tensor, root_rank=root_rank,
                           name=str(count))
            if rank != root_rank:
                if same(tensor.asnumpy(), root_tensor.asnumpy()):
                    print("broadcast", count, dtype, dim,
                          mx.nd.max(tensor == root_tensor))
                    print("tensor", hvd.rank(), tensor)
                    print("root_tensor", hvd.rank(), root_tensor)
                    print("comparison", hvd.rank(), tensor == root_tensor)
                assert not same(tensor.asnumpy(), root_tensor.asnumpy()), \
                    'hvd.broadcast modifies source tensor'
            if not same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()):
                print("broadcast", count, dtype, dim)
                print("broadcast_tensor", hvd.rank(), broadcast_tensor)
                print("root_tensor", hvd.rank(), root_tensor)
                print("comparison", hvd.rank(),
                      broadcast_tensor == root_tensor)
            assert same(broadcast_tensor.asnumpy(), root_tensor.asnumpy()), \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_grad(self):
        """Test the correctness of the broadcast gradient."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        dtypes = ['int32',   'int64',
                  'float32', 'float64'] 
        dims = [1, 2, 3]
        ctx = self._current_context()
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        root_rank = 1
        tensor_dict = {}
        root_dict = {}
        for dtype, dim, in itertools.product(dtypes, dims):
            tensor_dict[count] = mx.nd.ones(shapes[dim], ctx=ctx) * rank
            root_dict[count] = mx.nd.ones(shapes[dim], ctx=ctx) * root_rank
            tensor_dict[count] = tensor_dict[count].astype(dtype)
            root_dict[count] = root_dict[count].astype(dtype)

            # Only do broadcasting using and on broadcast_tensor
            count += 1

        hvd.broadcast_parameters(tensor_dict, root_rank=root_rank)
        for i in range(count):
            if not same(tensor_dict[i].asnumpy(), root_dict[i].asnumpy()):
                print("broadcast", count, dtype, dim)
                print("broadcast_tensor", hvd.rank(), tensor_dict[i])
                print("root_tensor", hvd.rank(), root_dict[i])
                print("comparison", hvd.rank(), tensor_dict[i] == root_dict[i])
            assert same(tensor_dict[i].asnumpy(), root_dict[i].asnumpy()), \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_error(self):
        """Test that the broadcast returns an error if any dimension besides
           the first is different among the tensors being broadcasted."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        ctx = self._current_context()
        shape = (17, rank+1)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)

        try:
            output = hvd.broadcast(tensor, 0)
            output.wait_to_read()
            assert False, 'hvd.broadcast did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast_type_error(self):
        """Test that the broadcast returns an error if the types being broadcasted
           differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        ctx = self._current_context()
        shape = (17, 3)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        if rank % 2 == 0:
            tensor = tensor.astype('int32')
        else:
            tensor = tensor.astype('float32')

        try:
            output = hvd.broadcast(tensor, 0)
            output.wait_to_read()
            assert False, 'hvd.broadcast did not throw error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast_rank_error(self):
        """Test that the broadcast returns an error if different ranks
           specify different root rank."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        ctx = self._current_context()
        shape = (17, 17, 17)
        tensor = mx.nd.ones(shape=shape, ctx=ctx)
        try:
            output = hvd.broadcast(tensor, root_rank=rank)
            output.wait_to_read()
            assert False, 'hvd.broadcast did not throw rank error'
        except (MXNetError, RuntimeError):
            pass

    def test_horovod_broadcast_deferred_init_parameters(self):
        """Test that the deferred initialized parameters are broadcasted."""
        hvd.init()
        root_rank = 0
        rank = hvd.rank()

        # This test does not apply if there is only one worker.
        if hvd.size() == 1:
            return

        mx.random.seed(rank)
        layer = mx.gluon.nn.Conv2D(10, 2)
        layer.initialize()
        hvd.broadcast_parameters(layer.collect_params(), root_rank=root_rank)

        x = mx.nd.ones((5, 4, 10, 10))
        layer(x)
        tensors = [p.data() for _, p in sorted(layer.collect_params().items())]
        root_tensors = []
        for tensor in tensors:
            root_tensors.append(hvd.broadcast(tensor, root_rank=root_rank))

        for tensor, root_tensor in zip(tensors, root_tensors):
            assert same(tensor.asnumpy(), root_tensor.asnumpy()), \
                'horovod did not broadcast deferred initialized parameter correctly'


if __name__ == '__main__':
    unittest.main()
