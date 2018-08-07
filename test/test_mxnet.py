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

import itertools
import mxnet as mx
import unittest
import numpy as np

#import horovod.mxnet as hvd
import mxnet as mx

class MXTests(unittest.TestCase):
    """
    Tests for ops in horovod.mxnet.
    """

    def test_horovod_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        kv = mx.kv.create("horovod")
        size = kv.num_workers
        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        dev = mx.gpu(kv.local_rank)
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        kv.init([i for i in range(len(dtypes)*len(dims))], [mx.nd.zeros(shapes[dim]) for dtype, dim in itertools.product(dtypes, dims)])
        for dtype, dim in itertools.product(dtypes, dims):
            # MXNet uses gpu_id as part of the seed, so to get identical seeds
            # we must set a context.
            mx.random.seed(1234, ctx=dev)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim], ctx=dev)
            summed = mx.nd.zeros(shapes[dim], ctx=dev)
            tensor = tensor.astype(dtype)
            summed = summed.astype(dtype)
            kv.pushpull(count, tensor, summed)
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
                print("allreduce", count, dtype, dim, max_difference, threshold)
                print("tensor", kv.rank, tensor)
                print("summed", kv.rank, summed)
                print("multiplied", kv.rank, multiplied)
            assert max_difference <= threshold, 'hvd.allreduce produces incorrect results'

    def test_horovod_allreduce_average(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        kv = mx.kv.create("horovod")
        size = kv.num_workers
        dtypes = ['int32',   'int64',
                  'float32', 'float64']
        dims = [1, 2, 3]
        dev = mx.gpu(kv.local_rank)
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        kv.init([i for i in range(len(dtypes)*len(dims))], [mx.nd.zeros(shapes[dim]) for dtype, dim in itertools.product(dtypes, dims)])
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=dev)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim], ctx=dev)
            tensor = tensor.astype(dtype)
            averaged = mx.nd.zeros(shapes[dim], ctx=dev)
            averaged = averaged.astype(dtype)
            kv.pushpull(count, tensor, averaged, average=True)
            max_difference = mx.nd.max(mx.nd.subtract(averaged, tensor))
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
                print("average", count, dtype, dim, max_difference, threshold)
                print("tensor", kv.rank, tensor)
                print("averaged", kv.rank, averaged)
            assert max_difference <= threshold, 'hvd.allreduce produces incorrect results for average'
    
    def test_horovod_allreduce_inplace(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        kv = mx.kv.create("horovod")
        size = kv.num_workers
        dtypes = ['int32',   'int64',
                  'float32', 'float64'] 
        dims = [1, 2, 3]
        dev = mx.gpu(kv.local_rank)
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        kv.init([i for i in range(len(dtypes)*len(dims))], [mx.nd.zeros(shapes[dim]) for dtype, dim in itertools.product(dtypes, dims)])
        for dtype, dim in itertools.product(dtypes, dims):
            mx.random.seed(1234, ctx=dev)
            tensor = mx.nd.random.uniform(-100, 100, shape=shapes[dim], ctx=dev)
            tensor = tensor.astype(dtype)
            multiplied = tensor * size
            kv.pushpull(count, tensor, tensor)
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
                print("tensor", kv.rank, tensor)
                print("multiplied", kv.rank, multiplied)
            assert max_difference <= threshold, 'hvd.allreduce produces incorrect results for self'

    # Requires hvd.poll and hvd.synchronize
    #def test_horovod_allreduce_async_fused(self):

    # Above tests are already multi gpu
    #def test_horovod_allreduce_multi_gpu(self):

    # TODO(carlyang) This test currently hangs
    @unittest.skip("")
    def test_horovod_allreduce_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different rank or dimension."""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers
        dev = mx.gpu(kv.local_rank)

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # Same rank, different dimension
        mx.random.seed(1234, ctx=dev)
        dims = (17 + rank, 17 + rank, 17 + rank)
        kv.init("1", mx.nd.zeros(dims))
        tensor = mx.nd.random.uniform(-100, 100, shape=dims, ctx=dev)
        try:
            kv.pushpull("1", tensor, tensor)
            assert False, 'hvd.allreduce did not throw error'
        except Exception as e:
            print(e)

    def test_horovod_allreduce_rank_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different rank or dimension."""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers
        dev = mx.gpu(kv.local_rank)

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # Same number of elements, different rank
        mx.random.seed(1234, ctx=dev)
        if rank == 0:
            dims = (17, 23 * 57)
        else:
            dims = (17, 23, 57)
        tensor = mx.nd.random.uniform(-100, 100, shape=dims, ctx=dev)
        kv.init("1", mx.nd.zeros(dims))
        try:
            kv.pushpull("1", tensor, tensor)
            assert False, 'hvd.allreduce did not throw rank error'
        except Exception as e:
            print(e)

    def test_horovod_allreduce_type_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different type."""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers
        dev = mx.gpu(kv.local_rank)

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # Same rank, different dimension
        dims = (17, 17, 17)
        kv.init("1", mx.nd.zeros(dims, ctx=dev))
        tensor = mx.nd.zeros(shape=dims, ctx=dev)
        if rank % 2 == 0:
            tensor.astype('int32')

        try:
            kv.pushpull("1", tensor, tensor)
            assert False, 'hvd.allreduce did not throw type error'
        except Exception as e:
            print(e)

    def test_horovod_allreduce_cpu_gpu_error(self):
        """Test that the allreduce raises an error if different ranks try to
        perform reduction on CPU and GPU."""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers
        dev = mx.gpu(kv.local_rank)

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # Same rank, different dimension
        dims = (17, 17, 17)
        if rank % 2 == 0:
            dev = mx.gpu(kv.rank)
        else:
            dev = mx.cpu(kv.rank)

        try:
            kv.pushpull("1", tensor, tensor)
            assert False, 'hvd.allreduce did not throw cpu-gpu error'
        except Exception as e:
            print(e)

    # Currently, MXNet doesn't track gradient of hvd.allreduce op
    #def test_horovod_allreduce_grad(self):

    # Currently, MXNet doesn't track gradient of hvd.allreduce op
    #def test_horovod_allreduce_grad_average(self):

    @unittest.skip("")
    def test_horovod_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.FloatTensor,
                       torch.cuda.DoubleTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            tensor = tensor.type(dtype)
            gathered = hvd.allgather(tensor)

            assert list(gathered.shape) == [17 * size] + [17] * (dim - 1)

            for i in range(size):
                rank_tensor = gathered[i * 17:(i + 1) * 17]
                assert list(rank_tensor.shape) == [17] * dim, \
                    'hvd.allgather produces incorrect gathered shape'
                assert rank_tensor.data.min() == i, 'hvd.allgather produces incorrect gathered tensor'
                assert rank_tensor.data.max() == i, 'hvd.allgather produces incorrect gathered tensor'

    @unittest.skip("")
    def test_horovod_allgather_variable_size(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.FloatTensor,
                       torch.cuda.DoubleTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = torch.FloatTensor(
                *([tensor_sizes[rank]] + [17] * (dim - 1))).fill_(1).mul_(rank)
            tensor = tensor.type(dtype)
            gathered = hvd.allgather(tensor)

            expected_size = sum(tensor_sizes)
            assert list(gathered.shape) == [expected_size] + [17] * (dim - 1)

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = gathered[sum(
                    tensor_sizes[:i]):sum(tensor_sizes[:i + 1])]
                assert list(rank_tensor.shape) == rank_size
                assert rank_tensor.data.min() == i
                assert rank_tensor.data.max() == i

    @unittest.skip("")
    def test_horovod_allgather_error(self):
        """Test that the allgather returns an error if any dimension besides
        the first is different among the tensors being gathered."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        tensor_size = [17] * 3
        tensor_size[1] = 10 * (rank + 1)
        tensor = torch.FloatTensor(*tensor_size).fill_(1).mul_(rank)

        try:
            hvd.allgather(tensor)
            assert False, 'hvd.allgather did not throw error'
        except torch.FatalError:
            pass

    @unittest.skip("")
    def test_horovod_allgather_type_error(self):
        """Test that the allgather returns an error if the types being gathered
        differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        tensor_size = [17] * 3
        if rank % 2 == 0:
            tensor = torch.IntTensor(*tensor_size)
        else:
            tensor = torch.FloatTensor(*tensor_size)

        try:
            hvd.allgather(tensor)
            assert False, 'hvd.allgather did not throw error'
        except torch.FatalError:
            pass

    # MXNet doesn't track gradient of hvd.allgather
    #def test_horovod_allgather_grad(self):
        """Test the correctness of the allgather gradient."""

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        dtypes = ['int32',   'int64',
                  'float32', 'float64'] 
        dims = [1, 2, 3]
        dev = mx.gpu(kv.local_rank)
        count = 0
        shapes = [(), (17), (17, 17), (17, 17, 17)]
        kv.init([i for i in range(len(dtypes)*len(dims))], [mx.nd.zeros(shapes[dim]) for dtype, dim in itertools.product(dtypes, dims)])
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = mx.nd.ones(shapes[dim], ctx=dev) * rank
            root_tensor = mx.nd.ones(shapes[dim], ctx=dev) * root_rank
            tensor = tensor.astype(dtype)
            root_tensor = root_tensor.astype(dtype)

            # Only do broadcasting using and on broadcast_tensor
            broadcast_tensor = tensor.copy()
            kv.broadcast(count, broadcast_tensor, root_rank)
            if rank != root_rank:
                if mx.nd.max(tensor == root_tensor) != 0:
                    print("broadcast", count, dtype, dim, mx.nd.max(tensor == root_tensor))
                    print("tensor", kv.rank, tensor)
                    print("root_tensor", kv.rank, root_tensor)
                    print("comparison", kv.rank, tensor == root_tensor)
                assert mx.nd.max(tensor == root_tensor) == 0, \
                    'hvd.broadcast modifies source tensor'
            if mx.nd.min(broadcast_tensor == root_tensor) != 1:
                print("broadcast", count, dtype, dim)
                print("broadcast_tensor", kv.rank, broadcast_tensor)
                print("root_tensor", kv.rank, root_tensor)
                print("comparison", kv.rank, broadcast_tensor == root_tensor)
            assert mx.nd.min(broadcast_tensor == root_tensor) == 1, \
                'hvd.broadcast produces incorrect broadcasted tensor'

    # Does not apply to MXNet broadcast, which only does inplace
    #def test_horovod_broadcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""

    def test_horovod_broadcast_error(self):
        """Test that the broadcast returns an error if any dimension besides
        the first is different among the tensors being broadcasted."""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers
        dev = mx.gpu(kv.local_rank)

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        tensor_size = (17, 10*(rank+1), 17)
        tensor = mx.nd.ones(tensor_size, ctx=dev) * rank
        kv.init("1", mx.nd.zeros(tensor_size))
        try:
            kv.broadcast("1", tensor, 0)
            assert False, 'hvd.broadcast did not throw error'
        except Exception as e:
            print(e)

    def test_horovod_broadcast_type_error(self):
        """Test that the broadcast returns an error if the types being broadcasted
        differ among the processes"""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers
        dev = mx.gpu(kv.local_rank)

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        tensor_size = (17, 17, 17)
        tensor = mx.nd.ones(tensor_size, ctx=dev)
        kv.init("1", mx.nd.zeros(tensor_size))
        if rank % 2 == 0:
            tensor = tensor.astype('int32')

        try:
            kv.broadcast("1", tensor, 0)
            assert False, 'hvd.broadcast did not throw type error'
        except Exception as e:
            print(e)

    def test_horovod_broadcast_rank_error(self):
        """Test that the broadcast returns an error if different ranks
        specify different root rank."""
        kv = mx.kv.create("horovod")
        rank = kv.rank
        size = kv.num_workers
        dev = mx.gpu(kv.local_rank)

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        tensor_size = (17,17,17)
        tensor = mx.nd.ones(tensor_size, ctx=dev)
        kv.init("1", mx.nd.zeros(tensor_size))
        try:
            kv.broadcast("1", tensor, rank)
            assert False, 'hvd.broadcast did not throw rank error'
        except Exception as e:
            print(e)

    # Does not apply to MXNet, which does not track gradient here
    #def test_horovod_broadcast_grad(self):
        """Test the correctness of the broadcast gradient."""

if __name__ == '__main__':
    unittest.main()
