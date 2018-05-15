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
import torch

import horovod.torch as hvd


def test_horovod_allreduce():
    """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
    hvd.init()
    size = hvd.size()
    dtypes = [torch.IntTensor, torch.LongTensor,
              torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
        tensor = tensor.type(dtype)
        summed = hvd.allreduce(tensor, average=False)
        multiplied = tensor * size
        max_difference = summed.sub(multiplied).max()

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                  torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        assert max_difference <= threshold, 'hvd.allreduce produces incorrect results'


def test_horovod_allreduce_average():
    """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
    hvd.init()
    size = hvd.size()
    dtypes = [torch.IntTensor, torch.LongTensor,
              torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
        tensor = tensor.type(dtype)
        averaged = hvd.allreduce(tensor, average=True)
        max_difference = averaged.sub(tensor).max()

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                  torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        assert max_difference <= threshold, 'hvd.allreduce produces incorrect results'


def test_horovod_allreduce_inplace():
    """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
    hvd.init()
    size = hvd.size()
    dtypes = [torch.IntTensor, torch.LongTensor,
              torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
        tensor = tensor.type(dtype)
        multiplied = tensor * size
        hvd.allreduce_(tensor, average=False)
        max_difference = tensor.sub(multiplied).max()

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                  torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        assert max_difference <= threshold, 'hvd.allreduce produces incorrect results'


def test_horovod_allreduce_async_fused():
    """Test that the allreduce correctly sums 1D, 2D, 3D tensors
    with Tensor Fusion."""
    hvd.init()
    size = hvd.size()
    dtypes = [torch.IntTensor, torch.LongTensor,
              torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                   torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
    dims = [1, 2, 3]
    tests = []
    is_hvd_poll_false_once = False
    for dtype, dim in itertools.product(dtypes, dims):
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
        tensor = tensor.type(dtype)
        handle = hvd.allreduce_async(tensor, average=False)
        if not hvd.poll(handle):
            is_hvd_poll_false_once = True
        multiplied = tensor * size
        tests.append((dtype, multiplied, handle))

    # Make sure it's an asynchronous operation.
    assert is_hvd_poll_false_once, 'hvd.poll() always returns True, not an async op?'

    for dtype, multiplied, handle in tests:
        summed = hvd.synchronize(handle)
        max_difference = summed.sub(multiplied).max()

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.IntTensor, torch.LongTensor,
                                  torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        assert max_difference <= threshold, 'hvd.allreduce produces incorrect results'


def test_horovod_allreduce_multi_gpu():
    """Test that the allreduce works on multiple GPUs."""
    # Only do this test if there are GPUs available.
    if not torch.cuda.is_available():
        return

    hvd.init()
    local_rank = hvd.local_rank()
    size = hvd.size()

    iter = 0
    dtypes = [torch.cuda.IntTensor, torch.cuda.LongTensor,
              torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
    dims = [1, 2, 3]
    for dtype, dim in itertools.product(dtypes, dims):
        iter += 1
        torch.manual_seed(1234)
        tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
        device = local_rank * 2 + (iter + local_rank) % 2
        tensor = tensor.cuda(device).type(dtype)
        multiplied = tensor * size
        hvd.allreduce_(tensor, average=False)
        max_difference = tensor.sub(multiplied).max()

        # Threshold for floating point equality depends on number of
        # ranks, since we're comparing against precise multiplication.
        if size <= 3 or dtype in [torch.cuda.IntTensor, torch.cuda.LongTensor]:
            threshold = 0
        elif size < 10:
            threshold = 1e-4
        elif size < 15:
            threshold = 5e-4
        else:
            break

        assert max_difference <= threshold, 'hvd.allreduce produces incorrect results'


def test_horovod_allreduce_error():
    """Test that the allreduce raises an error if different ranks try to
    send tensors of different rank or dimension."""
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    # Same rank, different dimension
    torch.manual_seed(1234)
    dims = [17 + rank] * 3
    tensor = torch.FloatTensor(*dims).random_(-100, 100)
    try:
        hvd.allreduce(tensor)
        assert False, 'hvd.allreduce did not throw error'
    except torch.FatalError:
        pass

    # Same number of elements, different rank
    torch.manual_seed(1234)
    if rank == 0:
        dims = [17, 23 * 57]
    else:
        dims = [17, 23, 57]
    tensor = torch.FloatTensor(*dims).random_(-100, 100)
    try:
        hvd.allreduce(tensor)
        assert False, 'hvd.allreduce did not throw error'
    except torch.FatalError:
        pass


def test_horovod_allreduce_type_error():
    """Test that the allreduce raises an error if different ranks try to
    send tensors of different type."""
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    # Same rank, different dimension
    dims = [17] * 3
    if rank % 2 == 0:
        tensor = torch.IntTensor(*dims)
    else:
        tensor = torch.FloatTensor(*dims)

    try:
        hvd.allreduce(tensor)
        assert False, 'hvd.allreduce did not throw error'
    except torch.FatalError:
        pass


def test_horovod_allreduce_cpu_gpu_error():
    """Test that the allreduce raises an error if different ranks try to
    perform reduction on CPU and GPU."""
    # Only do this test if there are GPUs available.
    if not torch.cuda.is_available():
        return

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    # Same rank, different dimension
    dims = [17] * 3
    if rank % 2 == 0:
        tensor = torch.cuda.FloatTensor(*dims)
    else:
        tensor = torch.FloatTensor(*dims)

    try:
        hvd.allreduce(tensor)
        assert False, 'hvd.allreduce did not throw error'
    except torch.FatalError:
        pass


def test_horovod_allgather():
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
            assert rank_tensor.min() == i, 'hvd.allgather produces incorrect gathered tensor'
            assert rank_tensor.max() == i, 'hvd.allgather produces incorrect gathered tensor'


def test_horovod_allgather_variable_size():
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
            assert rank_tensor.min() == i
            assert rank_tensor.max() == i


def test_horovod_allgather_error():
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


def test_horovod_allgather_type_error():
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


def test_horovod_broadcast():
    """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
              torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                   torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.FloatTensor,
                   torch.cuda.DoubleTensor]
    dims = [1, 2, 3]
    root_ranks = list(range(size))
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
        tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
        root_tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(root_rank)
        tensor = tensor.type(dtype)
        root_tensor = root_tensor.type(dtype)
        broadcasted_tensor = hvd.broadcast(tensor, root_rank)
        if rank != root_rank:
            assert (tensor == root_tensor).max() == 0, \
                'hvd.broadcast modifies source tensor'
        assert (broadcasted_tensor == root_tensor).min() == 1, \
            'hvd.broadcast produces incorrect broadcasted tensor'


def test_horovod_broadcast_inplace():
    """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
              torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
    if torch.cuda.is_available():
        dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                   torch.cuda.IntTensor, torch.cuda.LongTensor, torch.cuda.FloatTensor,
                   torch.cuda.DoubleTensor]
    dims = [1, 2, 3]
    root_ranks = list(range(size))
    for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
        tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
        root_tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(root_rank)
        tensor = tensor.type(dtype)
        root_tensor = root_tensor.type(dtype)
        broadcasted_tensor = hvd.broadcast_(tensor, root_rank)
        assert (tensor == broadcasted_tensor).min() == 1, \
            'hvd.broadcast does not modify source tensor'
        assert (broadcasted_tensor == root_tensor).min() == 1, \
            'hvd.broadcast produces incorrect broadcasted tensor'


def test_horovod_broadcast_error():
    """Test that the broadcast returns an error if any dimension besides
    the first is different among the tensors being broadcasted."""
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
        hvd.broadcast(tensor, 0)
        assert False, 'hvd.broadcast did not throw error'
    except torch.FatalError:
        pass


def test_horovod_broadcast_type_error():
    """Test that the broadcast returns an error if the types being broadcasted
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
        hvd.broadcast(tensor, 0)
        assert False, 'hvd.broadcast did not throw error'
    except torch.FatalError:
        pass


def test_horovod_broadcast_rank_error():
    """Test that the broadcast returns an error if different ranks
    specify different root rank."""
    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    # This test does not apply if there is only one worker.
    if size == 1:
        return

    tensor = torch.FloatTensor(*([17] * 3)).fill_(1)

    try:
        hvd.broadcast(tensor, rank)
        assert False, 'hvd.broadcast did not throw error'
    except torch.FatalError:
        pass
