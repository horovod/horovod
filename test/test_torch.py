# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2019 Intel Corporation
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

from distutils.version import LooseVersion

import inspect
import itertools
import os
import pytest
import sys
import unittest
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import horovod.torch as hvd

from common import mpi_env_rank_and_size, temppath

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

_v2_api = LooseVersion(torch.__version__) >= LooseVersion('1.0.0')
_fp16_supported = _v2_api

ccl_supported_types = set([torch.CharTensor, torch.IntTensor,
                           torch.LongTensor, torch.FloatTensor, 
                           torch.DoubleTensor])


class TorchTests(unittest.TestCase):
    """
    Tests for ops in horovod.torch.
    """

    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def convert_cpu_fp16_to_fp32(self, *values):
        # PyTorch doesn't support any CPU ops on FP16 tensors.
        # In case we need to do ops, we will convert tensor to FP32 here.
        result = []
        for value in values:
            if value.dtype in [torch.float16, torch.HalfTensor]:
                result.append(value.float())
            else:
                result.append(value)
        return result

    def cast_and_place(self, tensor, dtype):
        if dtype.is_cuda:
            return tensor.cuda(hvd.local_rank()).type(dtype)
        return tensor.type(dtype)

    def filter_supported_types(self, types):
        if 'CCL_ROOT' in os.environ:
           types = [t for t in types if t in ccl_supported_types]
        return types

    def test_horovod_reinit(self):
        """Test that Horovod can init -> shutdown -> init successfully."""
        mpi_rank, _ = mpi_env_rank_and_size()
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))

        is_mpi = gloo_rank == -1
        if is_mpi:
            # Only applies for Gloo
            self.skipTest("Gloo is not available")

        hvd.init()
        rank, size = hvd.rank(), hvd.size()

        hvd.shutdown()
        hvd.init()
        rank2, size2 = hvd.rank(), hvd.size()

        assert rank == rank2
        assert size == size2

    def test_horovod_rank(self):
        """Test that the rank returned by hvd.rank() is correct."""
        mpi_rank, _ = mpi_env_rank_and_size()
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))

        # The mpi rank does not match gloo rank, we need to figure which one
        # we are using to run the test.
        is_mpi = gloo_rank == -1
        hvd.init()
        rank = hvd.rank()

        if is_mpi:
            assert mpi_rank == rank
        else:
            assert gloo_rank == rank

    def test_horovod_size(self):
        """Test that the size returned by hvd.size() is correct."""
        _, mpi_size = mpi_env_rank_and_size()
        gloo_size = int(os.getenv('HOROVOD_SIZE', -1))

        # The mpi size does not match gloo size, we need to figure which one
        # we are using to run the test.
        is_mpi = gloo_size == -1
        hvd.init()
        size = hvd.size()
        if is_mpi:
            assert mpi_size == size
        else:
            assert gloo_size == size

    def test_horovod_allreduce(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                     torch.FloatTensor, torch.DoubleTensor])
        if _fp16_supported:
            dtypes += self.filter_supported_types([torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            summed = hvd.allreduce(tensor, average=False)
            tensor, summed = self.convert_cpu_fp16_to_fp32(tensor, summed)
            multiplied = tensor * size
            max_difference = summed.data.sub(multiplied).max()

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

    def test_horovod_allreduce_average(self):
        """Test that the allreduce correctly averages 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                     torch.FloatTensor, torch.DoubleTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            averaged = hvd.allreduce(tensor, average=True)
            max_difference = averaged.data.sub(tensor).max()

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

    def test_horovod_allreduce_inplace(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                     torch.FloatTensor, torch.DoubleTensor])
        if _fp16_supported:
            dtypes += self.filter_supported_types([torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            multiplied = self.cast_and_place(tensor * size, dtype)
            tensor = self.cast_and_place(tensor, dtype)
            hvd.allreduce_(tensor, average=False)
            tensor, multiplied = self.convert_cpu_fp16_to_fp32(tensor, multiplied)
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

    def test_horovod_allreduce_async_fused(self):
        """Test that the allreduce correctly sums 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                  torch.FloatTensor, torch.DoubleTensor])
        if _fp16_supported:
            dtypes += self.filter_supported_types([torch.HalfTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        tests = []
        is_hvd_poll_false_once = False
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            handle = hvd.allreduce_async(tensor, average=False)
            if not hvd.poll(handle):
                is_hvd_poll_false_once = True
            tensor, = self.convert_cpu_fp16_to_fp32(tensor)
            multiplied = tensor * size
            tests.append((dtype, multiplied, handle))

        # Make sure it's an asynchronous operation.
        assert is_hvd_poll_false_once, 'hvd.poll() always returns True, not an async op?'

        for dtype, multiplied, handle in tests:
            summed = hvd.synchronize(handle)
            summed, = self.convert_cpu_fp16_to_fp32(summed)
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

    def test_horovod_allreduce_multi_gpu(self):
        """Test that the allreduce works on multiple GPUs."""
        # Only do this test if there are GPUs available.
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # Skip the test if there are not enough GPUs.
        if torch.cuda.device_count() < hvd.local_size() * 2:
            self.skipTest("Not enough GPUs available")

        iter = 0
        dtypes = [torch.cuda.IntTensor, torch.cuda.LongTensor,
                  torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.cuda.HalfTensor]

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

    def test_horovod_allreduce_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different rank or dimension."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Same rank, different dimension
        torch.manual_seed(1234)
        dims = [17 + rank] * 3
        tensor = torch.FloatTensor(*dims).random_(-100, 100)
        try:
            hvd.allreduce(tensor)
            assert False, 'hvd.allreduce did not throw error'
        except (torch.FatalError, RuntimeError):
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
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_allreduce_type_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different type."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Same rank, different dimension
        dims = [17] * 3
        if rank % 2 == 0:
            tensor = torch.IntTensor(*dims)
        else:
            tensor = torch.FloatTensor(*dims)

        try:
            hvd.allreduce(tensor)
            assert False, 'hvd.allreduce did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_allreduce_cpu_gpu_error(self):
        """Test that the allreduce raises an error if different ranks try to
        perform reduction on CPU and GPU."""
        # Only do this test if there are GPUs available.
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        if os.environ.get('HOROVOD_MIXED_INSTALL'):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            self.skipTest("Not compiled with HOROVOD_GPU_ALLREDUCE")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Same rank, different dimension
        dims = [17] * 3
        if rank % 2 == 0:
            tensor = torch.cuda.FloatTensor(*dims)
        else:
            tensor = torch.FloatTensor(*dims)

        try:
            hvd.allreduce(tensor)
            assert False, 'hvd.allreduce did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_allreduce_duplicate_name_error(self):
        """Test that the allreduce raises an error if there are
        two concurrent operations with the same name."""
        hvd.init()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dims = [17] * 3
        tensor = torch.FloatTensor(*dims)

        hvd.allreduce_async(tensor, name='duplicate_name')
        try:
            for i in range(10):
                hvd.allreduce_async(tensor, name='duplicate_name')
            assert False, 'hvd.allreduce_async did not throw error'
        except (torch.FatalError, ValueError):
            pass

    def test_horovod_allreduce_grad(self):
        """Test the correctness of the allreduce gradient."""
        hvd.init()
        size = hvd.size()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            summed = hvd.allreduce(tensor, average=False)

            summed.backward(self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones([17] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_grad_average(self):
        """Test the correctness of the allreduce averaged gradient."""
        hvd.init()
        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            torch.manual_seed(1234)
            tensor = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()
            summed = hvd.allreduce(tensor, average=True)

            summed.backward(self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones([17] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.HalfTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            gathered = hvd.allgather(tensor)
            tensor, gathered = self.convert_cpu_fp16_to_fp32(tensor, gathered)

            assert list(gathered.shape) == [17 * size] + [17] * (dim - 1)

            for i in range(size):
                rank_tensor = gathered[i * 17:(i + 1) * 17]
                assert list(rank_tensor.shape) == [17] * dim, \
                    'hvd.allgather produces incorrect gathered shape'
                assert rank_tensor.data.min() == i, 'hvd.allgather produces incorrect gathered tensor'
                assert rank_tensor.data.max() == i, 'hvd.allgather produces incorrect gathered tensor'

    def test_horovod_allgather_variable_size(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.HalfTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = torch.FloatTensor(
                *([tensor_sizes[rank]] + [17] * (dim - 1))).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            gathered = hvd.allgather(tensor)
            tensor, gathered = self.convert_cpu_fp16_to_fp32(tensor, gathered)

            expected_size = sum(tensor_sizes)
            assert list(gathered.shape) == [expected_size] + [17] * (dim - 1)

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = gathered[sum(
                    tensor_sizes[:i]):sum(tensor_sizes[:i + 1])]
                assert list(rank_tensor.shape) == rank_size
                assert rank_tensor.data.min() == i
                assert rank_tensor.data.max() == i

    def test_horovod_allgather_async_fused(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.HalfTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        tests = []
        is_hvd_poll_false_once = False
        for dtype, dim in itertools.product(dtypes, dims):
            rank_shape = [17] * dim
            tensor = torch.FloatTensor(*(rank_shape)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            handle = hvd.allgather_async(tensor)
            if not hvd.poll(handle):
                is_hvd_poll_false_once = True
            tests.append((handle, rank_shape))

        # Make sure it's an asynchronous operation.
        assert is_hvd_poll_false_once, 'hvd.poll() always returns True, not an async op?'

        for handle, rank_shape in tests:
            gathered = hvd.synchronize(handle)
            gathered, = self.convert_cpu_fp16_to_fp32(gathered)

            for i in range(size):
                rank_tensor = gathered[i * 17:(i + 1) * 17]
                assert list(rank_tensor.shape) == rank_shape, \
                    'hvd.allgather produces incorrect gathered shape'
                assert rank_tensor.data.min() == i, 'hvd.allgather produces incorrect gathered tensor'
                assert rank_tensor.data.max() == i, 'hvd.allgather produces incorrect gathered tensor'

    def test_horovod_allgather_error(self):
        """Test that the allgather returns an error if any dimension besides
        the first is different among the tensors being gathered."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor_size = [17] * 3
        tensor_size[1] = 10 * (rank + 1)
        tensor = torch.FloatTensor(*tensor_size).fill_(1).mul_(rank)

        try:
            hvd.allgather(tensor)
            assert False, 'hvd.allgather did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_allgather_type_error(self):
        """Test that the allgather returns an error if the types being gathered
        differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor_size = [17] * 3
        if rank % 2 == 0:
            tensor = torch.IntTensor(*tensor_size)
        else:
            tensor = torch.FloatTensor(*tensor_size)

        try:
            hvd.allgather(tensor)
            assert False, 'hvd.allgather did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_allgather_duplicate_name_error(self):
        """Test that the allgather raises an error if there are
        two concurrent operations with the same name."""
        hvd.init()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dims = [17] * 3
        tensor = torch.FloatTensor(*dims)

        hvd.allgather_async(tensor, name='duplicate_name')
        try:
            for i in range(10):
                hvd.allgather_async(tensor, name='duplicate_name')
            assert False, 'hvd.allgather_async did not throw error'
        except (torch.FatalError, ValueError):
            pass

    def test_horovod_allgather_grad(self):
        """Test the correctness of the allgather gradient."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [3, 2, 7, 4, 6, 8, 10] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = torch.FloatTensor(
                *([tensor_sizes[rank]] + [17] * (dim - 1))).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()

            grad_list = []
            for r, size in enumerate(tensor_sizes):
                grad_list.append(self.cast_and_place(
                    torch.ones([size] + [17] * (dim - 1)), dtype) * r)
            grad_ys = torch.cat(grad_list, dim=0)

            gathered = hvd.allgather(tensor)
            gathered.backward(grad_ys)
            grad_out = tensor.grad.data.cpu().numpy()

            expected = np.ones(
                [tensor_sizes[rank]] + [17] * (dim - 1)
            ) * rank * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.HalfTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            root_tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)
            broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            tensor, root_tensor, broadcasted_tensor = \
                self.convert_cpu_fp16_to_fp32(tensor, root_tensor, broadcasted_tensor)
            if rank != root_rank:
                assert (tensor == root_tensor).max() == 0, \
                    'hvd.broadcast modifies source tensor'
            assert (broadcasted_tensor.data == root_tensor).min() == 1, \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_inplace(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [torch.ByteTensor, torch.CharTensor, torch.ShortTensor,
                  torch.IntTensor, torch.LongTensor, torch.FloatTensor, torch.DoubleTensor]
        if _fp16_supported:
            dtypes += [torch.HalfTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.ByteTensor, torch.cuda.CharTensor, torch.cuda.ShortTensor,
                       torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            root_tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(root_rank)
            tensor = self.cast_and_place(tensor, dtype)
            root_tensor = self.cast_and_place(root_tensor, dtype)
            broadcasted_tensor = hvd.broadcast_(tensor, root_rank)
            tensor, root_tensor, broadcasted_tensor = \
                self.convert_cpu_fp16_to_fp32(tensor, root_tensor, broadcasted_tensor)
            assert (tensor == broadcasted_tensor).min() == 1, \
                'hvd.broadcast does not modify source tensor'
            assert (broadcasted_tensor == root_tensor).min() == 1, \
                'hvd.broadcast produces incorrect broadcasted tensor'

    def test_horovod_broadcast_error(self):
        """Test that the broadcast returns an error if any dimension besides
        the first is different among the tensors being broadcasted."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor_size = [17] * 3
        tensor_size[1] = 10 * (rank + 1)
        tensor = torch.FloatTensor(*tensor_size).fill_(1).mul_(rank)

        try:
            hvd.broadcast(tensor, 0)
            assert False, 'hvd.broadcast did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_broadcast_type_error(self):
        """Test that the broadcast returns an error if the types being broadcasted
        differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor_size = [17] * 3
        if rank % 2 == 0:
            tensor = torch.IntTensor(*tensor_size)
        else:
            tensor = torch.FloatTensor(*tensor_size)

        try:
            hvd.broadcast(tensor, 0)
            assert False, 'hvd.broadcast did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_broadcast_rank_error(self):
        """Test that the broadcast returns an error if different ranks
        specify different root rank."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor = torch.FloatTensor(*([17] * 3)).fill_(1)

        try:
            hvd.broadcast(tensor, rank)
            assert False, 'hvd.broadcast did not throw error'
        except (torch.FatalError, RuntimeError):
            pass

    def test_horovod_broadcast_duplicate_name_error(self):
        """Test that the broadcast raises an error if there are
        two concurrent operations with the same name."""
        hvd.init()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dims = [17] * 3
        tensor = torch.FloatTensor(*dims)

        hvd.broadcast_async(tensor, root_rank=0, name='duplicate_name')
        try:
            for i in range(10):
                hvd.broadcast_async(tensor, root_rank=0, name='duplicate_name')
            assert False, 'hvd.broadcast_async did not throw error'
        except (torch.FatalError, ValueError):
            pass

    def test_horovod_broadcast_grad(self):
        """Test the correctness of the broadcast gradient."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Only Tensors of floating point dtype can require gradients
        dtypes = [torch.FloatTensor, torch.DoubleTensor]
        if torch.cuda.is_available():
            dtypes += [torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = torch.FloatTensor(*([17] * dim)).fill_(1).mul_(rank)
            tensor = self.cast_and_place(tensor, dtype)
            tensor.requires_grad_()

            broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            broadcasted_tensor.backward(self.cast_and_place(torch.ones([17] * dim), dtype))
            grad_out = tensor.grad.data.cpu().numpy()

            c = size if rank == root_rank else 0
            expected = np.ones([17] * dim) * c
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_broadcast_state(self):
        hvd.init()

        N, D_in, H, D_out = 64, 100, 10, 10
        x = torch.randn(N, D_in).requires_grad_()
        y = torch.randn(N, D_out).requires_grad_()

        def new_optimizer(cls, opt_params, model):
            p = {
                k: v for k, v in opt_params.items()
                if k in inspect.getargspec(cls.__init__).args
            }
            return cls(model.parameters(), **p)

        def create_model(opt_class, opt_params):
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_out),
            )

            optimizer = new_optimizer(opt_class, opt_params, model)
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters())

            return model, optimizer

        def get_model_param_values(model):
            params = sorted(model.state_dict().items())
            return [(k, v.clone()) for k, v in params]

        def get_optimizer_param_values(optimizer):
            results = []
            state_dict = optimizer.state_dict()
            for group in state_dict['param_groups']:
                for param_id in group['params']:
                    if param_id not in state_dict['state']:
                        continue
                    params = sorted(state_dict['state'][param_id].items())
                    for k, v in params:
                        results.append(
                            (k, v.clone() if torch.is_tensor(v) else v))
            return results

        # L-BFGS is currently unsupported, as are sparse tensors, which are
        # required by SparseAdam optimizer
        optimizers = [
            (subclass.__name__, subclass)
            for subclass in torch.optim.Optimizer.__subclasses__()
            if subclass.__module__.startswith('torch.optim') and
               subclass != torch.optim.LBFGS and
               subclass != torch.optim.SparseAdam
        ]
        optimizers.sort()

        opt_params_list = [
            dict(lr=0.2, momentum=0.9, weight_decay=0.1, centered=True),
            dict(lr=0.2)
        ]

        for (opt_name, opt_class), opt_params in itertools.product(optimizers, opt_params_list):
            model, optimizer = create_model(opt_class, opt_params)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_param_values = get_model_param_values(model)
            for name, model_param_value in model_param_values:
                hvd.broadcast_(model_param_value, root_rank=0)

            opt_param_values_updated = []
            opt_param_values = get_optimizer_param_values(optimizer)
            for name, opt_param_value in opt_param_values:
                is_tensor = torch.is_tensor(opt_param_value)
                if not is_tensor:
                    t = type(opt_param_value)
                    opt_param_value = torch.Tensor([opt_param_value])
                hvd.broadcast_(opt_param_value, root_rank=0)
                if not is_tensor:
                    opt_param_value = t(opt_param_value.cpu().numpy()[0])
                opt_param_values_updated.append((name, opt_param_value))
            opt_param_values = opt_param_values_updated

            with temppath() as fname:
                if hvd.rank() == 0:
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state, fname)

                model, optimizer = create_model(opt_class, opt_params)
                if hvd.rank() == 0:
                    checkpoint = torch.load(fname)
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])

            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            model_param_value_after = get_model_param_values(model)
            for before, after in zip(model_param_values,
                                     model_param_value_after):
                name, model_param_value = before
                name_after, model_param_value_after = after
                self.assertEqual(name, name_after)
                self.assertEqual(type(model_param_value),
                                 type(model_param_value_after))
                self.assertTrue(
                    (model_param_value == model_param_value_after).all())

            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

            expected_tensors = 4
            if 'momentum' not in opt_params and opt_class == torch.optim.SGD:
                # SGD only maintains state when momentum is specified, otherwise
                # it does not populate the state dict, so it will contain no tensors.
                expected_tensors = 0
            self.assertEqual(len(optimizer.state_dict()['state'].values()), expected_tensors)

            opt_param_values_after = get_optimizer_param_values(optimizer)
            for before, after in zip(opt_param_values, opt_param_values_after):
                name, opt_param_value = before
                name_after, opt_param_value_after = after
                self.assertEqual(name, name_after)
                self.assertEqual(type(opt_param_value),
                                 type(opt_param_value_after))
                if torch.is_tensor(opt_param_value):
                    self.assertTrue(
                        (opt_param_value == opt_param_value_after).all())
                else:
                    self.assertEqual(opt_param_value, opt_param_value_after)

    # TODO: investigate why this hangs on K80s
    @unittest.skip
    def test_broadcast_state_gpu(self):
        # Only do this test if there are GPUs available.
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")
        # Set default tensor type, ensuring optimizer tensor-wrapping is robust
        # to this setting.
        try:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.test_broadcast_state()
        finally:
            torch.set_default_tensor_type(torch.FloatTensor)

    def test_broadcast_state_options(self):
        hvd.init()

        N, D_in, H, D_out = 64, 100, 10, 10
        x = torch.randn(N, D_in).requires_grad_()
        y = torch.randn(N, D_out).requires_grad_()

        params_0 = dict(lr=0.1, momentum=0.8, weight_decay=0.2, nesterov=True,
                        betas=(0.9, 0.999), etas=(0.8, 2.4), step_sizes=(1e-5, 100))
        params_1 = dict(lr=0.2, momentum=0.9, weight_decay=0.1, nesterov=False,
                        betas=(0.8, 0.9), etas=(0.25, 1.75), step_sizes=(1e-7, 5))

        def create_model(opt_class):
            model = torch.nn.Sequential(
                torch.nn.Linear(D_in, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_out),
            )

            params = params_0 if hvd.rank() == 0 else params_1
            p = {
                k: v for k, v in params.items()
                if k in inspect.getargspec(opt_class.__init__).args
            }
            opt = opt_class(model.parameters(), **p)
            opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())

            return model, opt

        # Include subclass name so we can sort them lexicographically, otherwise different
        # ranks will have different optimizer orderings
        optimizers = [
            (subclass.__name__, subclass)
            for subclass in torch.optim.Optimizer.__subclasses__()
            if subclass.__module__.startswith('torch.optim') and
               subclass != torch.optim.LBFGS and
               subclass != torch.optim.SparseAdam
        ]
        optimizers.sort()

        for _, opt_class in optimizers:
            model, optimizer = create_model(opt_class)
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
            p0 = {
                k: v for k, v in params_0.items()
                if k in inspect.getargspec(opt_class.__init__).args
            }
            for k, p in p0.items():
                p_actual = optimizer.param_groups[0][k]
                if not isinstance(p, Iterable):
                    p_actual = [p_actual]
                    p = [p]
                for i in range(len(p)):
                    self.assertEqual(type(p_actual[i]), type(p[i]))
                    self.assertAlmostEqual(p_actual[i], p[i], delta=1e-5)

            # Ensure that the parameter option types are compatible with ops
            y_pred = model(x)
            loss = F.mse_loss(y_pred, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @pytest.mark.skipif(LooseVersion(torch.__version__) < LooseVersion('0.4.1'),
                        reason='Cannot optimize parameters that do not require gradients before PyTorch 0.4.1')
    def test_broadcast_state_no_grad(self):
        class ModelNoGrad(nn.Module):
            def __init__(self, a, b):
                super(ModelNoGrad, self).__init__()
                self.a = nn.Parameter(a.int(), requires_grad=False)
                self.b = nn.Parameter(b)

            def forward(self, x):
                return torch.index_select(self.b, 0, self.a.long()) * x

        hvd.init()

        a = torch.Tensor([1, 3])
        b = torch.rand(4)

        model = ModelNoGrad(a, b)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-6, momentum=0.9, nesterov=True)
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        grad = optimizer.param_groups[0]['params'][1].grad
        bgrad = hvd.broadcast(grad, root_rank=0)

        assert optimizer.param_groups[0]['params'][0].grad is None
        assert torch.all(torch.eq(grad, bgrad)).item()

    def test_broadcast_object(self):
        hvd.init()

        expected_obj = {
            'hello': 123,
            0: [1, 2]
        }
        obj = expected_obj if hvd.rank() == 0 else {}

        obj = hvd.broadcast_object(obj, root_rank=0)
        self.assertDictEqual(obj, expected_obj)

    def test_compression_fp16(self):
        valid_dtypes = [torch.float32, torch.float64]
        invalid_dtypes = [torch.uint8, torch.int8, torch.int16,
                          torch.int32, torch.int64]

        tensor_size = [5] * 3
        compression = hvd.Compression.fp16

        for dtype in valid_dtypes:
            tensor = torch.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, torch.float16)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - tensor_decompressed.data.numpy())
            self.assertLess(err, 0.00000001)

        for dtype in invalid_dtypes:
            tensor = torch.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, dtype)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            if dtype != torch.int8:  # Cannot cast to NumPy with a CharTensor
                expected = np.ones(tensor_size)
                err = np.linalg.norm(expected - tensor_decompressed.data.numpy())
                self.assertLess(err, 0.00000001)

    def test_force_allreduce(self):
        """Test that allreduce is forced on all gradients during opt.step()."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        N, D_in, H, D_out = 64, 100, 10, 10
        x = torch.randn(N, D_in).requires_grad_()
        y = torch.randn(N, D_out).requires_grad_()

        def new_optimizer(cls, opt_params, model):
            p = {
                k: v for k, v in opt_params.items()
                if k in inspect.getargspec(cls.__init__).args
            }
            return cls(model.parameters(), **p)

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = torch.nn.Linear(D_in, H)
                self.fc2 = torch.nn.Linear(H, D_out)
                self.fc3 = torch.nn.Linear(D_out, D_out)

            def forward(self, x_):
                x_ = F.relu(self.fc1(x_))
                x1_ = self.fc2(x_)
                x2_ = self.fc3(F.relu(x1_))
                return x1_, x2_

        def create_model(opt_class, opt_params):
            model = Net()
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            opt = new_optimizer(opt_class, opt_params, model)
            opt = hvd.DistributedOptimizer(
                opt, named_parameters=model.named_parameters())
            return model, opt

        # L-BFGS is currently unsupported, as are sparse tensors, which are
        # required by SparseAdam optimizer
        optimizers = [
            (subclass.__name__, subclass)
            for subclass in torch.optim.Optimizer.__subclasses__()
            if subclass.__module__.startswith('torch.optim') and
               subclass != torch.optim.LBFGS and
               subclass != torch.optim.SparseAdam
        ]
        optimizers.sort()

        opt_params_list = [
            dict(lr=0.2, momentum=0.9, weight_decay=0.1, centered=True),
            dict(lr=0.2)
        ]

        for (opt_name, opt_class), opt_params in itertools.product(optimizers, opt_params_list):
            model, optimizer = create_model(opt_class, opt_params)
            y_pred1, y_pred2 = model(x)
            if rank == 0:
                loss = F.mse_loss(y_pred1, y, size_average=False)
            else:
                loss = F.mse_loss(y_pred2, y, size_average=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test_model_parallelism(self):
        """Test that tensors on different GPUs are supported."""
        # Only do this test if there are GPUs available.
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Skip the test if there are not enough GPUs.
        if torch.cuda.device_count() < hvd.local_size() * 2:
            self.skipTest("Not enough GPUs available")

        first_device = local_rank * 2
        second_device = local_rank * 2 + 1

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                # Place parts of model on different GPUs.
                self.conv1 = torch.nn.Conv2d(1, 100, 1).cuda(first_device)
                self.conv2 = torch.nn.Conv2d(100, 1, 1).cuda(second_device)

            def forward(self, x):
                x = x.cuda(first_device)
                x = self.conv1(x)
                x = x.cuda(second_device)
                x = self.conv2(x)
                return x

        model = Net()
        inp = torch.rand([1, 1, 1000, 1000])

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())

        loss = model(inp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_delta_optimizer(self):
        """Test that delta optimizer."""
        hvd.init()
        # TODO support non-MPI Adasum operation
        # Only do this test if there are GPUs available.
        if not hvd.mpi_enabled() or not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 100, 1).cuda(local_rank)
                self.conv2 = torch.nn.Conv2d(100, 1, 1).cuda(local_rank)

            def forward(self, x):
                x = x.cuda(local_rank)
                x = self.conv1(x)
                x = x.cuda(local_rank)
                x = self.conv2(x)
                return x

        model = Net()
        inp = torch.rand([1, 1, 1000, 1000])

        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters(), op=hvd.Adasum)
        loss = model(inp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_duplicate_names(self):
        """Test that passing duplicate names to optimizer will fail."""
        net1 = torch.nn.Conv2d(1, 1, 1)
        net2 = torch.nn.Conv2d(1, 1, 1)

        parameters = itertools.chain(net1.parameters(), net2.parameters())
        opt = torch.optim.SGD(parameters, lr=0.1)

        # This will have duplicate names, since both net1 and net2 have 'weight' and 'bias'
        named_parameters = itertools.chain(net1.named_parameters(), net2.named_parameters())
        try:
            hvd.DistributedOptimizer(opt, named_parameters=named_parameters)
            assert False, 'hvd.DistributedOptimizer did not throw error'
        except ValueError:
            pass

    def test_dynamic_requires_grad(self):
        """Test that makes sure that gradients can be turned off/on dynamically."""
        hvd.init()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        gen = torch.nn.Conv2d(1, 10, 1)
        disc = torch.nn.Conv2d(10, 1, 1)
        inp = torch.rand([1, 1, 100, 100])

        gen_opt = torch.optim.SGD(gen.parameters(), lr=0.1)
        gen_opt = hvd.DistributedOptimizer(gen_opt, named_parameters=gen.named_parameters())

        disc_opt = torch.optim.SGD(disc.parameters(), lr=0.1)
        disc_opt = hvd.DistributedOptimizer(disc_opt, named_parameters=disc.named_parameters())

        def train_step(train_generator=False, train_discriminator=False):
            for p in gen.parameters():
                p.requires_grad_(train_generator)
            for p in disc.parameters():
                p.requires_grad_(train_discriminator)

            gen_opt.zero_grad()
            disc_opt.zero_grad()

            loss = disc(gen(inp)).sum()
            loss.backward()

            for p in gen.parameters():
                assert train_generator == p.grad.max().is_nonzero(), \
                    'Gradient for generator is zero but it should be trained or vice versa.'
            for p in disc.parameters():
                assert train_discriminator == p.grad.max().is_nonzero(), \
                    'Gradient for discriminator is zero but it should be trained or vice versa.'

            if train_generator:
                gen_opt.step()
            if train_discriminator:
                disc_opt.step()

        for x in range(10):
            # Step 1: train generator.
            train_step(train_generator=True)

            # Step 2: train discriminator.
            train_step(train_discriminator=True)

    def test_gradient_clipping(self):
        """Test gradient clipping example."""
        hvd.init()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        x = torch.ones(1, 1).requires_grad_()
        y = torch.ones(1, 1).requires_grad_()

        model = torch.nn.Linear(1, 1)
        model.weight = torch.nn.Parameter(torch.zeros(1, 1) + 0.5)
        model.bias = torch.nn.Parameter(torch.zeros(1))
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters())

        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.synchronize()
        prior_grad = model.weight.grad.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        clipped_grad = model.weight.grad.item()
        assert abs(prior_grad) > abs(clipped_grad)
        with optimizer.skip_synchronize():
            optimizer.step()

    def test_synchronize_step_warning(self):
        """
        Test that .synchronize() followed by .step() without
        optimizer.skip_synchronize() context will produce a warning.
        """
        hvd.init()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        x = torch.zeros(1, 1).requires_grad_()
        y = torch.ones(1, 1).requires_grad_()

        model = torch.nn.Linear(1, 1)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters())

        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        with warnings.catch_warnings(record=True) as ws:
            optimizer.step()
            assert len(ws) == 1
            assert 'optimizer.step() called without optimizer.skip_synchronize()' \
                in str(ws[0].message)

    def test_no_named_parameters(self):
        """Test that leaving the default named_parameters=None will not throw an error."""
        hvd.init()

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 100, 1)
                self.conv2 = torch.nn.Conv2d(100, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        model = Net()
        inp = torch.rand([1, 1, 1000, 1000])

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = hvd.DistributedOptimizer(opt)

        loss = model(inp).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_missing_named_parameters(self):
        """Test that naming half of the model parameters will throw an error."""
        hvd.init()

        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 100, 1)
                self.conv2 = torch.nn.Conv2d(100, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        model = Net()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        try:
            hvd.DistributedOptimizer(opt,
                named_parameters=list(model.named_parameters())[0:1])
            assert False, 'hvd.DistributedOptimizer did not throw error'
        except ValueError:
            pass

    def test_horovod_join_allreduce(self):
        """Test Join op with allreduce."""
        # "Join Op is not supported for PyTorch < 1.0"
        if not _v2_api:
            self.skipTest("Join Op not available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = self.filter_supported_types([torch.IntTensor, torch.LongTensor,
                     torch.FloatTensor, torch.DoubleTensor])
        if torch.cuda.is_available():
            dtypes += [torch.cuda.IntTensor, torch.cuda.LongTensor,
                       torch.cuda.FloatTensor, torch.cuda.DoubleTensor]
            if _fp16_supported:
                dtypes += [torch.cuda.HalfTensor]
        dims = [1, 2, 3]
        first_join_ranks = [0, 1]
        cachings = [False, True]
        for dtype, dim, first_join_rank, caching in itertools.product(dtypes, dims, first_join_ranks, cachings):
            torch.manual_seed(1234)

            # Use two tensors to test fusion
            tensor_a = torch.FloatTensor(*([5] * dim)).random_(-100, 100)
            tensor_a = self.cast_and_place(tensor_a, dtype)
            tensor_b = torch.FloatTensor(*([17] * dim)).random_(-100, 100)
            tensor_b = self.cast_and_place(tensor_b, dtype)

            if caching:
                handle_a = hvd.allreduce_async(tensor_a, name="tensor_a", average=True)
                handle_b = hvd.allreduce_async(tensor_b, name="tensor_b", average=True)
                averaged_a = hvd.synchronize(handle_a)
                averaged_b = hvd.synchronize(handle_b)

            if rank == first_join_rank:
                if dtype.is_cuda:
                    ret = hvd.join(hvd.local_rank())
                else:
                    ret = hvd.join()
            else:
                handle_a = hvd.allreduce_async(tensor_a, name="tensor_a", average=True)
                handle_b = hvd.allreduce_async(tensor_b, name="tensor_b", average=True)
                averaged_a = hvd.synchronize(handle_a)
                averaged_b = hvd.synchronize(handle_b)
                if dtype.is_cuda:
                    ret = hvd.join(hvd.local_rank())
                else:
                    ret = hvd.join()

                max_difference_a = averaged_a.data.sub(tensor_a * (size - 1) / size).max()
                max_difference_b = averaged_b.data.sub(tensor_b * (size - 1) / size).max()
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
                assert max_difference_a <= threshold, 'hvd.join with hvd.allreduce produces incorrect results'
                assert max_difference_b <= threshold, 'hvd.join with hvd.allreduce produces incorrect results'

    def test_horovod_join_allgather(self):
        """Test Join op with allgather."""
        # "Join Op is not supported for PyTorch < 1.0"
        if not _v2_api:
            self.skipTest("Join Op not available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dims = [17] * 3
        tensor = torch.FloatTensor(*dims)

        if rank == 0:
            if torch.cuda.is_available():
                ret = hvd.join(hvd.local_rank())
            else:
                ret = hvd.join()
        else:
            try:
                hvd.allgather(tensor)
                assert False, 'hvd.allgather did not throw error'
            except (torch.FatalError, RuntimeError):
                pass

            ret = hvd.join(hvd.local_rank())

    def test_horovod_join_broadcast(self):
        """Test Join op with allgather."""
        # "Join Op is not supported for PyTorch < 1.0"
        if not _v2_api:
            self.skipTest("Join Op not available")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dims = [17] * 3
        tensor = torch.FloatTensor(*dims)

        if rank == 0:
            ret = hvd.join(hvd.local_rank())
        else:
            try:
                broadcasted_tensor = hvd.broadcast(tensor, 1)
                assert False, 'hvd.broadcast did not throw error'
            except (torch.FatalError, RuntimeError):
                pass

            if torch.cuda.is_available():
                ret = hvd.join(hvd.local_rank())
            else:
                ret = hvd.join()
    
    def test_horovod_sync_batch_norm(self):
        """Tests Horovod version of SyncBatchNorm."""
        if not torch.cuda.is_available():
            self.skipTest("No GPUs available")

        if sys.version_info < (3,):
            # TODO: remove this check after Py2 deprecation
            self.skipTest("Python 3 only feature")

        hvd.init()

        ts_list = [
            torch.stack([
                torch.tensor([
                    [r, r + 1],
                    [r * 2, r * 2 + 1],
                    [r * 3, r * 3 + 1],
                    [r * 4, r * 4 + 1]
                ])
                for r in range(hvd.size())
            ]),
            torch.stack([
                torch.tensor([
                    [r + 1],
                    [r * 2 + 1],
                    [r * 3 + 1],
                    [r * 4 + 1]
                ])
                for r in range(hvd.size())
            ]),
        ]

        for ts in ts_list:
            sync_bn = hvd.SyncBatchNorm(num_features=4)
            sync_bn.cuda(hvd.local_rank())

            bn = torch.nn.BatchNorm1d(num_features=4)
            bn.cuda(hvd.local_rank())

            ts = ts.cuda(hvd.local_rank()).float()
            ts1 = ts.clone().requires_grad_()
            ts2 = ts.clone().requires_grad_()

            # Training
            sync_bn_out = sync_bn(ts1[hvd.rank()].unsqueeze(0))
            bn_out = bn(ts2)
            assert (sync_bn_out - bn_out[hvd.rank()].unsqueeze(0)).abs().sum() < 1e-6
            assert (sync_bn.running_mean - bn.running_mean).abs().sum() < 1e-6
            assert (sync_bn.running_var - bn.running_var).abs().sum() < 1e-6

            # Gradients
            sync_bn_out.sum().backward()
            bn_out.mean(dim=0).sum().backward()
            assert (hvd.allreduce(sync_bn.weight.grad, name='sync_bn.weight.grad') - bn.weight.grad).abs().sum() < 1e-6
            assert (hvd.allreduce(sync_bn.bias.grad, name='sync_bn.bias.grad') - bn.bias.grad).abs().sum() < 1e-6
            assert (hvd.allreduce(ts1.grad, name='ts1.grad') - ts2.grad).abs().sum() < 1e-6

if __name__ == "__main__":
   unittest.main()
