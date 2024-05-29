# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
# Modifications copyright (C) 2019 Intel Corporation
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
# =============================================================================

"""Tests for horovod.tensorflow.mpi_ops."""

from packaging import version

import itertools
import numpy as np
import os
import platform
import math
import pytest
import sys
import tensorflow as tf
from horovod.tensorflow.util import _executing_eagerly
from tensorflow.python.ops import resource_variable_ops
try:
    from tensorflow.python.ops.variables import RefVariable
except ImportError:
    # TF 2.13+
    from tensorflow.python.ops.ref_variable import RefVariable

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import mpi_env_rank_and_size, skip_or_fail_gpu_test

import horovod.tensorflow as hvd

from base_test_tensorflow import *

_IS_TF2 = version.parse(tf.__version__) >= version.parse('2.0.0')
_is_mac = platform.system() == 'Darwin'

class TensorFlowTests(BaseTensorFlowTests):
    """
    Tests for ops in horovod.tensorflow.
    """
    def __init__(self, *args, **kwargs):
        super(TensorFlowTests, self).__init__(*args, **kwargs)

    def test_gpu_required(self):
        if not tf.test.is_gpu_available(cuda_only=True):
            skip_or_fail_gpu_test(self, "No GPUs available")

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

    def test_horovod_rank_op(self):
        """Test that the rank returned by hvd.rank_op() is correct."""
        hvd.init()
        rank = self.evaluate(hvd.rank_op())
        self.assertTrue(rank == hvd.rank(),
                        "hvd.rank_op produces incorrect results")

    def test_horovod_local_rank_op(self):
        """Test that the local rank returned by hvd.local_rank_op() is correct."""
        hvd.init()
        local_rank = self.evaluate(hvd.local_rank_op())
        self.assertTrue(local_rank == hvd.local_rank(),
                        "hvd.local_rank_op produces incorrect results")

    def test_horovod_size_op(self):
        """Test that the size returned by hvd.size_op() is correct."""
        hvd.init()
        size = self.evaluate(hvd.size_op())
        self.assertTrue(size == hvd.size(),
                        "hvd.size_op produces incorrect results")

    def test_horovod_local_size_op(self):
        """Test that the local size returned by hvd.local_size_op() is correct."""
        hvd.init()
        local_size = self.evaluate(hvd.local_size_op())
        self.assertTrue(local_size == hvd.local_size(),
                        "hvd.local_size_op produces incorrect results")

    def test_horovod_allreduce_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                # prevent underflows/overflows in uint8, int8
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_average_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                averaged = hvd.allreduce(tensor, op=hvd.Average)
            # handle int8, uint8 overflows when allreduce sums up and averages the values
            tensor = tf.cast((tensor*size)/size, dtype=dtype)
            difference = tf.cast(averaged, dtype=dtype) - tensor
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_min_cpu(self):
        """Test on CPU that the allreduce correctly minimizes 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Min)
            reference = tf.math.reduce_min(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(difference)

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for min")

    def test_horovod_allreduce_max_cpu(self):
        """Test on CPU that the allreduce correctly maximizes 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Max)
            reference = tf.math.reduce_max(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(difference)

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for max")

    def test_horovod_allreduce_product_cpu(self):
        """Test on CPU that the allreduce correctly multiplies 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Product)
            reference = tf.math.reduce_prod(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(difference)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for product")

    def test_horovod_allreduce_cpu_fused(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            test = max_difference <= threshold
            tests.append(test)
        self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                        "hvd.allreduce produces incorrect results")

    # Note: TF does not support FP64 op attributes so scaling factor is cast to FP32
    # by op and loses precision. We skip FP64 version of pre/postscale tests for this reason.
    # See https://github.com/tensorflow/tensorflow/pull/39452 for PR to resolve this limitation.
    def test_horovod_allreduce_cpu_prescale(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors
           with prescaling"""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                np.random.seed(1234)
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum,
                                       prescale_factor=factor)

                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                tensor = tf.cast(tensor, tf.float32 if dtype == tf.float16 else
                                 tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * tensor, dtype) * size
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_cpu_postscale(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors
           with postscaling"""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                np.random.seed(1234)
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum,
                                       postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                multiplied = tf.cast(multiplied, tf.float32 if dtype == tf.float16 else
                                     tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * multiplied, dtype)
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_indexed_slices_cpu(self):
        """Test on CPU that the allreduce correctly sums tf.IndexedSlices."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                # prevent underflows/overflows in uint8, int8
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = self.random_uniform([17] * dim, minval, maxval)
                slice_values = tf.cast(slice_values, dtype=dtype)
                tensor = tf.IndexedSlices(tf.stack([slice_values] * 2),
                                          tf.convert_to_tensor([hvd.rank(), hvd.rank()+1]))
                result = hvd.allreduce(tensor, average=False)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            keys = result.indices
            result = tf.math.unsorted_segment_sum(result.values, keys, tf.size(tf.unique(keys)[0]))

            slices = [slice_values] * (hvd.size() + 1)
            for i in range(1, hvd.size()):
                slices[i] *= 2
            reference = tf.stack(slices)

            difference = result - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_indexed_slices_average_cpu(self):
        """Test on CPU that the allreduce correctly averages tf.IndexedSlices."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                # prevent underflows/overflows in uint8, int8
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = self.random_uniform([17] * dim, minval, maxval)
                slice_values = tf.cast(slice_values, dtype=dtype)
                tensor = tf.IndexedSlices(tf.stack([slice_values] * 2),
                                          tf.convert_to_tensor([hvd.rank(), hvd.rank()+1]))
                result = hvd.allreduce(tensor, op=hvd.Average)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            keys = result.indices
            result = tf.math.unsorted_segment_sum(result.values, keys, tf.size(tf.unique(keys)[0]))

            slices = [slice_values] * (hvd.size() + 1)
            for i in range(0, hvd.size()+1):
                if i == 0 or i == hvd.size():
                    slices[i] /= hvd.size()
                else:
                    slices[i] *= 2
                    slices[i] /= hvd.size()
            reference = tf.stack(slices)

            difference = result - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")


    def test_horovod_allreduce_indexed_slices_gpu(self):
        """Test on GPU that the allreduce correctly sums tf.IndexedSlices."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        local_rank = hvd.local_rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                # prevent underflows/overflows in uint8, int8
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = self.random_uniform([17] * dim, minval, maxval)
                slice_values = tf.cast(slice_values, dtype=dtype)
                tensor = tf.IndexedSlices(tf.stack([slice_values] * 2),
                                          tf.convert_to_tensor([hvd.rank(), hvd.rank()+1]))
                result = hvd.allreduce(tensor, average=False)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            keys = result.indices
            result = tf.math.unsorted_segment_sum(result.values, keys, tf.size(tf.unique(keys)[0]))

            slices = [slice_values] * (hvd.size() + 1)
            for i in range(1, hvd.size()):
                slices[i] *= 2
            reference = tf.stack(slices)

            difference = result - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_indexed_slices_average_gpu(self):
        """Test on GPU that the allreduce correctly averages tf.IndexedSlices."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        local_rank = hvd.local_rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                # prevent underflows/overflows in uint8, int8
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = self.random_uniform([17] * dim, minval, maxval)
                slice_values = tf.cast(slice_values, dtype=dtype)
                tensor = tf.IndexedSlices(tf.stack([slice_values] * 2),
                                          tf.convert_to_tensor([hvd.rank(), hvd.rank()+1]))
                result = hvd.allreduce(tensor, op=hvd.Average)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            keys = result.indices
            result = tf.math.unsorted_segment_sum(result.values, keys, tf.size(tf.unique(keys)[0]))

            slices = [slice_values] * (hvd.size() + 1)
            for i in range(0, hvd.size()+1):
                if i == 0 or i == hvd.size():
                    slices[i] /= hvd.size()
                else:
                    slices[i] *= 2
                    slices[i] /= hvd.size()
            reference = tf.stack(slices)

            difference = result - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_indexed_slices_op_error(self):
        """Tests that the allreduce errors out on unsupported op for tf.IndexedSlices."""
        hvd.init()
        size = hvd.size()
        tensor = tf.IndexedSlices(tf.ones(1,3),
                                  tf.convert_to_tensor([hvd.rank()]))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.allreduce(tensor, op=hvd.Min))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.allreduce(tensor, op=hvd.Max))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.allreduce(tensor, op=hvd.Product))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.allreduce(tensor, op=hvd.Adasum))

    def test_horovod_allreduce_indexed_slices_prescale_postscale_error(self):
        """Test on CPU that the allreduce correctly errors with tf.IndexedSlices with pre/postscaling."""
        hvd.init()
        factor = 0.5
        tensor = tf.IndexedSlices(tf.ones(1,3),
                                  tf.convert_to_tensor([hvd.rank()]))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.allreduce(tensor, average=False,
                                        prescale_factor=factor))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.allreduce(tensor, average=False,
                                        postscale_factor=factor))

    def test_horovod_allreduce_gpu(self):
        """Test that the allreduce works on GPUs."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")

    def test_horovod_allreduce_average_gpu(self):
        """Test that the allreduce with average works on GPUs."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                averaged = hvd.allreduce(tensor, op=hvd.Average)
            # handle int8, uint8 overflows when allreduce sums up and averages the values
            tensor = tf.cast((tensor*size)/size, dtype=dtype)
            difference = tf.cast(averaged, dtype=dtype) - tensor
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")

    def test_horovod_allreduce_min_gpu(self):
        """Test on GPU that the allreduce correctly minimizes 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Min)
            reference = tf.math.reduce_min(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(difference)

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for min")

    def test_horovod_allreduce_max_gpu(self):
        """Test on GPU that the allreduce correctly maximizes 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Max)
            reference = tf.math.reduce_max(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(difference)

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for max")

    def test_horovod_allreduce_product_gpu(self):
        """Test on GPU that the allreduce correctly multiplies 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensors = self.random_uniform([size] + [17] * dim, -100, 100)
                tensors = tf.cast(tensors, dtype=dtype)
                tensor = tensors[rank,...]
                result = hvd.allreduce(tensor, op=hvd.Product)
            reference = tf.math.reduce_prod(tensors, axis=0)
            difference = tf.cast(result, dtype=dtype) - reference
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(difference)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results for product")

    def test_horovod_allreduce_gpu_fused(self):
        """Test that the allreduce works on GPUs with Tensor Fusion.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            test = max_difference <= threshold
            tests.append(test)
        self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                        "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_multi_gpu(self):
        """Test that the allreduce works on multiple GPUs.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()
        local_size = hvd.local_size()

        # Only do this test if there are enough GPUs available.
        if len(tf.config.experimental.list_physical_devices('GPU')) < 2 * local_size:
            self.skipTest("Too few GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        iter = 0
        gpu_ids = [local_rank * 2, local_rank * 2 + 1]
        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            iter += 1
            with tf.device("/gpu:%d" % gpu_ids[(iter + local_rank) % 2]):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.allreduce on GPU produces incorrect results")

    def test_horovod_allreduce_gpu_prescale(self):
        """Test on GPU that the allreduce correctly sums 1D, 2D, 3D tensors
           with prescaling"""

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            return

        hvd.init()
        size = hvd.size()
        local_rank = hvd.local_rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                np.random.seed(1234)
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum,
                                       prescale_factor=factor)

                # Scaling done in FP64 math for integer types.
                tensor = tf.cast(tensor, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * tensor, dtype) * size
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))
            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_gpu_postscale(self):
        """Test on GPU that the allreduce correctly sums 1D, 2D, 3D tensors
           with postscaling"""

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            return

        hvd.init()
        size = hvd.size()
        local_rank = hvd.local_rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                np.random.seed(1234)
                factor = np.random.uniform()
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensor = self.random_uniform([17] * dim, minval, maxval)
                tensor = tf.cast(tensor, dtype=dtype)
                summed = hvd.allreduce(tensor, op=hvd.Sum,
                                       postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types.
                multiplied = tf.cast(multiplied, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * multiplied, dtype)
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.allreduce produces incorrect results")

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
        dims = [17 + rank] * 3
        tensor = self.random_uniform(dims, -1.0, 1.0)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.allreduce(tensor))

        # Same number of elements, different rank
        if rank == 0:
            dims = [17, 23 * 57]
        else:
            dims = [17, 23, 57]
        tensor = self.random_uniform(dims, -1.0, 1.0)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.allreduce(tensor))

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
        tensor = tf.ones(dims,
                         dtype=tf.int32 if rank % 2 == 0 else tf.float32)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.allreduce(tensor))

    def test_horovod_allreduce_cpu_gpu_error(self):
        """Test that the allreduce raises an error if different ranks try to
        perform reduction on CPU and GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        device = "/gpu:%d" % local_rank if local_rank % 2 == 0 else "/cpu:0"
        with tf.device(device):
            # Same rank, different dimension
            dims = [17] * 3
            tensor = tf.ones(dims, dtype=tf.int32)
            with self.assertRaises(tf.errors.FailedPreconditionError):
                self.evaluate(hvd.allreduce(tensor))

    def test_horovod_allreduce_grad_cpu(self):
        """Test the correctness of the allreduce gradient on CPU."""
        hvd.init()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.allreduce(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, op=hvd.Sum)

                grad_ys = tf.ones([5] * dim)
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([5] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_average_grad_cpu(self):
        """Test the correctness of the allreduce with average gradient on CPU."""
        hvd.init()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        averaged = hvd.allreduce(tensor, op=hvd.Average)
                else:
                    tensor = self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)
                    averaged = hvd.allreduce(tensor, op=hvd.Average)

                grad_ys = tf.ones([5] * dim, dtype=dtype)
                if _executing_eagerly():
                    grad_out = tape.gradient(averaged, tensor, grad_ys)
                else:
                    grad = tf.gradients(averaged, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([5] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_grad_gpu(self):
        """Test the correctness of the allreduce gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(
                        self.random_uniform([5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.allreduce(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, op=hvd.Sum)

                grad_ys = tf.ones([5] * dim)
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([5] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_average_grad_gpu(self):
        """Test the correctness of the allreduce with average gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(
                        self.random_uniform([5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        averaged = hvd.allreduce(tensor, op=hvd.Average)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    averaged = hvd.allreduce(tensor, op=hvd.Average)

                grad_ys = tf.ones([5] * dim, dtype=dtype)
                if _executing_eagerly():
                    grad_out = tape.gradient(averaged, tensor, grad_ys)
                else:
                    grad = tf.gradients(averaged, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([5] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_allreduce_cpu(self):
        """Test on CPU that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [tf.cast(self.random_uniform(
                    [17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)
            multiplied = [tensor * size for tensor in tensors]
            differences = [t1 - t2 for t1, t2 in zip(summed, multiplied)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_indexed_slices_cpu(self):
        """Test on CPU that the grouped allreduce correctly sums tf.IndexedSlices."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]
                results = hvd.grouped_allreduce(tensors, average=False)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            processed_results = []
            for x in results:
                keys = x.indices
                processed_results.append(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])))

            references = []
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(1, hvd.size()):
                    slices[i] *= 2
                references.append(tf.stack(slices))

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_mixed_indexed_slices_cpu(self):
        """Test on CPU that the grouped allreduce correctly sums a mix of tensors and tf.IndexedSlices."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                # Create indexed slices
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]

                # Append additional set of regular tensors
                for x in slice_values:
                    tensors.append(x)
                results = hvd.grouped_allreduce(tensors, average=False)

            # Convert indexed slice to tensor (summing entries for duplicate indices) or append standard tensor results
            processed_results = []
            for x in results:
                if isinstance(x, tf.IndexedSlices):
                    keys = x.indices
                    processed_results.append(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])))
                else:
                    processed_results.append(x)

            references = []
            # First append references for indexed slices
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(1, hvd.size()):
                    slices[i] *= 2
                references.append(tf.stack(slices))

            # Next, append references for standard tensors
            for x in slice_values:
                references.append(x * size)

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_average_cpu(self):
        """Test on CPU that the grouped allreduce correctly averages 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [tf.cast(self.random_uniform(
                    [17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                result = hvd.grouped_allreduce(tensors, op=hvd.Average)
            differences = [t1 - t2 for t1, t2 in zip(result, tensors)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_min_cpu(self):
        """Test on CPU that the grouped allreduce correctly finds minimum of 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                full_tensors = [tf.cast(self.random_uniform([size] + [17] * dim, -100, 100),
                                        dtype=dtype) for _ in range(5)]
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [t[rank, ...]  for t in full_tensors]
                result = hvd.grouped_allreduce(tensors, op=hvd.Min)
            reference = [tf.math.reduce_min(t, axis=0) for t in full_tensors]
            differences = [t1 - t2 for t1, t2 in zip(result, reference)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_max_cpu(self):
        """Test on CPU that the grouped allreduce correctly finds maximum of 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                full_tensors = [tf.cast(self.random_uniform([size] + [17] * dim, -100, 100),
                                        dtype=dtype) for _ in range(5)]
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [t[rank, ...]  for t in full_tensors]
                result = hvd.grouped_allreduce(tensors, op=hvd.Max)
            reference = [tf.math.reduce_max(t, axis=0) for t in full_tensors]
            differences = [t1 - t2 for t1, t2 in zip(result, reference)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_product_cpu(self):
        """Test on CPU that the grouped allreduce correctly finds product of 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                full_tensors = [tf.cast(self.random_uniform([size] + [17] * dim, -100, 100),
                                        dtype=dtype) for _ in range(5)]
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [t[rank, ...]  for t in full_tensors]
                result = hvd.grouped_allreduce(tensors, op=hvd.Product)
            reference = [tf.math.reduce_prod(t, axis=0) for t in full_tensors]
            differences = [t1 - t2 for t1, t2 in zip(result, reference)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_indexed_slices_average_cpu(self):
        """Test on CPU that the grouped allreduce correctly averages tf.IndexedSlices."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]
                results = hvd.grouped_allreduce(tensors, op=hvd.Average)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            processed_results = []
            for x in results:
                keys = x.indices
                processed_results.append(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])))

            references = []
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(0, hvd.size()+1):
                    if i == 0 or i == hvd.size():
                        slices[i] /= hvd.size()
                    else:
                        slices[i] *= 2
                        slices[i] /= hvd.size()
                references.append(tf.stack(slices))

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_mixed_indexed_slices_average_cpu(self):
        """Test on CPU that the grouped allreduce correctly averages a mix of tensors and tf.IndexedSlices."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                # Create indexed slices
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]

                # Append additional set of regular tensors
                for x in slice_values:
                    tensors.append(x)
                results = hvd.grouped_allreduce(tensors, op=hvd.Average)

            # Convert indexed slice to tensor (summing entries for duplicate indices) or append standard tensor results
            processed_results = []
            for x in results:
                if isinstance(x, tf.IndexedSlices):
                    keys = x.indices
                    processed_results.append(tf.cast(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])),
                                                     dtype=dtype))
                else:
                    processed_results.append(x)

            references = []
            # First append references for indexed slices
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(0, hvd.size() + 1):
                    if i == 0 or i == hvd.size():
                        slices[i] /= hvd.size()
                    else:
                        slices[i] *= 2
                        slices[i] /= hvd.size()
                references.append(tf.cast(tf.stack(slices), dtype=dtype))

            # Next, append references for standard tensors
            for x in slice_values:
                references.append(x)

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype in [tf.int8, tf.uint8] else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_indexed_slices_op_error(self):
        """Tests that the grouped allreduce errors out on unsupported op for tf.IndexedSlices."""
        hvd.init()
        slice_values = [tf.ones(1,3) for _ in range(5)]
        tensors = [tf.IndexedSlices(slice_values,
                                    tf.convert_to_tensor([hvd.rank()])) for x in slice_values]
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Min))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Max))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Product))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Adasum))

    def test_horovod_grouped_allreduce_mixed_indexed_slices_op_error(self):
        """Tests that the grouped allreduce errors out on unsupported op for tf.IndexedSlices in list
           with a mix of tensors and tf.IndexedSlice."""
        hvd.init()
        slice_values = [tf.ones(1,3) for _ in range(5)]
        tensors = [tf.IndexedSlices(slice_values,
                                    tf.convert_to_tensor([hvd.rank()])) for x in slice_values]

        # Append additional set of regular tensors
        for x in slice_values:
            tensors.append(x)

        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Min))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Max))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Product))
        with self.assertRaises(NotImplementedError):
            self.evaluate(hvd.grouped_allreduce(tensors, op=hvd.Adasum))

    def test_horovod_grouped_allreduce_gpu(self):
        """Test on GPU that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [tf.cast(self.random_uniform(
                    [17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)
            multiplied = [tensor * size for tensor in tensors]
            differences = [t1 - t2 for t1, t2 in zip(summed, multiplied)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce on GPU produces incorrect results")

    def test_horovod_grouped_allreduce_indexed_slices_gpu(self):
        """Test on GPU that the grouped allreduce correctly sums tf.IndexedSlices."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]
                results = hvd.grouped_allreduce(tensors, average=False)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            processed_results = []
            for x in results:
                keys = x.indices
                processed_results.append(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])))

            references = []
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(1, hvd.size()):
                    slices[i] *= 2
                references.append(tf.stack(slices))

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_mixed_indexed_slices_gpu(self):
        """Test on GPU that the grouped allreduce correctly sums a mix of tensors and tf.IndexedSlices."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                # Create indexed slices
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]

                # Append additional set of regular tensors
                for x in slice_values:
                    tensors.append(x)
                results = hvd.grouped_allreduce(tensors, average=False)

            # Convert indexed slice to tensor (summing entries for duplicate indices) or append standard tensor results
            processed_results = []
            for x in results:
                if isinstance(x, tf.IndexedSlices):
                    keys = x.indices
                    processed_results.append(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])))
                else:
                    processed_results.append(x)

            references = []
            # First append references for indexed slices
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(1, hvd.size()):
                    slices[i] *= 2
                references.append(tf.stack(slices))

            # Next, append references for standard tensors
            for x in slice_values:
                references.append(x * size)

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_average_gpu(self):
        """Test on GPU that the grouped allreduce correctly averages 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [tf.cast(self.random_uniform(
                    [17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                result = hvd.grouped_allreduce(tensors, op=hvd.Average)
            differences = [t1 - t2 for t1, t2 in zip(result, tensors)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_min_gpu(self):
        """Test on GPU that the grouped allreduce correctly finds minimum of 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                full_tensors = [tf.cast(self.random_uniform([size] + [17] * dim, -100, 100),
                                        dtype=dtype) for _ in range(5)]
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [t[rank, ...]  for t in full_tensors]
                result = hvd.grouped_allreduce(tensors, op=hvd.Min)
            reference = [tf.math.reduce_min(t, axis=0) for t in full_tensors]
            differences = [t1 - t2 for t1, t2 in zip(result, reference)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_max_gpu(self):
        """Test on GPU that the grouped allreduce correctly finds maximum of 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                full_tensors = [tf.cast(self.random_uniform([size] + [17] * dim, -100, 100),
                                        dtype=dtype) for _ in range(5)]
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [t[rank, ...]  for t in full_tensors]
                result = hvd.grouped_allreduce(tensors, op=hvd.Max)
            reference = [tf.math.reduce_max(t, axis=0) for t in full_tensors]
            differences = [t1 - t2 for t1, t2 in zip(result, reference)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            threshold = 0
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_product_gpu(self):
        """Test on GPU that the grouped allreduce correctly finds product of 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                full_tensors = [tf.cast(self.random_uniform([size] + [17] * dim, -100, 100),
                                        dtype=dtype) for _ in range(5)]
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                tensors = [t[rank, ...]  for t in full_tensors]
                result = hvd.grouped_allreduce(tensors, op=hvd.Product)
            reference = [tf.math.reduce_prod(t, axis=0) for t in full_tensors]
            differences = [t1 - t2 for t1, t2 in zip(result, reference)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")
            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_indexed_slices_average_gpu(self):
        """Test on GPU that the grouped allreduce correctly averages tf.IndexedSlices."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]
                results = hvd.grouped_allreduce(tensors, op=hvd.Average)

            # Convert indexed slice to tensor (summing entries for duplicate indices)
            processed_results = []
            for x in results:
                keys = x.indices
                processed_results.append(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])))

            references = []
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(0, hvd.size()+1):
                    if i == 0 or i == hvd.size():
                        slices[i] /= hvd.size()
                    else:
                        slices[i] *= 2
                        slices[i] /= hvd.size()
                references.append(tf.stack(slices))

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_mixed_indexed_slices_average_gpu(self):
        """Test on GPU that the grouped allreduce correctly averages a mix of tensors and tf.IndexedSlices."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]

        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % hvd.local_rank()):
                maxval = 100 if dtype not in [tf.uint8, tf.int8] else 1
                minval = -maxval if dtype not in [tf.uint8] else 0
                # Create indexed slices
                slice_values = [tf.cast(self.random_uniform([17] * dim, minval, maxval), dtype=dtype) for _ in range(5)]
                tensors = [tf.IndexedSlices(tf.stack([x]*2),
                                            tf.convert_to_tensor([hvd.rank(), hvd.rank()+1])) for x in slice_values]

                # Append additional set of regular tensors
                for x in slice_values:
                    tensors.append(x)
                results = hvd.grouped_allreduce(tensors, op=hvd.Average)

            # Convert indexed slice to tensor (summing entries for duplicate indices) or append standard tensor results
            processed_results = []
            for x in results:
                if isinstance(x, tf.IndexedSlices):
                    keys = x.indices
                    processed_results.append(tf.cast(tf.math.unsorted_segment_sum(x.values, keys, tf.size(tf.unique(keys)[0])),
                                                     dtype=dtype))
                else:
                    processed_results.append(x)

            references = []
            # First append references for indexed slices
            for x in slice_values:
                slices = [x] * (hvd.size() + 1)
                for i in range(0, hvd.size() + 1):
                    if i == 0 or i == hvd.size():
                        slices[i] /= hvd.size()
                    else:
                        slices[i] *= 2
                        slices[i] /= hvd.size()
                references.append(tf.cast(tf.stack(slices), dtype=dtype))

            # Next, append references for standard tensors
            for x in slice_values:
                references.append(x)

            differences = [t1 - t2 for t1, t2 in zip(processed_results, references)]
            differences = [tf.cast(diff, tf.int32) if dtype in [tf.int8, tf.uint8] else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_grad_cpu(self):
        """Test the correctness of the grouped allreduce gradient on CPU."""
        hvd.init()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    tensors = [self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)) for _ in range(5)]
                    with tf.GradientTape(persistent=True) as tape:
                        summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)
                else:
                    tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)

                grads_ys = [tf.ones([5] * dim, dtype=dtype) for _ in range(5)]
                if _executing_eagerly():
                    grads_out = [tape.gradient(s, t, g) for s, t, g in zip(summed, tensors, grads_ys)]
                else:
                    grads = [tf.gradients(s, t, g)[0] for s, t, g in zip(summed, tensors, grads_ys)]
                    grads_out = [self.evaluate(grad) for grad in grads]

            expected = np.ones([5] * dim) * size
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_allreduce_grad_gpu(self):
        """Test the correctness of the grouped allreduce gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                if _executing_eagerly():
                    tensors = [self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)) for _ in range(5)]
                    with tf.GradientTape(persistent=True) as tape:
                        summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)
                else:
                    tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    summed = hvd.grouped_allreduce(tensors, op=hvd.Sum)

                grads_ys = [tf.ones([5] * dim, dtype=dtype) for _ in range(5)]
                if _executing_eagerly():
                    grads_out = [tape.gradient(s, t, g) for s, t, g in zip(summed, tensors, grads_ys)]
                else:
                    grads = [tf.gradients(s, t, g)[0] for s, t, g in zip(summed, tensors, grads_ys)]
                    grads_out = [self.evaluate(grad) for grad in grads]

            expected = np.ones([5] * dim) * size
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allgather_cpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = tf.ones([17] * dim) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/cpu:0"):
                gathered = hvd.allgather(tensor)

            gathered_tensor = self.evaluate(gathered)
            self.assertEqual(list(gathered_tensor.shape),
                             [17 * size] + [17] * (dim - 1))

            for i in range(size):
                rank_tensor = tf.slice(gathered_tensor,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                self.assertEqual(list(rank_tensor.shape), [17] * dim)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                    "hvd.allgather produces incorrect gathered tensor")


    def test_horovod_allgather_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = tf.ones([17] * dim) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/gpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)

            gathered_tensor = self.evaluate(gathered)
            self.assertEqual(list(gathered_tensor.shape),
                             [17 * size] + [17] * (dim - 1))

            for i in range(size):
                rank_tensor = tf.slice(gathered_tensor,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                self.assertEqual(list(rank_tensor.shape), [17] * dim)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                    "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_fused_cpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        tests = []
        shape_tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = tf.ones([17] * dim) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/cpu:0"):
                gathered = hvd.allgather(tensor)

            shape_tests.append(
                tf.reduce_all(tf.equal(tf.shape(gathered),
                                       [17 * size] + [17] * (dim - 1))))

            for i in range(size):
                rank_tensor = tf.slice(gathered,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2

                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                tests.append(
                    tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value)))

            shape_tests_passed, value_tests_passed = \
                self.evaluate([tf.reduce_all(shape_tests), tf.reduce_all(tests)])

            self.assertTrue(shape_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

            self.assertTrue(value_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_fused_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors
        with Tensor Fusion."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        tests = []
        shape_tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            tensor = tf.ones([17] * dim) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/gpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)

            shape_tests.append(
                tf.reduce_all(tf.equal(tf.shape(gathered),
                                       [17 * size] + [17] * (dim - 1))))

            for i in range(size):
                rank_tensor = tf.slice(gathered,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2

                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                tests.append(
                    tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value)))

            shape_tests_passed, value_tests_passed = \
                self.evaluate([tf.reduce_all(shape_tests), tf.reduce_all(tests)])

            self.assertTrue(shape_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

            self.assertTrue(value_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_variable_size_fused_cpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors with
        Tensor Fusion, even if those tensors have different sizes along the
        first dim."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        tests = []
        shape_tests = []

        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/cpu:0"):
                gathered = hvd.allgather(tensor)
            shape_tests.append(
                tf.reduce_all(tf.equal(tf.shape(gathered),
                             [sum(tensor_sizes)] + [17] * (dim - 1))))

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = tf.slice(
                    gathered, [sum(tensor_sizes[:i])] + [0] * (dim - 1),
                    rank_size)
                self.assertEqual(list(rank_tensor.shape), rank_size)
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2

                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                tests.append(tf.reduce_all(
                    tf.equal(tf.cast(rank_tensor, tf.int32), value)))

            shape_tests_passed, value_tests_passed = \
                self.evaluate([tf.reduce_all(shape_tests), tf.reduce_all(tests)])

            self.assertTrue(shape_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

            self.assertTrue(value_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_variable_size_fused_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors with
        Tensor Fusion, even if those tensors have different sizes along the
        first dim."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        tests = []
        shape_tests = []

        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/gpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)
            shape_tests.append(
                tf.reduce_all(tf.equal(tf.shape(gathered),
                             [sum(tensor_sizes)] + [17] * (dim - 1))))

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = tf.slice(
                    gathered, [sum(tensor_sizes[:i])] + [0] * (dim - 1),
                    rank_size)
                self.assertEqual(list(rank_tensor.shape), rank_size)
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2

                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                tests.append(tf.reduce_all(
                    tf.equal(tf.cast(rank_tensor, tf.int32), value)))

            shape_tests_passed, value_tests_passed = \
                self.evaluate([tf.reduce_all(shape_tests), tf.reduce_all(tests)])

            self.assertTrue(shape_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

            self.assertTrue(value_tests_passed,
                            "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_variable_size_gpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/gpu:%d" % local_rank):
                gathered = hvd.allgather(tensor)

            gathered_tensor = self.evaluate(gathered)
            expected_size = sum(tensor_sizes)
            self.assertEqual(list(gathered_tensor.shape),
                             [expected_size] + [17] * (dim - 1))

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = tf.slice(
                    gathered, [sum(tensor_sizes[:i])] + [0] * (dim - 1),
                    rank_size)
                self.assertEqual(list(rank_tensor.shape), rank_size)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                    "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_variable_size_cpu(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            # Support tests up to MPI Size of 35
            if size > 35:
                break

            tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
            tensor_sizes = tensor_sizes[:size]

            tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            with tf.device("/cpu:0"):
                gathered = hvd.allgather(tensor)

            gathered_tensor = self.evaluate(gathered)
            expected_size = sum(tensor_sizes)
            self.assertEqual(list(gathered_tensor.shape),
                             [expected_size] + [17] * (dim - 1))

            for i in range(size):
                rank_size = [tensor_sizes[i]] + [17] * (dim - 1)
                rank_tensor = tf.slice(
                    gathered, [sum(tensor_sizes[:i])] + [0] * (dim - 1),
                    rank_size)
                self.assertEqual(list(rank_tensor.shape), rank_size)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                    "hvd.allgather produces incorrect gathered tensor")

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
        tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.allgather(tensor))

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
        dtype = tf.int32 if rank % 2 == 0 else tf.float32
        tensor = tf.ones(tensor_size, dtype=dtype) * rank
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.allgather(tensor))

    def test_horovod_allgather_grad_cpu(self):
        """Test the correctness of the allgather gradient on CPU."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor_sizes = [3, 2, 7, 4, 6, 8, 10] * 5
            tensor_sizes = tensor_sizes[:size]

            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    with tf.GradientTape() as tape:
                        tensor = self.tfe.Variable(
                            tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank)
                        if dtype == tf.bool:
                            tensor = tensor % 2
                        tensor = tf.cast(tensor, dtype=dtype)
                        gathered = hvd.allgather(tensor)
                        grad_list = []
                        for r, tensor_size in enumerate(tensor_sizes):
                            g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                            grad_list.append(g)
                        grad_ys = tf.concat(grad_list, axis=0)
                    grad_out = tape.gradient(gathered, tensor, grad_ys)
                else:
                    tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
                    if dtype == tf.bool:
                        tensor = tensor % 2
                    tensor = tf.cast(tensor, dtype=dtype)
                    gathered = hvd.allgather(tensor)

                    grad_list = []
                    for r, tensor_size in enumerate(tensor_sizes):
                        g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                        grad_list.append(g)
                    grad_ys = tf.concat(grad_list, axis=0)

                    grad = tf.gradients(gathered, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(
                [tensor_sizes[rank]] + [17] * (dim - 1)
            ) * rank
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" %
                            (grad_out, expected, str(err)))

    def test_horovod_allgather_grad_gpu(self):
        """Test the correctness of the allgather gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor_sizes = [3, 2, 7, 4, 6, 8, 10] * 5
            tensor_sizes = tensor_sizes[:size]

            with tf.device("/gpu:%d" % local_rank):
                if _executing_eagerly():
                    with tf.GradientTape() as tape:
                        tensor = self.tfe.Variable(
                            tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank)
                        if dtype == tf.bool:
                            tensor = tensor % 2
                        tensor = tf.cast(tensor, dtype=dtype)
                        gathered = hvd.allgather(tensor)
                        grad_list = []
                        for r, tensor_size in enumerate(tensor_sizes):
                            g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                            grad_list.append(g)
                        grad_ys = tf.concat(grad_list, axis=0)
                    grad_out = tape.gradient(gathered, tensor, grad_ys)
                else:
                    tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
                    if dtype == tf.bool:
                        tensor = tensor % 2
                    tensor = tf.cast(tensor, dtype=dtype)
                    gathered = hvd.allgather(tensor)

                    grad_list = []
                    for r, tensor_size in enumerate(tensor_sizes):
                        g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                        grad_list.append(g)
                    grad_ys = tf.concat(grad_list, axis=0)

                    grad = tf.gradients(gathered, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(
                [tensor_sizes[rank]] + [17] * (dim - 1)
            ) * rank
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" %
                            (grad_out, expected, str(err)))

    def test_horovod_broadcast_cpu(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on CPU."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            with tf.device("/cpu:0"):
                tensor = tf.ones([17] * dim) * rank
                root_tensor = tf.ones([17] * dim) * root_rank
                if dtype == tf.bool:
                    tensor = tensor % 2
                    root_tensor = root_tensor % 2
                tensor = tf.cast(tensor, dtype=dtype)
                root_tensor = tf.cast(root_tensor, dtype=dtype)
                broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            self.assertTrue(
                self.evaluate(tf.reduce_all(tf.equal(
                    tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                "hvd.broadcast produces incorrect broadcasted tensor")

    def test_horovod_broadcast_gpu(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = tf.ones([17] * dim) * rank
            root_tensor = tf.ones([17] * dim) * root_rank
            if dtype == tf.bool:
                tensor = tensor % 2
                root_tensor = root_tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            root_tensor = tf.cast(root_tensor, dtype=dtype)
            with tf.device("/gpu:%d" % local_rank):
                broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            self.assertTrue(
                self.evaluate(tf.reduce_all(tf.equal(
                    tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                "hvd.broadcast produces incorrect broadcasted tensor")

    def test_horovod_broadcast_inplace_cpu(self):
        """Test that the inplace broadcast correctly broadcasts 1D, 2D, 3D variables on CPU."""
        if version.parse(tf.__version__) < version.parse('2.6.0'):
            self.skipTest("Custom Ops using resource variables only work with TF 2.6+")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [tf.uint8, tf.int8,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for use_resource in [False, True]:
            if not use_resource and _executing_eagerly():
                continue
            for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
                with tf.device("/cpu:0"):
                    if dtype == tf.bool:
                        initial_value = tf.cast((tf.ones([17] * dim) * rank) % 2, dtype)
                    else:
                        initial_value = tf.cast(tf.ones([17] * dim) * rank, dtype)
                    if not hvd._executing_eagerly():
                        if use_resource:
                            var = resource_variable_ops.ResourceVariable(initial_value)
                        else:
                            var = RefVariable(initial_value)
                        init = tf.compat.v1.global_variables_initializer()
                        self.evaluate(init)
                    else:
                        assert use_resource
                        var = self.tfe.Variable(initial_value)
                    root_tensor = tf.ones([17] * dim) * root_rank
                    if dtype == tf.bool:
                        root_tensor = root_tensor % 2
                    broadcasted_tensor, = hvd.broadcast_([var], root_rank)
                    self.assertEqual(var.dtype.base_dtype, dtype)
                    self.assertEqual(broadcasted_tensor.dtype.base_dtype, dtype)
                    np.testing.assert_array_equal(self.evaluate(broadcasted_tensor), self.evaluate(var),
                                                  err_msg="broadcasted_var and var may not differ, actually they should have the same underlying buffer")
                    self.assertTrue(
                        self.evaluate(tf.reduce_all(tf.equal(
                            tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                        "Inplace hvd.broadcast_ produces incorrect broadcasted variable value")

    def test_horovod_broadcast_inplace_gpu(self):
        """Test that the inplace broadcast correctly broadcasts 1D, 2D, 3D variables on GPU."""
        if version.parse(tf.__version__) < version.parse('2.6.0'):
            self.skipTest("Custom Ops using resource variables only work with TF 2.6+")

        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # dtypes that are supported both for variable assignments and by Horovod
        dtypes = [tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for use_resource in [False, True]:
            if not use_resource and _executing_eagerly():
                continue
            for counter, (dtype, dim, root_rank) in enumerate(itertools.product(dtypes, dims, root_ranks)):
                with tf.device("/gpu:%d" % local_rank):
                    if dtype == tf.bool:
                        initial_value = tf.cast((tf.ones([17] * dim) * rank) % 2, dtype)
                    else:
                        initial_value = tf.cast(tf.ones([17] * dim) * rank, dtype)
                    root_tensor = tf.ones([17] * dim) * root_rank
                    if dtype == tf.bool:
                        root_tensor = root_tensor % 2
                    if not hvd._executing_eagerly():
                        if use_resource:
                            var = resource_variable_ops.ResourceVariable(initial_value)
                        else:
                            var = RefVariable(initial_value)
                        init = tf.compat.v1.global_variables_initializer()
                        self.evaluate(init)
                    else:
                        assert use_resource
                        var = self.tfe.Variable(initial_value)
                    broadcasted_tensor, = hvd.broadcast_([var], root_rank)
                    self.assertEqual(var.dtype.base_dtype, dtype)
                    self.assertEqual(broadcasted_tensor.dtype.base_dtype, dtype)
                    np.testing.assert_array_equal(self.evaluate(broadcasted_tensor), self.evaluate(var),
                                                  err_msg="broadcasted_var and var may not differ, actually they should have the same underlying buffer")
                    self.assertTrue(
                        self.evaluate(tf.reduce_all(tf.equal(
                            tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                        "Inplace hvd.broadcast_ produces incorrect broadcasted variable value")

    def test_horovod_broadcast_inplace_multiple_cpu(self):
        """Test that the inplace broadcast correctly broadcasts multiple variables on CPU."""
        if version.parse(tf.__version__) < version.parse('2.6.0'):
            self.skipTest("Custom Ops using resource variables only work with TF 2.6+")

        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        dtypes = [tf.float32]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for use_resource in [False, True]:
            if not use_resource and _executing_eagerly():
                continue
            for dtype, root_rank in itertools.product(dtypes, root_ranks):
                with tf.device("/cpu:0"):
                    variables = []
                    root_tensors = []
                    for dim in dims:
                        initial_value = tf.cast(tf.ones([17] * dim) * rank, dtype)
                        if not hvd._executing_eagerly():
                            if use_resource:
                                var = resource_variable_ops.ResourceVariable(initial_value, name=f"dim_{dim}_var")
                            else:
                                var = RefVariable(initial_value, name=f"dim_{dim}_var")
                            init = tf.compat.v1.global_variables_initializer()
                            self.evaluate(init)
                        else:
                            assert use_resource
                            var = self.tfe.Variable(initial_value, name=f"dim_{dim}_var")
                        root_tensor = tf.ones([17] * dim) * root_rank
                        variables.append(var)
                        root_tensors.append(root_tensor)

                    broadcasted_tensors = hvd.broadcast_(variables, root_rank)
                    for broadcasted_tensor, var, root_tensor in zip(broadcasted_tensors, variables, root_tensors):
                        self.assertEqual(var.dtype.base_dtype, dtype)
                        self.assertEqual(broadcasted_tensor.dtype.base_dtype, dtype)
                        np.testing.assert_array_equal(self.evaluate(broadcasted_tensor), self.evaluate(var),
                                                      err_msg="broadcasted_var and var may not differ, actually they should have the same underlying buffer")
                        self.assertTrue(
                            self.evaluate(tf.reduce_all(tf.equal(
                                tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                            "Inplace hvd.broadcast_ produces incorrect broadcasted variable value")

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
        tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.broadcast(tensor, 0))

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
        dtype = tf.int32 if rank % 2 == 0 else tf.float32
        tensor = tf.ones(tensor_size, dtype=dtype) * rank
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.broadcast(tensor, 0))

    def test_horovod_broadcast_rank_error(self):
        """Test that the broadcast returns an error if different ranks
        specify different root rank."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor = tf.ones([17] * 3, dtype=tf.float32)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.broadcast(tensor, rank))

    def test_horovod_broadcast_grad_cpu(self):
        """Test the correctness of the broadcast gradient on CPU."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(tf.ones([5] * dim) * rank)
                else:
                    tensor = tf.ones([5] * dim) * rank
                if dtype == tf.bool:
                    tensor = tensor % 2
                if _executing_eagerly():
                    with tf.GradientTape() as tape:
                        tensor = tf.cast(tensor, dtype=dtype)
                        broadcasted_tensor = hvd.broadcast(tensor, root_rank)
                    grad_out = tape.gradient(broadcasted_tensor, tensor)
                else:
                    tensor = tf.cast(tensor, dtype=dtype)
                    broadcasted_tensor = hvd.broadcast(tensor, root_rank)
                    grad_ys = tf.ones([5] * dim)
                    grad = tf.gradients(broadcasted_tensor, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            c = 1 if rank == root_rank else 0
            expected = np.ones([5] * dim) * c
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_broadcast_grad_gpu(self):
        """Test the correctness of the broadcast gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        root_ranks = list(range(size))
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            with tf.device("/gpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(tf.ones([5] * dim) * rank)
                else:
                    tensor = tf.ones([5] * dim) * rank
                if dtype == tf.bool:
                    tensor = tensor % 2
                if _executing_eagerly():
                    with tf.GradientTape() as tape:
                        tensor = tf.cast(tensor, dtype=dtype)
                        broadcasted_tensor = hvd.broadcast(tensor, root_rank)
                    grad_out = tape.gradient(broadcasted_tensor, tensor)
                else:
                    tensor = tf.cast(tensor, dtype=dtype)
                    broadcasted_tensor = hvd.broadcast(tensor, root_rank)
                    grad_ys = tf.ones([5] * dim)
                    grad = tf.gradients(broadcasted_tensor, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            c = 1 if rank == root_rank else 0
            expected = np.ones([5] * dim) * c
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_alltoall_cpu(self):
        """Test that the alltoall correctly distributes 1D, 2D, and 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.uint16, tf.int16,
                                              tf.int32, tf.int64, tf.float16, tf.float32,
                                              tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                splits = tf.convert_to_tensor([rank+1] * size, dtype=tf.int32)
                collected, received_splits = hvd.alltoall(tensor, splits)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

                self.assertSequenceEqual(self.evaluate(received_splits).tolist(), [rk + 1 for rk in range(size)],
                                         "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_gpu(self):
        """Test that the alltoall correctly distributes 1D, 2D, and 3D tensors on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                splits = tf.convert_to_tensor([rank+1] * size, dtype=tf.int32)
                collected, received_splits = hvd.alltoall(tensor, splits)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

                self.assertSequenceEqual(self.evaluate(received_splits).tolist(), [rk + 1 for rk in range(size)],
                                         "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_equal_split_cpu(self):
        """Test that the alltoall correctly distributes 1D tensors with default splitting."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.uint16, tf.int16,
                                              tf.int32, tf.int64, tf.float16, tf.float32,
                                              tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                collected = hvd.alltoall(tensor)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

    def test_horovod_alltoall_equal_split_gpu(self):
        """Test that the alltoall correctly distributes 1D tensors with default splitting on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                collected = hvd.alltoall(tensor)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

    def test_horovod_alltoall_empty_cpu(self):
        """Test that the alltoall correctly deals with an empty input tensor."""
        hvd.init()
        size = hvd.size()

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.uint16, tf.int16,
                                              tf.int32, tf.int64, tf.float16, tf.float32,
                                              tf.float64])
        for dtype in dtypes:
            with tf.device("/cpu:0"):
                vals = [[] for i in range(size)]
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                collected = hvd.alltoall(tensor)

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), 0)),
                    "hvd.alltoall collected wrong number of values")

    def test_horovod_alltoall_empty_gpu(self):
        """Test that the alltoall correctly deals with an empty input tensor."""
        # ncclGroupEnd failed: invalid usage

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        for dtype in dtypes:
            with tf.device("/gpu:%s" % local_rank):
                vals = [[] for i in range(size)]
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                collected = hvd.alltoall(tensor)

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), 0)),
                    "hvd.alltoall collected wrong number of values")

    def test_horovod_alltoall_one_rank_sends_nothing_cpu(self):
        """Test where one rank sends nothing in an alltoall."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()

        if hvd.size() < 2:
            self.skipTest("Only one worker available")

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.uint16, tf.int16,
                                              tf.int32, tf.int64, tf.float16, tf.float32,
                                              tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if rank == 1:
                    splits = tf.convert_to_tensor([0] * size, dtype=tf.int32)
                    vals = []
                    tensor = tf.convert_to_tensor(vals, dtype=dtype)
                    tensor = tf.reshape(tensor, shape=[0] + (dim-1)*[2])
                else:
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    vals = []
                    for i in range(size):
                        vals += [i] * (rank + 1)
                    tensor = tf.convert_to_tensor(vals, dtype=dtype)
                    for _ in range(dim - 1):
                        tensor = tf.expand_dims(tensor, axis=1)
                        tensor = tf.concat([tensor, tensor], axis=1)

                collected, received_splits = hvd.alltoall(tensor, splits, name="a2a")

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1)
                                                               - (1+1) * 2 ** (dim-1)  # subtract missing rank 1 contributions
                                           )),
                    "hvd.alltoall collected wrong number of values")

                self.assertSequenceEqual(self.evaluate(received_splits).tolist(),
                                         [rk + 1 if rk != 1 else 0 for rk in range(size)],
                                         "hvd.alltoall returned incorrect received_splits")


    def test_horovod_alltoall_one_rank_sends_nothing_gpu(self):
        """Test where one rank sends nothing in an alltoall."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()
        rank = hvd.rank()

        if hvd.size() < 2:
            self.skipTest("Only one worker available")

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                if rank == 1:
                    splits = tf.convert_to_tensor([0] * size, dtype=tf.int32)
                    vals = []
                    tensor = tf.convert_to_tensor(vals, dtype=dtype)
                    tensor = tf.reshape(tensor, shape=[0] + (dim-1)*[2])
                else:
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    vals = []
                    for i in range(size):
                        vals += [i] * (rank + 1)
                    tensor = tf.convert_to_tensor(vals, dtype=dtype)
                    for _ in range(dim - 1):
                        tensor = tf.expand_dims(tensor, axis=1)
                        tensor = tf.concat([tensor, tensor], axis=1)

                collected, received_splits = hvd.alltoall(tensor, splits, name="a2a")

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1)
                                                               - (1+1) * 2 ** (dim-1)  # subtract missing rank 1 contributions
                                           )),
                    "hvd.alltoall collected wrong number of values")

                self.assertSequenceEqual(self.evaluate(received_splits).tolist(),
                                         [rk + 1 if rk != 1 else 0 for rk in range(size)],
                                         "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_one_rank_receives_nothing_cpu(self):
        """Test where one rank receives nothing in an alltoall."""
        hvd.init()
        size = hvd.size()
        rank = hvd.rank()

        if hvd.size() < 2:
            self.skipTest("Only one worker available")

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.uint16, tf.int16,
                                              tf.int32, tf.int64, tf.float16, tf.float32,
                                              tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                # send nothing to rank 0
                splits = tf.convert_to_tensor([0] + [rank + 1] * (size - 1), dtype=tf.int32)
                vals = []
                for i in range(1, size):
                    vals += [i] * (rank + 1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                    tensor = tf.expand_dims(tensor, axis=1)
                    tensor = tf.concat([tensor, tensor], axis=1)

                collected, received_splits = hvd.alltoall(tensor, splits, name="a2a")
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")
                if rank == 0:
                    expected_size = 0
                    expected_rsplits = [0] * size
                else:
                    expected_size = size * (size + 1) // 2 * 2**(dim - 1)
                    expected_rsplits = [rk + 1 for rk in range(size)]
                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), expected_size)),
                    "hvd.alltoall collected wrong number of values")
                self.assertSequenceEqual(self.evaluate(received_splits).tolist(), expected_rsplits,
                                         "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_one_rank_receives_nothing_gpu(self):
        """Test where one rank receives nothing in an alltoall."""
        # ncclGroupEnd failed: invalid usage

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        size = hvd.size()
        rank = hvd.rank()
        local_rank = hvd.local_rank()

        if hvd.size() < 2:
            self.skipTest("Only one worker available")

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                # send nothing to rank 0
                splits = tf.convert_to_tensor([0] + [rank + 1] * (size - 1), dtype=tf.int32)
                vals = []
                for i in range(1, size):
                    vals += [i] * (rank + 1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                    tensor = tf.expand_dims(tensor, axis=1)
                    tensor = tf.concat([tensor, tensor], axis=1)

                collected, received_splits = hvd.alltoall(tensor, splits, name="a2a")
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")
                if rank == 0:
                    expected_size = 0
                    expected_rsplits = [0] * size
                else:
                    expected_size = size * (size + 1) // 2 * 2**(dim - 1)
                    expected_rsplits = [rk + 1 for rk in range(size)]
                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), expected_size)),
                    "hvd.alltoall collected wrong number of values")
                self.assertSequenceEqual(self.evaluate(received_splits).tolist(), expected_rsplits,
                                         "hvd.alltoall returned incorrect received_splits")


    def test_horovod_alltoall_zero_splits_cpu(self):
        """Test alltoall with some ranks not participating / splits set to zero."""
        hvd.init()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        active_ranks = range(0, hvd.size() // 2)
        silent_ranks = range(hvd.size() // 2, hvd.size())

        active_splits = [1 if r in active_ranks else 0 for r in range(hvd.size())]
        active_shape = [sum(active_splits), 4]
        silent_splits = [0] * hvd.size()
        silent_shape = [0, 4]

        with tf.device("/cpu:0"):
            if hvd.rank() in active_ranks:
                source_tensor = tf.fill(active_shape, value=tf.cast(hvd.rank(), tf.int32))
                splits = tf.convert_to_tensor(active_splits)
            else:
                source_tensor = tf.fill(silent_shape, value=tf.cast(hvd.rank(), tf.int32))
                splits = tf.convert_to_tensor(silent_splits)
            collected, received_splits = hvd.alltoall(source_tensor, splits, name="alltoall_zero_splits")
            result = self.evaluate(collected)

        if hvd.rank() in active_ranks:
            expected_result_shape = active_shape
        else:
            expected_result_shape = silent_shape
        self.assertSequenceEqual(result.shape, expected_result_shape)
        if hvd.rank() in active_ranks:
            for r_idx, r in enumerate(active_ranks):
                self.assertTrue(np.all(result[r_idx, ...] == r))
        else:
            self.assertLen(result, 0)
        if hvd.rank() in active_ranks:
            expected_rsplits = active_splits
        else:
            expected_rsplits = silent_splits
        self.assertSequenceEqual(self.evaluate(received_splits).tolist(), expected_rsplits,
                                 "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_zero_splits_gpu(self):
        """Test alltoall with some ranks not participating / splits set to zero."""
        # ncclCommInitRank failed: invalid usage
        hvd.init()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        active_ranks = range(0, hvd.size() // 2)
        silent_ranks = range(hvd.size() // 2, hvd.size())

        active_splits = [1 if r in active_ranks else 0 for r in range(hvd.size())]
        active_shape = [sum(active_splits), 4]
        silent_splits = [0] * hvd.size()
        silent_shape = [0, 4]

        with tf.device("/gpu:%s" % hvd.local_rank()):
            if hvd.rank() in active_ranks:
                source_tensor = tf.fill(active_shape, value=tf.cast(hvd.rank(), tf.int32))
                splits = tf.convert_to_tensor(active_splits)
            else:
                source_tensor = tf.fill(silent_shape, value=tf.cast(hvd.rank(), tf.int32))
                splits = tf.convert_to_tensor(silent_splits)
            collected, received_splits = hvd.alltoall(source_tensor, splits, name="alltoall_zero_splits")
            result = self.evaluate(collected)

        if hvd.rank() in active_ranks:
            expected_result_shape = active_shape
        else:
            expected_result_shape = silent_shape
        self.assertSequenceEqual(result.shape, expected_result_shape)
        if hvd.rank() in active_ranks:
            for r_idx, r in enumerate(active_ranks):
                self.assertTrue(np.all(result[r_idx, ...] == r))
        else:
            self.assertLen(result, 0)
        if hvd.rank() in active_ranks:
            expected_rsplits = active_splits
        else:
            expected_rsplits = silent_splits
        self.assertSequenceEqual(self.evaluate(received_splits).tolist(), expected_rsplits,
                                 "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_type_error(self):
        """Test that the alltoall returns an error if the tensor types differ
           across the processes."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        with tf.device("/cpu:0"):
            if rank % 2:
                tensor = tf.ones([size], dtype=tf.int32)
            else:
                tensor = tf.ones([size], dtype=tf.float32)

            with self.assertRaises(tf.errors.FailedPreconditionError):
                self.evaluate(hvd.alltoall(tensor))

    def test_horovod_alltoall_equal_split_length_error(self):
        """Test that the alltoall with default splitting returns an error if the tensor length is not a multiple
        of the number of workers."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        with tf.device("/cpu:0"):
            tensor = tf.ones([size + 1], dtype=tf.float32)

            with self.assertRaises(tf.errors.InvalidArgumentError):
                self.evaluate(hvd.alltoall(tensor))

    def test_horovod_alltoall_splits_error(self):
        """Test that the alltoall returns an error if the sum of the splits entries exceeds
        the first dimension of the input tensor."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        with tf.device("/cpu:0"):
            tensor = tf.ones([size-1], dtype=tf.float32)
            splits = tf.ones([size], dtype=tf.int32)

            with self.assertRaises(tf.errors.InvalidArgumentError):
                self.evaluate(hvd.alltoall(tensor))

    def test_horovod_alltoall_rank_error(self):
        """Test that the alltoall returns an error if any dimension besides
        the first is different among the tensors being processed."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        tensor_size = [2 * size] * 3
        tensor_size[1] = 10 * (rank + 1)
        with tf.device("/cpu:0"):
            tensor = tf.ones(tensor_size)

            with self.assertRaises(tf.errors.FailedPreconditionError):
                self.evaluate(hvd.alltoall(tensor))

    def test_horovod_alltoall_grad_cpu(self):
        """Test the correctness of the alltoall gradient on CPU."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)

                if _executing_eagerly():
                    tensor = self.tfe.Variable(tensor)
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    with tf.GradientTape() as tape:
                        collected, received_splits = hvd.alltoall(tensor, splits)
                else:
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    collected, received_splits = hvd.alltoall(tensor, splits)

                grad_ys = tf.ones(tf.shape(collected))
                if _executing_eagerly():
                    grad_out = tape.gradient(collected, tensor, grad_ys)
                else:
                    grad = tf.gradients(collected, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(tensor.get_shape().as_list())
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_alltoall_grad_gpu(self):
        """Test the correctness of the alltoall gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)

                if _executing_eagerly():
                    tensor = self.tfe.Variable(tensor)
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    with tf.GradientTape() as tape:
                        collected, received_splits = hvd.alltoall(tensor, splits)
                else:
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    collected, received_splits = hvd.alltoall(tensor, splits)

                grad_ys = tf.ones(tf.shape(collected))
                if _executing_eagerly():
                    grad_out = tape.gradient(collected, tensor, grad_ys)
                else:
                    grad = tf.gradients(collected, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(tensor.get_shape().as_list())
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_alltoall_equal_split_grad_cpu(self):
        """Test the correctness of the alltoall gradient with default splitting on CPU."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)

                if _executing_eagerly():
                    tensor = self.tfe.Variable(tensor)
                    with tf.GradientTape() as tape:
                        collected = hvd.alltoall(tensor)
                else:
                    collected = hvd.alltoall(tensor)

                grad_ys = tf.ones(tf.shape(collected))
                if _executing_eagerly():
                    grad_out = tape.gradient(collected, tensor, grad_ys)
                else:
                    grad = tf.gradients(collected, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(tensor.get_shape().as_list())
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_alltoall_equal_split_grad_gpu(self):
        """Test the correctness of the alltoall gradient with default splitting on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                vals = []
                for i in range(size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)

                if _executing_eagerly():
                    tensor = self.tfe.Variable(tensor)
                    with tf.GradientTape() as tape:
                        collected = hvd.alltoall(tensor)
                else:
                    collected = hvd.alltoall(tensor)

                grad_ys = tf.ones(tf.shape(collected))
                if _executing_eagerly():
                    grad_out = tape.gradient(collected, tensor, grad_ys)
                else:
                    grad = tf.gradients(collected, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones(tensor.get_shape().as_list())
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_broadcast_eager_mode_error(self):
        """Test that tries to broadcast tensorflow global variables
        in eager execution mode. This call should raise a RuntimeError."""

        if not hvd.util._executing_eagerly():
            self.skipTest("Only in eager execution mode")

        with self.assertRaises(RuntimeError):
            hvd.broadcast_global_variables(root_rank=0)

    def test_horovod_broadcast_graph_mode(self):
        """Test that tries to broadcast tensorflow global variables
        in graph execution mode. This call should not raise any exception."""

        if hvd.util._executing_eagerly():
            self.skipTest("Not in eager execution mode")

        hvd.broadcast_global_variables(root_rank=0)

    def test_compression_fp16(self):
        valid_dtypes = [tf.float16, tf.float32, tf.float64]
        invalid_dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                          tf.int32, tf.int64, tf.bool]

        tensor_size = [17] * 3
        compression = hvd.Compression.fp16

        for dtype in valid_dtypes:
            tensor = tf.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, tf.float16)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            actual = self.evaluate(tensor_decompressed)
            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - actual)
            self.assertLess(err, 0.00000001)

        for dtype in invalid_dtypes:
            tensor = tf.ones(tensor_size, dtype=dtype)

            tensor_compressed, ctx = compression.compress(tensor)
            self.assertEqual(tensor_compressed.dtype, dtype)

            tensor_decompressed = compression.decompress(tensor_compressed, ctx)
            self.assertEqual(tensor_decompressed.dtype, dtype)

            actual = self.evaluate(tensor_decompressed)
            expected = np.ones(tensor_size)
            err = np.linalg.norm(expected - actual)
            self.assertLess(err, 0.00000001)

    def test_broadcast_object(self):
        hvd.init()

        with tf.device("/cpu:0"):
            expected_obj = {
                'hello': 123,
                0: [1, 2]
            }
            obj = expected_obj if hvd.rank() == 0 else {}

            obj = hvd.broadcast_object(obj, root_rank=0)
            self.assertDictEqual(obj, expected_obj)

    def test_broadcast_object_fn(self):
        if hvd._executing_eagerly() or _IS_TF2:
            # Only for TF 1.0 in graph mode
            return

        hvd.init()

        with tf.device("/cpu:0"):
            expected_obj = {
                'hello': 123,
                0: [1, 2]
            }
            obj = expected_obj if hvd.rank() == 0 else {}

            bcast = hvd.broadcast_object_fn(root_rank=0)
            obj = bcast(obj)
            self.assertDictEqual(obj, expected_obj)

    def test_allgather_object(self):
        hvd.init()

        with tf.device("/cpu:0"):
            d = {'metric_val_1': hvd.rank()}
            if hvd.rank() == 1:
                d['metric_val_2'] = 42

            results = hvd.allgather_object(d)

            expected = [{'metric_val_1': i} for i in range(hvd.size())]
            if hvd.size() > 1:
                expected[1] = {'metric_val_1': 1, 'metric_val_2': 42}

            self.assertEqual(len(results), hvd.size())
            self.assertListEqual(results, expected)

    def test_elastic_state(self):
        if not hvd._executing_eagerly() and _IS_TF2:
            # Only support TF 2.0 in eager mode
            return

        hvd.init()

        with tf.device("/cpu:0"):
            v = 1.0 if hvd.rank() == 0 else 2.0
            weights1 = [
                np.array([[v, v], [v, v]]),
                np.array([v, v])
            ]
            vars1 = [tf.Variable(arr) for arr in weights1]

            weights2 = [
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([0.0, 0.0])
            ]

            if not hvd._executing_eagerly():
                init = tf.global_variables_initializer()
                self.evaluate(init)

            state = hvd.elastic.TensorFlowState(vars1, batch=20 + hvd.rank(), epoch=10 + hvd.rank())
            state.sync()

            weights1 = [np.ones_like(w) for w in weights1]

            # After sync, all values should match the root rank
            for w in self.evaluate(vars1):
                self.assertAllClose(w, np.ones_like(w))
            assert state.batch == 20
            assert state.epoch == 10

            # Partially modify then restore
            self.assign(vars1, weights2)
            state.batch = 21
            state.epoch = 11

            state.restore()

            for w1, w2 in zip(self.evaluate(vars1), weights1):
                self.assertAllClose(w1, w2)
            assert state.batch == 20
            assert state.epoch == 10

            # Partially modify then commit
            self.assign(vars1, weights2)
            state.batch = 21
            state.epoch = 11

            state.commit()
            state.restore()

            for w1, w2 in zip(self.evaluate(vars1), weights2):
                self.assertAllClose(w1, w2)
            assert state.batch == 21
            assert state.epoch == 11

    def test_horovod_join_allreduce(self):
        """Test that the hvd.join with allreduce works on GPUs."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            self.skipTest("Not compiled with HOROVOD_GPU_ALLREDUCE")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")


        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        first_join_ranks = [0, 1]

        for dtype, dim, first_join_rank in itertools.product(dtypes, dims, first_join_ranks):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                if local_rank == first_join_rank:
                    ret = self.evaluate(hvd.join())
                else:
                    summed = hvd.allreduce(tensor, op=hvd.Sum)
                    multiplied = tensor * (size-1)
                    max_difference = tf.reduce_max(tf.abs(summed - multiplied))

                    if size <= 3 or dtype in [tf.int32, tf.int64]:
                        threshold = 0
                    elif size < 10:
                        threshold = 1e-4
                    elif size < 15:
                        threshold = 5e-4
                    else:
                        return
                    diff = self.evaluate(max_difference)
                    ret = self.evaluate(hvd.join())
                    self.assertTrue(diff <= threshold,
                                    "hvd.join with hvd.allreduce on GPU produces incorrect results")

                self.assertNotEqual(ret, first_join_rank,
                                    msg="The return value of hvd.join() may not be equal to first_join_rank")
                ret_values = hvd.allgather_object(ret)
                self.assertSequenceEqual(ret_values, [ret] * size,
                                         msg="hvd.join() did not return the same value on each rank")

    def test_horovod_syncbn_gpu(self):
        """Test that the SyncBatchNormalization implementation is correct on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        hvd.init()
        with tf.device("/gpu:%d" % hvd.local_rank()):
            x_list = [
                tf.convert_to_tensor(np.stack([
                    np.array([
                        [r, r + 1],
                        [r * 2, r * 2 + 1],
                        [r * 3, r * 3 + 1],
                        [r * 4, r * 4 + 1]
                    ], dtype=np.float32)
                    for r in range(hvd.size())
                ]), np.float32),
                tf.convert_to_tensor(np.stack([
                    np.array([
                        [r + 1],
                        [r * 2 + 1],
                        [r * 3 + 1],
                        [r * 4 + 1]
                    ], dtype=np.float32)
                    for r in range(hvd.size())
                ]), np.float32),
            ]

            for x in x_list:
                bn = tf.keras.layers.BatchNormalization(axis=1, fused=False)
                sync_bn = hvd.SyncBatchNormalization(axis=1)
                bn_func = bn(x, training=True)
                sync_bn_func = sync_bn(tf.expand_dims(x[hvd.rank()], 0), training=True)

                try:
                  init = tf.global_variables_initializer()
                except AttributeError:
                  init = tf.compat.v1.global_variables_initializer()
                self.evaluate(init)
                bn_out = self.evaluate(bn_func)
                sync_bn_out = self.evaluate(sync_bn_func)

                self.assertAllClose(sync_bn_out, np.expand_dims(bn_out[hvd.rank()], 0))
                self.assertAllClose(self.evaluate(sync_bn.moving_mean), self.evaluate(bn.moving_mean))
                self.assertAllClose(self.evaluate(sync_bn.moving_variance), self.evaluate(bn.moving_variance))

    def test_horovod_syncbn_cpu(self):
        """Test that the SyncBatchNormalization implementation is correct on CPU."""

        hvd.init()
        with tf.device("/cpu:0"):
            x_list = [
                tf.convert_to_tensor(np.stack([
                    np.array([
                        [r, r + 1],
                        [r * 2, r * 2 + 1],
                        [r * 3, r * 3 + 1],
                        [r * 4, r * 4 + 1]
                    ], dtype=np.float32)
                    for r in range(hvd.size())
                ]), np.float32),
                tf.convert_to_tensor(np.stack([
                    np.array([
                        [r + 1],
                        [r * 2 + 1],
                        [r * 3 + 1],
                        [r * 4 + 1]
                    ], dtype=np.float32)
                    for r in range(hvd.size())
                ]), np.float32),
            ]

            for x in x_list:
                bn = tf.keras.layers.BatchNormalization(axis=1, fused=False)
                sync_bn = hvd.SyncBatchNormalization(axis=1)
                bn_func = bn(x, training=True)
                sync_bn_func = sync_bn(tf.expand_dims(x[hvd.rank()], 0), training=True)

                try:
                  init = tf.global_variables_initializer()
                except AttributeError:
                  init = tf.compat.v1.global_variables_initializer()
                self.evaluate(init)
                bn_out = self.evaluate(bn_func)
                sync_bn_out = self.evaluate(sync_bn_func)

                self.assertAllClose(sync_bn_out, np.expand_dims(bn_out[hvd.rank()], 0))
                self.assertAllClose(self.evaluate(sync_bn.moving_mean), self.evaluate(bn.moving_mean))
                self.assertAllClose(self.evaluate(sync_bn.moving_variance), self.evaluate(bn.moving_variance))

    def _grad_agg_compute_expected_value(self, backward_passes_per_step, batch_id):
        sum_per_aggregation = 0.0
        for _ in range(backward_passes_per_step):
            grads_for_batch = 0.0
            for rank in range(hvd.size()):
                grads_for_batch += rank

            # Apply `average_aggregated_gradients`.
            grads_for_batch /= float(backward_passes_per_step)

            # Averages across workers.
            sum_per_aggregation += grads_for_batch / float(hvd.size())

        aggregations_completed = math.floor((batch_id + 1) / backward_passes_per_step)
        return aggregations_completed * sum_per_aggregation


    def test_local_gradient_aggregation(self):
        """Test that local gradient aggregation works as expected."""

        if _executing_eagerly() or not hasattr(tf, 'ConfigProto'):
            pytest.skip("Gradient aggregation for Legacy Optimizer only support graph mode.")

        hvd.init()
        with self.test_session(config=config) as sess:

            class TestOptimizer(tf.compat.v1.train.Optimizer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.variable = tf.Variable([0.0])

                def compute_gradients(self, *args, **kwargs):
                    return [(tf.constant([float(hvd.rank())]), self.variable)]

                def _apply_dense(self, grad, var):
                    return var.assign_add(grad)

            backward_passes_per_step = 4
            opt = TestOptimizer(name="test", use_locking=False)
            opt = hvd.DistributedOptimizer(
                optimizer=opt,
                backward_passes_per_step=backward_passes_per_step,
                average_aggregated_gradients=True,
            )

            grads_and_vars = opt.compute_gradients()

            # Use custom variable for global_step instead of default one from
            # `tf.compat.v1.train.get_or_create_global_step` to ensure
            # the correct counter is being incremented.
            global_step = tf.compat.v1.get_variable(
                'test_global_step', initializer=tf.constant(0, dtype=tf.int64),
                trainable=False)
            # Check the global_step before the update_op to ensure that the global step is
            # well-ordered in the graph, occuring within apply_gradients().
            with tf.compat.v1.control_dependencies(g for g, _ in grads_and_vars):
                # Use +0 instead of tf.identity() since tf.identity() has some weird semantics
                # with control_dependencies: github.com/tensorflow/tensorflow/issues/4663
                global_step_before_update = global_step + 0
            with tf.compat.v1.control_dependencies([global_step_before_update]):
                update_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            steps_seen = []
            for idx in range(10):
                step, _ = sess.run((global_step_before_update, update_op))
                computed_value = sess.run(opt._optimizer.variable.read_value())[0]
                steps_seen.append(step)
                self.assertEquals(computed_value, self._grad_agg_compute_expected_value(
                    backward_passes_per_step, idx))

            assert steps_seen == list(range(10))

    def test_local_gradient_aggregation_sparse(self):
        """Test the the gradient aggregation works for sparse gradients."""

        if _executing_eagerly() or not hasattr(tf, 'ConfigProto'):
            pytest.skip("Gradient aggregation for Legacy Optimizer only support graph mode.")

        hvd.init()
        with self.test_session(config=config) as sess:
            class SparseGradientOptimizer(tf.compat.v1.train.Optimizer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.variable = tf.Variable([0.0])
                    self.grad = tf.IndexedSlices(
                        tf.constant([float(hvd.rank())]), tf.constant([0]), tf.constant([1])
                    )

                def compute_gradients(self, *args, **kwargs):
                    return [(self.grad, self.variable)]

                def _apply_dense(self, grad, var):
                    return var.assign_add(grad)

            backward_passes_per_step = 4
            opt = SparseGradientOptimizer(name="sparse_test", use_locking=False)
            opt = hvd.DistributedOptimizer(
                optimizer=opt,
                backward_passes_per_step=backward_passes_per_step,
                average_aggregated_gradients=True,
                sparse_as_dense=True,
            )

            grads_and_vars = opt.compute_gradients()
            update_op = opt.apply_gradients(grads_and_vars)

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            for idx in range(10):
                sess.run(update_op)
                computed_value = sess.run(opt._optimizer.variable.read_value())[0]
                self.assertEquals(computed_value, self._grad_agg_compute_expected_value(
                    backward_passes_per_step, idx))

    def test_horovod_reducescatter_cpu(self):
        """Test on CPU that the reducescatter correctly sums or averages and scatters 1D, 2D, 3D tensors."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=red_op)
            if red_op == hvd.Sum:
                expected = tf.cast(tensor[rank * 4:(rank + 1) * 4] * size, reduced.dtype)
            elif red_op == hvd.Average:
                expected = tf.cast(tensor[rank * 4:(rank + 1) * 4], reduced.dtype)
            max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_cpu_prescale(self):
        """Test on CPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors with prescaling."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                factor = np.random.uniform()
                tensor = self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=hvd.Sum, prescale_factor=factor)

                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                tensor = tf.cast(tensor, tf.float32 if dtype == tf.float16 else
                                 tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)

                expected = tf.cast(factor * tensor[rank * 4:(rank + 1) * 4], reduced.dtype) * size
                max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_cpu_postscale(self):
        """Test on CPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                factor = np.random.uniform()
                tensor = self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=hvd.Sum, postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                multiplied = tf.cast(multiplied, tf.float32 if dtype == tf.float16 else
                                     tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)

                expected = tf.cast(factor * multiplied[rank * 4:(rank + 1) * 4], reduced.dtype)
                max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")


    def test_horovod_reducescatter_cpu_fused(self):
        """Test on CPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors
        with Tensor Fusion."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)
            expected = tensor[rank * 4:(rank + 1) * 4] * size
            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            test = max_difference <= threshold
            tests.append(test)
        self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                        "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_cpu_uneven(self):
        """Test on CPU that the reducescatter correctly sums and scatters tensors that cannot
           be distributed evenly over the Horovod processes"""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        for dtype in dtypes:
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [size * 4 + size // 2], -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)

            if rank < size // 2:
                low = rank * (4 + 1)
                high = low + (4 + 1)
            else:
                low = (size // 2) * (4 + 1) + (rank - size // 2) * 4
                high = low + 4
            expected = tensor[low:high] * size

            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff, expected_shape, summed_shape = self.evaluate([max_difference, tf.shape(expected), tf.shape(summed)])
            self.assertSequenceEqual(expected_shape, summed_shape,
                                     "hvd.reducescatter produces incorrect shapes")
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_cpu_uneven_fused(self):
        """Test on CPU that the reducescatter correctly sums and scatters tensors that cannot
           be distributed evenly over the Horovod processes, with Tensor Fusion"""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        indices = [0, 1, 2, 3]
        tests = []
        infos = []
        for dtype, index in itertools.product(dtypes, indices):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [size * 4 + size // 2], -100, 100,
                    seed=1234 + index,
                    dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)

            if rank < size // 2:
                low = rank * (4 + 1)
                high = low + (4 + 1)
            else:
                low = (size // 2) * (4 + 1) + (rank - size // 2) * 4
                high = low + 4
            expected = tensor[low:high] * size

            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            test = max_difference <= threshold
            tests.append(test)
        i = self.evaluate([tf.reduce_all(tests)] + infos)
        successful = i.pop(0)
        self.assertTrue(successful,
                        "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_scalar_error(self):
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        with tf.device("/cpu:0"):
            scalar = tf.constant(rank, dtype=tf.float32)
            with self.assertRaises((tf.errors.InvalidArgumentError, ValueError, tf.errors.FailedPreconditionError)):
                self.evaluate(hvd.reducescatter(scalar))

    def test_horovod_reducescatter_gpu(self):
        """Test that the reducescatter works on GPUs."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=red_op)
            if red_op == hvd.Sum:
                expected = tf.cast(tensor[rank * 4:(rank + 1) * 4] * size, reduced.dtype)
            elif red_op == hvd.Average:
                expected = tf.cast(tensor[rank * 4:(rank + 1) * 4], reduced.dtype)
            max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter on GPU produces incorrect results")

    def test_horovod_reducescatter_gpu_prescale(self):
        """Test that the reducescatter works on GPUs with prescaling."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(123456)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                factor = np.random.uniform()
                tensor = self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=hvd.Sum, prescale_factor=factor)

                # Scaling done in FP64 math for integer types
                tensor = tf.cast(tensor, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                expected = tf.cast(factor * tensor[rank * 4:(rank + 1) * 4], reduced.dtype) * size
                max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter on GPU produces incorrect results")

    def test_horovod_reducescatter_gpu_postscale(self):
        """Test on GPU that the reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                factor = np.random.uniform()
                tensor = self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                reduced = hvd.reducescatter(tensor, op=hvd.Sum, postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types.
                multiplied = tf.cast(multiplied, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)

                expected = tf.cast(factor * multiplied[rank * 4:(rank + 1) * 4], reduced.dtype)
                max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")


    def test_horovod_reducescatter_gpu_fused(self):
        """Test that the reducescatter works on GPUs with Tensor Fusion.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4] * dim, -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)
            expected = tensor[rank * 4:(rank + 1) * 4] * size
            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            test = max_difference <= threshold
            tests.append(test)
        self.assertTrue(self.evaluate(tf.reduce_all(tests)),
                        "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_gpu_uneven(self):
        """Test on GPU that the reducescatter correctly sums and scatters tensors that cannot
           be distributed evenly over the Horovod processes"""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        for dtype in dtypes:
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4 + size // 2], -100, 100, dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)

            if rank < size // 2:
                low = rank * (4 + 1)
                high = low + (4 + 1)
            else:
                low = (size // 2) * (4 + 1) + (rank - size // 2) * 4
                high = low + 4
            expected = tensor[low:high] * size

            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff, expected_shape, summed_shape = self.evaluate([max_difference, tf.shape(expected),
                                                                tf.shape(summed)])
            self.assertSequenceEqual(expected_shape, summed_shape,
                                     "hvd.reducescatter produces incorrect shapes")
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_gpu_uneven_fused(self):
        """Test on GPU that the reducescatter correctly sums and scatters tensors that cannot
           be distributed evenly over the Horovod processes, with Tensor Fusion"""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()

        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        indices = [0, 1, 2, 3]
        tests = []
        infos = []
        for dtype, index in itertools.product(dtypes, indices):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [size * 4 + size // 2], -100, 100,
                    seed=1234 + index,
                    dtype=dtype)
                summed = hvd.reducescatter(tensor, op=hvd.Sum)

            if rank < size // 2:
                low = rank * (4 + 1)
                high = low + (4 + 1)
            else:
                low = (size // 2) * (4 + 1) + (rank - size // 2) * 4
                high = low + 4
            expected = tensor[low:high] * size

            max_difference = tf.reduce_max(tf.abs(summed - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            test = max_difference <= threshold
            tests.append(test)
            # infos.append({"0_t": tensor, "1_e": expected, "2_s": summed, "3_ok": tf.reduce_all(test)})
        i = self.evaluate([tf.reduce_all(tests)] + infos)
        succesful = i.pop(0)
        self.assertTrue(succesful,
                        "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_error(self):
        """Test that the reducescatter raises an error if different ranks try to
        send tensors of different rank or dimension."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Same rank, different dimension
        dims = [17 + rank] * 3
        tensor = self.random_uniform(dims, -1.0, 1.0)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.reducescatter(tensor))

        # Same number of elements, different rank
        if rank == 0:
            dims = [17, 23 * 57]
        else:
            dims = [17, 23, 57]
        tensor = self.random_uniform(dims, -1.0, 1.0)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.reducescatter(tensor))

    def test_horovod_reducescatter_type_error(self):
        """Test that the reducescatter raises an error if different ranks try to
        send tensors of different type."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        # Same rank, different dimension
        dims = [17] * 3
        tensor = tf.ones(dims,
                         dtype=tf.int32 if rank % 2 == 0 else tf.float32)
        with self.assertRaises(tf.errors.FailedPreconditionError):
            self.evaluate(hvd.reducescatter(tensor))

    def test_horovod_reducescatter_grad_cpu(self):
        """Test the correctness of the reducescatter gradient on CPU."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(self.random_uniform(
                        [size * 4] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.reducescatter(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform(
                        [size * 4] * dim, -100, 100, dtype=dtype)
                    summed = hvd.reducescatter(tensor, op=hvd.Sum)

                grad_ys = tf.ones([4] + [size * 4] * (dim - 1))
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([size * 4] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_reducescatter_grad_gpu(self):
        """Test the correctness of the reducescatter gradient on GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                if _executing_eagerly():
                    tensor = self.tfe.Variable(
                        self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.reducescatter(tensor, op=hvd.Sum)
                else:
                    tensor = self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                    summed = hvd.reducescatter(tensor, op=hvd.Sum)

                grad_ys = tf.ones([4] + [size * 4] * (dim - 1))
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([size * 4] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_reducescatter_cpu(self):
        """Test on CPU that the grouped reducescatter correctly sums or averages and scatters 1D, 2D, 3D tensors."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/cpu:0"):
                tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=red_op)
            if red_op == hvd.Sum:
                expected = [tf.cast(tensor[rank * 4:(rank + 1) * 4] * size, reduced[0].dtype) for tensor in tensors]
            elif red_op == hvd.Average:
                expected = [tf.cast(tensor[rank * 4:(rank + 1) * 4], reduced[0].dtype) for tensor in tensors]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_reducescatter produces incorrect results")

    def test_horovod_grouped_reducescatter_cpu_prescale(self):
        """Test on CPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with prescaling."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                factor = np.random.uniform()
                tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=hvd.Sum, prescale_factor=factor)

                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                tensors = [tf.cast(t, tf.float32 if dtype == tf.float16 else
                           tf.float64 if dtype in int_types else dtype) for t in tensors]
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)

                expected = [tf.cast(factor * t[rank * 4:(rank + 1) * 4], reduced[0].dtype) * size
                            for t in tensors]
                max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_reducescatter produces incorrect results")

    def test_horovod_grouped_reducescatter_cpu_postscale(self):
        """Test on CPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                factor = np.random.uniform()
                tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=hvd.Sum, postscale_factor=factor)

                multiplied = [t * size for t in tensors]
                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                multiplied = [tf.cast(t, tf.float32 if dtype == tf.float16 else
                              tf.float64 if dtype in int_types else dtype) for t in multiplied]
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)

                expected = [tf.cast(factor * m[rank * 4:(rank + 1) * 4], reduced[0].dtype)
                            for m in multiplied]
                max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_reducescatter produces incorrect results")


    def test_horovod_grouped_reducescatter_scalar_error(self):
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        with tf.device("/cpu:0"):
            tensor_and_scalar = [tf.zeros((2,1), dtype=tf.float32), tf.constant(rank, dtype=tf.float32)]
            with self.assertRaises((tf.errors.InvalidArgumentError, ValueError, tf.errors.FailedPreconditionError)):
                self.evaluate(hvd.grouped_reducescatter(tensor_and_scalar))

    def test_horovod_grouped_reducescatter_grad_cpu(self):
        """Test the correctness of the grouped reducescatter gradient on CPU."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    tensors = [self.tfe.Variable(self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype))
                               for _ in range(5)]
                    with tf.GradientTape(persistent=True) as tape:
                        summed = hvd.grouped_reducescatter(tensors, op=hvd.Sum)
                else:
                    tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                               for _ in range(5)]
                    summed = hvd.grouped_reducescatter(tensors, op=hvd.Sum)

                grads_ys = [tf.ones([4] + [size * 4] * (dim - 1)) for _ in range(5)]
                if _executing_eagerly():
                    grads_out = [tape.gradient(s, t, g) for s, t, g in zip(summed, tensors, grads_ys)]
                else:
                    grads = [tf.gradients(s, t, g)[0] for s, t, g in zip(summed, tensors, grads_ys)]
                    grads_out = self.evaluate(grads)

            expected = np.ones([size * 4] * dim) * size
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_reducescatter_gpu(self):
        """Test that the grouped reducescatter works on GPUs."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype)
                           for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=red_op)
            if red_op == hvd.Sum:
                expected = [tf.cast(tensor[rank * 4:(rank + 1) * 4] * size, reduced[0].dtype)
                            for tensor in tensors]
            elif red_op == hvd.Average:
                expected = [tf.cast(tensor[rank * 4:(rank + 1) * 4], reduced[0].dtype)
                            for tensor in tensors]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                return

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.grouped_reducescatter on GPU produces incorrect results")

    def test_horovod_grouped_reducescatter_gpu_prescale(self):
        """Test on GPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with prescaling."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")
        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                factor = np.random.uniform()
                tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=hvd.Sum, prescale_factor=factor)

                # Scaling done in FP64 math for integer types
                tensors = [tf.cast(t, tf.float64 if dtype in int_types else dtype) for t in tensors]
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)

                expected = [tf.cast(factor * t[rank * 4:(rank + 1) * 4], reduced[0].dtype) * size
                            for t in tensors]
                max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_reducescatter produces incorrect results")

    def test_horovod_grouped_reducescatter_gpu_postscale(self):
        """Test on GPU that the grouped reducescatter correctly sums and scatters 1D, 2D, 3D tensors with postscaling"""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")
        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        hvd.init()
        local_rank = hvd.local_rank()
        rank = hvd.rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        np.random.seed(12345)
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                factor = np.random.uniform()
                tensors = [self.random_uniform([size * 4] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                reduced = hvd.grouped_reducescatter(tensors, op=hvd.Sum, postscale_factor=factor)

                multiplied = [t * size for t in tensors]
                # Scaling done in FP64 math for integer types
                multiplied = [tf.cast(t, tf.float64 if dtype in int_types else dtype) for t in multiplied]
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)

                expected = [tf.cast(factor * m[rank * 4:(rank + 1) * 4], reduced[0].dtype)
                            for m in multiplied]
                max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in int_types:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_reducescatter produces incorrect results")


    def test_horovod_grouped_allgather_cpu(self):
        """Test that the grouped allgather correctly gathers 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensors = [tf.ones([17] * dim) * rank for _ in range(5)]
            if dtype == tf.bool:
                tensors = [tensor % 2 for tensor in tensors]
            tensors = [tf.cast(tensor, dtype=dtype) for tensor in tensors]
            with tf.device("/cpu:0"):
                gathered = hvd.grouped_allgather(tensors)

            gathered_tensors = self.evaluate(gathered)
            for gathered_tensor in gathered_tensors:
                self.assertEqual(list(gathered_tensor.shape),
                                 [17 * size] + [17] * (dim - 1))

            for i in range(size):
                rank_tensors = [tf.slice(gathered_tensor,
                                         [i * 17] + [0] * (dim - 1),
                                         [17] + [-1] * (dim - 1))
                                for gathered_tensor in gathered_tensors]
                self.assertEqual([rank_tensor.shape for rank_tensor in rank_tensors], len(tensors) * [[17] * dim])
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(all(self.evaluate(tf.reduce_all(
                    tf.equal(tf.cast(rank_tensor, tf.int32), value))) for rank_tensor in rank_tensors),
                    "hvd.grouped_allgather produces incorrect gathered tensor")

    def test_horovod_grouped_allgather_grad_cpu(self):
        """Test the correctness of the grouped allgather gradient on CPU."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor_sizes = [3, 2, 7, 4, 6, 8, 10] * 5
            tensor_sizes = tensor_sizes[:size]

            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    with tf.GradientTape(persistent=True) as tape:
                        tensors = [self.tfe.Variable(tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank)
                                   for _ in range(5)]
                        tensors = [tf.cast(tensor, dtype=dtype) for tensor in tensors]
                        gathered = hvd.grouped_allgather(tensors)
                        grad_list = []
                        for r, tensor_size in enumerate(tensor_sizes):
                            g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                            grad_list.append(g)
                        grads_ys = [tf.concat(grad_list, axis=0) for _ in range(5)]
                    grads_out = [tape.gradient(x, t, g) for x, t, g in zip(gathered, tensors, grads_ys)]
                else:
                    tensors = [tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
                               for _ in range(5)]
                    tensors = [tf.cast(tensor, dtype=dtype) for tensor in tensors]
                    gathered = hvd.grouped_allgather(tensors)

                    grad_list = []
                    for r, tensor_size in enumerate(tensor_sizes):
                        g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                        grad_list.append(g)
                    grad_ys = tf.concat(grad_list, axis=0)
                    grads = [tf.gradients(x, t, grad_ys)[0] for x, t in zip(gathered, tensors)]

                    grads_out = self.evaluate(grads)

            expected = np.ones(
                [tensor_sizes[rank]] + [17] * (dim - 1)
            ) * rank
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" %
                                (grad_out, expected, str(err)))

    def test_horovod_grouped_allgather_gpu(self):
        """Test that the grouped allgather correctly gathers 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensors = [tf.ones([17] * dim) * rank for _ in range(5)]
            if dtype == tf.bool:
                tensors = [tensor % 2 for tensor in tensors]
            tensors = [tf.cast(tensor, dtype=dtype) for tensor in tensors]
            with tf.device("/gpu:%d" % local_rank):
                gathered = hvd.grouped_allgather(tensors)

            gathered_tensors = self.evaluate(gathered)
            for gathered_tensor in gathered_tensors:
                self.assertEqual(list(gathered_tensor.shape),
                                 [17 * size] + [17] * (dim - 1))

            for i in range(size):
                rank_tensors = [tf.slice(gathered_tensor,
                                         [i * 17] + [0] * (dim - 1),
                                         [17] + [-1] * (dim - 1))
                                for gathered_tensor in gathered_tensors]
                self.assertEqual([rank_tensor.shape for rank_tensor in rank_tensors], len(tensors) * [[17] * dim])
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = i
                else:
                    value = i % 2
                self.assertTrue(all(self.evaluate(tf.reduce_all(
                    tf.equal(tf.cast(rank_tensor, tf.int32), value))) for rank_tensor in rank_tensors),
                    "hvd.grouped_allgather produces incorrect gathered tensor")

    def test_partial_distributed_gradient_tape(self):
        """ Note: test makes most sense with more than 1 nodes. """
        hvd.init()
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        # the keras model has 3 layers, we test cases with 0, 1, and 2 local layers.
        for num_local_layers in range(3):
            model = tf.keras.models.Sequential()
            initializer = tf.keras.initializers.Constant(hvd.rank())
            model.add(tf.keras.layers.Dense(2, input_shape=(3,), kernel_initializer=initializer, bias_initializer=initializer))
            model.add(tf.keras.layers.RepeatVector(3))
            model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, kernel_initializer=initializer, bias_initializer=initializer)))
            model.compile(loss=tf.keras.losses.MSE,
                            metrics=[tf.keras.metrics.categorical_accuracy])

            X = np.random.random((1, 3))
            Y = np.random.random((1, 3, 3))

            try:
                init = tf.global_variables_initializer()
            except AttributeError:
                init = tf.compat.v1.global_variables_initializer()
            self.evaluate(init)

            with tf.GradientTape(persistent=True) as tape:
                p = model(X, training=True)
                l = model.loss(Y, p)

            gradients = tape.gradient(l, model.trainable_weights)

            # deem local layers
            local_layers = model.layers[:num_local_layers]
            if _IS_TF2:
                var_grad = {var.ref():grad for var,grad in zip(model.trainable_weights, gradients)}
                local_vars = [var.ref() for layer in local_layers for var in layer.trainable_weights]
            else:
                var_grad = {var:grad for var,grad in zip(model.trainable_weights, gradients)}
                local_vars = [var for layer in local_layers for var in layer.trainable_weights]

            local_rank = hvd.local_rank()
            if tf.test.is_gpu_available(cuda_only=True):
                with tf.device("/gpu:%d" % local_rank):
                    tape = hvd.PartialDistributedGradientTape(tape, local_layers=local_layers)
                    allreduced_gradients = tape.gradient(l, model.trainable_weights)
                    local_vars_grads, global_vars_grads = tape.get_local_and_global_gradients(l, model.trainable_weights)
            else:
                tape = hvd.PartialDistributedGradientTape(tape, local_layers=local_layers)
                allreduced_gradients = tape.gradient(l, model.trainable_weights)
                local_vars_grads, global_vars_grads = tape.get_local_and_global_gradients(l, model.trainable_weights)

            for var,grad in zip(model.trainable_weights, allreduced_gradients):
                if _IS_TF2:
                    if var.ref() in local_vars:
                        # scale the gradients of local variable by size
                        avg_local_grad = var_grad[var.ref()]/hvd.size()
                        # local gradients should not change.
                        self.assertAllClose(grad, avg_local_grad)
                    else:
                        # non-local gradients shouldn't be equal given that the initial weights are set to ranks
                        self.assertNotAllClose(grad, var_grad[var.ref()])
                else:
                    if var in local_vars:
                        # scale the gradients of local variable by size
                        avg_local_grad = var_grad[var]/hvd.size()
                        # local gradients should not change.
                        self.assertAllClose(grad, avg_local_grad)
                    else:
                        # non-local gradients shouldn't be equal given that the initial weights are set to ranks
                        self.assertNotAllClose(grad, var_grad[var])

            for var, grad in local_vars_grads:
                if _IS_TF2:
                    # scale the gradients of local variable by size
                    avg_local_grad = var_grad[var.ref()]/hvd.size()
                    # local gradient from both decoupled_gradient() and gradient() calls.
                    self.assertAllClose(grad, avg_local_grad)
                else:
                    avg_local_grad = var_grad[var]/hvd.size()
                    # local gradient from both decoupled_gradient() and gradient() calls.
                    self.assertAllClose(grad, avg_local_grad)

            for var, grad in global_vars_grads:
                if _IS_TF2:
                    # non-local gradients from both decoupled_gradient() and gradient() calls.
                    self.assertNotAllClose(grad, var_grad[var.ref()])
                else:
                    self.assertNotAllClose(grad, var_grad[var])

    def test_model_parallel_model(self):
        class DummyMPModel2Devices(tf.keras.Model):
            def __init__(self):
                # For demonstation purpose, only supports 2 way parallel
                super().__init__()
                if hvd.rank() == 0:
                    self.embedding = tf.keras.layers.Embedding(7, 3, embeddings_initializer=tf.keras.initializers.Constant(1.))
                else:
                    self.embedding = tf.keras.layers.Embedding(9, 3, embeddings_initializer=tf.keras.initializers.Constant(2.))
                self.concat = tf.keras.layers.Concatenate()
                self.dense = tf.keras.layers.Dense(
                    1, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(1.))


            def call(self, inputs):
                x = self.embedding(inputs)
                y = hvd.alltoall(x)
                out = self.dense(tf.reshape(y, [1, 6]))
                return out

        tf.config.set_soft_device_placement(True)

        hvd.init()
        if hvd.size() != 2:
            self.skipTest("Test requires 2 workers.")

        rank = hvd.rank()
        local_rank = hvd.local_rank()

        if tf.test.is_gpu_available(cuda_only=True):
            with tf.device("/gpu:%d" % local_rank):
                mp_model = DummyMPModel2Devices()
                optimizer = tf.keras.optimizers.SGD(learning_rate=1.)
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                labels = tf.constant(1., shape=[1, 1])

                if rank == 0:
                    dp_inputs = tf.constant([rank + 1, rank], dtype=tf.int64)
                else:
                    dp_inputs = tf.constant([rank + 2, rank + 3], dtype=tf.int64)
        else:
            mp_model = DummyMPModel2Devices()
            optimizer = tf.keras.optimizers.SGD(learning_rate=1.)
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            labels = tf.constant(1., shape=[1, 1])

            if rank == 0:
                dp_inputs = tf.constant([rank + 1, rank], dtype=tf.int64)
            else:
                dp_inputs = tf.constant([rank + 2, rank + 3], dtype=tf.int64)

        @tf.function
        def mp_train_step(inputs):
            with tf.GradientTape() as tape:
                predictions = mp_model(inputs)
                loss = tf.math.reduce_mean(bce(labels, predictions))
            tape = hvd.DistributedGradientTape(tape)
            tape.register_local_source(mp_model.embedding.weights[0])
            gradients = tape.gradient(loss, mp_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, mp_model.trainable_variables))
            return loss

        if tf.test.is_gpu_available(cuda_only=True):
            with tf.device("/gpu:%d" % local_rank):
                # "Transpose" input from data parallel to model parallel
                mp_inputs = hvd.alltoall(dp_inputs)
                mp_loss = mp_train_step(mp_inputs)
        else:
            # "Transpose" input from data parallel to model parallel
            mp_inputs = hvd.alltoall(dp_inputs)
            mp_loss = mp_train_step(mp_inputs)


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowTests)

if __name__ == '__main__':
    tf.test.main()
