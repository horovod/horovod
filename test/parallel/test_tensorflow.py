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

from distutils.version import LooseVersion

import itertools
import numpy as np
import os
import math
import pytest
import sys
import tensorflow as tf
from horovod.tensorflow.util import _executing_eagerly
from tensorflow.python.framework import ops
import warnings

import horovod.tensorflow as hvd

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import mpi_env_rank_and_size, skip_or_fail_gpu_test

if hasattr(tf, 'ConfigProto'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

if hasattr(tf, 'config') and hasattr(tf.config, 'experimental') \
        and hasattr(tf.config.experimental, 'set_memory_growth'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    # Specifies the config to use with eager execution. Does not preclude
    # tests from running in the graph mode.
    tf.enable_eager_execution(config=config)

ccl_supported_types = set([tf.uint8, tf.int8, tf.uint16, tf.int16, 
                           tf.int32, tf.int64, tf.float32, tf.float64])

_IS_TF2 = LooseVersion(tf.__version__) >= LooseVersion('2.0.0')


class TensorFlowTests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """

    def __init__(self, *args, **kwargs):
        super(TensorFlowTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        if hasattr(tf, 'contrib') and hasattr(tf.contrib, 'eager'):
            self.tfe = tf.contrib.eager
        else:
            self.tfe = tf

    def evaluate(self, tensors):
        if _executing_eagerly():
            return self._eval_helper(tensors)
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)

    def assign(self, variables, values):
        if _executing_eagerly():
            for var, val in zip(variables, values):
                var.assign(val)
        else:
            sess = ops.get_default_session()
            if sess is None:
                with self.test_session(config=config) as sess:
                    for var, val in zip(variables, values):
                        var.load(val, sess)
            else:
                for var, val in zip(variables, values):
                    var.load(val, sess)

    def random_uniform(self, *args, **kwargs):
        if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
            tf.random.set_seed(1234)
            return tf.random.uniform(*args, **kwargs)
        else:
            tf.set_random_seed(1234)
            return tf.random_uniform(*args, **kwargs)

    def filter_supported_types(self, types):
        if 'CCL_ROOT' in os.environ:
           types = [t for t in types if t in ccl_supported_types]
        return types

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
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False)
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
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
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                averaged = hvd.allreduce(tensor, average=True)
            max_difference = tf.reduce_max(tf.abs(tf.cast(averaged, dtype=dtype) - tensor))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_cpu_fused(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False)
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
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
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                np.random.seed(1234)
                factor = np.random.uniform()
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False,
                                       prescale_factor=factor)

                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                tensor = tf.cast(tensor, tf.float32 if dtype == tf.float16 else
                                 tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * tensor, dtype) * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

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
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                np.random.seed(1234)
                factor = np.random.uniform()
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False,
                                       postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types, FP32 math for FP16 on CPU
                multiplied = tf.cast(multiplied, tf.float32 if dtype == tf.float16 else
                                     tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float32 if dtype == tf.float16 else
                                              tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * multiplied, dtype)
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

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

    def test_horovod_allreduce_gpu(self):
        """Test that the allreduce works on GPUs."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False)
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
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

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                averaged = hvd.allreduce(tensor, average=True)
            max_difference = tf.reduce_max(tf.abs(tf.cast(averaged, dtype=dtype) - tensor))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce on GPU produces incorrect results")

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

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        tests = []
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False)
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
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

        # Only do this test if there are enough GPUs available.
        if len(tf.config.experimental.list_physical_devices('GPU')) < 2:
            self.skipTest(("Too few GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        iter = 0
        gpu_ids = [local_rank * 2, local_rank * 2 + 1]
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            iter += 1
            with tf.device("/gpu:%d" % gpu_ids[(iter + local_rank) % 2]):
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False)
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
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
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                np.random.seed(1234)
                factor = np.random.uniform()
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False,
                                       prescale_factor=factor)

                # Scaling done in FP64 math for integer types.
                tensor = tf.cast(tensor, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * tensor, dtype) * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

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
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                np.random.seed(1234)
                factor = np.random.uniform()
                tensor = self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False,
                                       postscale_factor=factor)

                multiplied = tensor * size
                # Scaling done in FP64 math for integer types.
                multiplied = tf.cast(multiplied, tf.float64 if dtype in int_types else dtype)
                factor = tf.convert_to_tensor(factor, tf.float64 if dtype in int_types else dtype)
                multiplied = tf.cast(factor * multiplied, dtype)
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

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
                        summed = hvd.allreduce(tensor, average=False)
                else:
                    tensor = self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, average=False)

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
                        averaged = hvd.allreduce(tensor, average=True)
                else:
                    tensor = self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)
                    averaged = hvd.allreduce(tensor, average=True)

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
                        summed = hvd.allreduce(tensor, average=False)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, average=False)

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
                        averaged = hvd.allreduce(tensor, average=True)
                else:
                    tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    averaged = hvd.allreduce(tensor, average=True)

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
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                tensors = [self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                summed = hvd.grouped_allreduce(tensors, average=False)
            multiplied = [tensor * size for tensor in tensors]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(summed, multiplied)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_gpu(self):
        """Test on GPU that the grouped allreduce correctly sums 1D, 2D, 3D tensors."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if os.environ.get('HOROVOD_MIXED_INSTALL'):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()
        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                tensors = [self.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                summed = hvd.grouped_allreduce(tensors, average=False)
            multiplied = [tensor * size for tensor in tensors]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(summed, multiplied)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce on GPU produces incorrect results")

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
                        summed = hvd.grouped_allreduce(tensors, average=False)
                else:
                    tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    summed = hvd.grouped_allreduce(tensors, average=False)

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

        if os.environ.get('HOROVOD_MIXED_INSTALL'):
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
                        summed = hvd.grouped_allreduce(tensors, average=False)
                else:
                    tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    summed = hvd.grouped_allreduce(tensors, average=False)

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
            ) * rank * size
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
            ) * rank * size
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

            c = size if rank == root_rank else 0
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

            c = size if rank == root_rank else 0
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
                collected = hvd.alltoall(tensor, splits)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

    def test_horovod_alltoall_gpu(self):
        """Test that the alltoall correctly distributes 1D, 2D, and 3D tensors on GPU."""
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
                splits = tf.convert_to_tensor([rank+1] * size, dtype=tf.int32)
                collected = hvd.alltoall(tensor, splits)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

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

                collected = hvd.alltoall(tensor, splits, name="a2a")

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1)
                                                               - (1+1) * 2 ** (dim-1)  # subtract missing rank 1 contributions
                                           )),
                    "hvd.alltoall collected wrong number of values")

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

                collected = hvd.alltoall(tensor, splits, name="a2a")

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), size * (size + 1) // 2 * 2**(dim - 1)
                                                               - (1+1) * 2 ** (dim-1)  # subtract missing rank 1 contributions
                                           )),
                    "hvd.alltoall collected wrong number of values")

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

                collected = hvd.alltoall(tensor, splits, name="a2a")
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")
                if rank == 0:
                    expected_size = 0
                else:
                    expected_size = size * (size + 1) // 2 * 2**(dim - 1)
                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), expected_size)),
                    "hvd.alltoall collected wrong number of values")

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

                collected = hvd.alltoall(tensor, splits, name="a2a")
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")
                if rank == 0:
                    expected_size = 0
                else:
                    expected_size = size * (size + 1) // 2 * 2**(dim - 1)
                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), expected_size)),
                    "hvd.alltoall collected wrong number of values")


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
            collected = hvd.alltoall(source_tensor, splits, name="alltoall_zero_splits")
            result = self.evaluate(collected)

        print(hvd.rank(), "result.shape", result.shape)
        print(hvd.rank(), "result", result)
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
            collected = hvd.alltoall(source_tensor, splits, name="alltoall_zero_splits")
            result = self.evaluate(collected)

        print(hvd.rank(), "result.shape", result.shape)
        print(hvd.rank(), "result", result)
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
                        collected = hvd.alltoall(tensor, splits)
                else:
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    collected = hvd.alltoall(tensor, splits)

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
                        collected = hvd.alltoall(tensor, splits)
                else:
                    splits = tf.convert_to_tensor([rank + 1] * size, dtype=tf.int32)
                    collected = hvd.alltoall(tensor, splits)

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
                    self.evaluate(hvd.join())
                else:		
                    summed = hvd.allreduce(tensor, average=False)
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
                    self.evaluate(hvd.join())
                    self.assertTrue(diff <= threshold,
                             "hvd.join with hvd.allreduce on GPU produces incorrect results")

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
                bn_func = bn.apply(x, training=True)
                sync_bn_func = sync_bn.apply(tf.expand_dims(x[hvd.rank()], 0), training=True)

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
                bn_func = bn.apply(x, training=True)
                sync_bn_func = sync_bn.apply(tf.expand_dims(x[hvd.rank()], 0), training=True)

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
            update_op = opt.apply_gradients(grads_and_vars)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            def compute_expected_value(batch_id):
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

            for idx in range(10):
                _ = sess.run(update_op)
                computed_value = sess.run(opt._optimizer.variable.read_value())[0]
                assert computed_value == compute_expected_value(idx)


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowTests)

if __name__ == '__main__':
    tf.test.main()
