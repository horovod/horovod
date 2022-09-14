# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for horovod.tensorflow.xla_mpi_ops."""

import os
import sys
import pytest
import math
import numpy as np
import itertools
from packaging import version
import warnings

# Enable HVD XLA ops so that tf.function(jit_compile=True) works. This
# environment variable needs to be set up before loading Tensorflow, because
# it is needed to tell XLA to register the ops through C++ static
# initialization.
os.environ["HOROVOD_ENABLE_XLA_OPS"] = "1"

import tensorflow as tf
from horovod.tensorflow.util import _executing_eagerly
from tensorflow.python.framework import ops
import horovod.tensorflow as hvd
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))


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
                           tf.int32, tf.int64, tf.float32])

_IS_TF26 = version.parse(tf.__version__) >= version.parse('2.6.0')


@pytest.mark.skipif(not _IS_TF26, reason='TF2.6+ is required')
class XLATests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """

    def __init__(self, *args, **kwargs):
        super(XLATests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def evaluate(self, tensors):
        if _executing_eagerly():
            return self._eval_helper(tensors)
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)

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

    def test_horovod_allreduce_gpu(self):
        """Test that the allreduce works on XLA/GPUs."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        def hvd_allreduce_test(self, dtype, dim):
            tensor = self.random_uniform([17] * dim, -100, 100)
            tensor = tf.cast(tensor, dtype=dtype)
            summed = hvd.allreduce(tensor, average=False)
            multiplied = tensor * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))
            return max_difference

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float32, tf.float16, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                max_difference = tf.function(
                    hvd_allreduce_test, jit_compile=True)(self, dtype, dim)

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                self.skipTest(
                    "Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(
                diff <= threshold,
                "hvd.allreduce on XLA/GPU produces incorrect results")

    def test_horovod_allreduce_gpu_prescale(self):
        """Test on XLA/GPU that the allreduce correctly sums 1D, 2D, 3D tensors
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

        def hvd_allreduce_test(self, dtype, dim):
            np.random.seed(1234)
            factor = np.random.uniform()
            tensor = self.random_uniform([17] * dim, -100, 100)
            tensor = tf.cast(tensor, dtype=dtype)
            summed = hvd.allreduce(tensor, average=False,
                                   prescale_factor=factor)

            # Scaling done in FP64 math for integer types.
            tensor = tf.cast(
                tensor, tf.float64 if dtype in int_types else dtype)
            factor = tf.convert_to_tensor(
                factor, tf.float64 if dtype in int_types else dtype)
            multiplied = tf.cast(factor * tensor, dtype) * size
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))
            return max_difference

        dtypes = self.filter_supported_types(
            [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                max_difference = tf.function(
                    hvd_allreduce_test, jit_compile=True)(self, dtype, dim)

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
        """Test on XLA/GPU that the allreduce correctly sums 1D, 2D, 3D tensors
           with postscaling"""

        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_ALLREDUCE.
            return

        hvd.init()
        size = hvd.size()

        def hvd_allreduce_test(self, dtype, dim):
            np.random.seed(1234)
            factor = np.random.uniform()
            tensor = self.random_uniform([17] * dim, -100, 100)
            tensor = tf.cast(tensor, dtype=dtype)
            summed = hvd.allreduce(tensor, average=False,
                                   postscale_factor=factor)

            multiplied = tensor * size
            # Scaling done in FP64 math for integer types.
            multiplied = tf.cast(multiplied,
                                 tf.float64 if dtype in int_types else dtype)
            factor = tf.convert_to_tensor(
                factor, tf.float64 if dtype in int_types else dtype)
            multiplied = tf.cast(factor * multiplied, dtype)
            difference = summed - multiplied
            difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
            max_difference = tf.reduce_max(tf.abs(difference))
            return max_difference

        local_rank = hvd.local_rank()
        dtypes = self.filter_supported_types(
            [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32])
        int_types = [tf.uint8, tf.int8, tf.int32, tf.int64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                max_difference = tf.function(
                    hvd_allreduce_test, jit_compile=True)(self, dtype, dim)

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

    def test_horovod_allreduce_grad_gpu(self):
        """Test the correctness of the allreduce gradient on XLA/GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        def allreduce_grad_test(self, dtype, dim):
            tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
            summed = hvd.allreduce(tensor, average=False)

            grad_ys = tf.ones([5] * dim)
            grad = tf.gradients(summed, tensor, grad_ys)[0]
            return grad

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                grad = tf.function(allreduce_grad_test,
                                   jit_compile=True)(self, dtype, dim)
                grad_out = self.evaluate(grad)
            expected = np.ones([5] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allreduce_average_grad_gpu(self):
        """Test the correctness of the allreduce with average gradient on XLA/GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest(("No GPUs available"))

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        def allreduce_grad_test(self, dtype, dim):
            tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
            averaged = hvd.allreduce(tensor, average=True)

            grad_ys = tf.ones([5] * dim, dtype=dtype)
            grad = tf.gradients(averaged, tensor, grad_ys)[0]
            return grad

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                grad = tf.function(allreduce_grad_test,
                                   jit_compile=True)(self, dtype, dim)
                grad_out = self.evaluate(grad)
            expected = np.ones([5] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))


run_all_in_graph_and_eager_modes(XLATests)

if __name__ == '__main__':
    tf.test.main()
