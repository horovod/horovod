# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Uber Technologies, Inc.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.eager import context


import horovod.tensorflow as hvd

from common import mpi_env_rank_and_size


class MPITests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """

    def __init__(self, *args, **kwargs):
        super(MPITests, self).__init__(*args, **kwargs)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.tfe = tf.contrib.eager
        eager_context = context.context()
        eager_context.config = self.config
        self.eager_mode = eager_context._mode(context.EAGER_MODE)

    def test_horovod_rank(self):
        """Test that the rank returned by hvd.rank() is correct."""
        true_rank, _ = mpi_env_rank_and_size()
        hvd.init()
        rank = hvd.rank()
        assert true_rank == rank

    def test_horovod_size(self):
        """Test that the size returned by hvd.size() is correct."""
        _, true_size = mpi_env_rank_and_size()
        hvd.init()
        size = hvd.size()
        assert true_size == size

    def test_horovod_allreduce_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]

        def graph_and_eager(dtype, dim):
            with tf.device("/cpu:0"):
                tf.set_random_seed(1234)
                tensor = tf.random_uniform(
                    [17] * dim, -100, 100, dtype=dtype)
                summed = hvd.allreduce(tensor, average=False)
            multiplied = tensor * size
            max_difference = tf.reduce_max(tf.abs(summed - multiplied))
            diff = max_difference
            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if size <= 3 or dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif size < 10:
                threshold = 1e-4
            elif size < 15:
                threshold = 5e-4
            else:
                pass
            return max_difference, threshold

        with self.eager_mode:
            for dtype, dim in itertools.product(dtypes, dims):
                diff, threshold = graph_and_eager(dtype, dim)
                with self.subTest(msg='eager mode'):
                    self.assertTrue(diff <= threshold,
                                    "hvd.allreduce produces incorrect results")

        with self.test_session(config=self.config) as session:
            for dtype, dim in itertools.product(dtypes, dims):
                max_difference, threshold = graph_and_eager(dtype, dim)
                diff = session.run(max_difference)
                with self.subTest(msg='graph mode'):
                    self.assertTrue(diff <= threshold,
                                    "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_cpu_fused(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
        size = hvd.size()

        def graph_and_eager():
            dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
            dims = [1, 2, 3]
            tests = []
            for dtype, dim in itertools.product(dtypes, dims):
                with tf.device("/cpu:0"):
                    tf.set_random_seed(1234)
                    tensor = tf.random_uniform(
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
                    break

                test = max_difference <= threshold
                tests.append(test)
            return tests

        with self.eager_mode:
            tests = graph_and_eager()
            with self.subTest(msg='eager mode'):
                self.assertTrue(tf.reduce_all(tests),
                                "hvd.allreduce produces incorrect results")

        with self.test_session(config=self.config) as session:
            tests = graph_and_eager()
            with self.subTest(msg='eager mode'):
                self.assertTrue(session.run(tf.reduce_all(tests)),
                                "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_gpu(self):
        """Test that the allreduce works on GPUs.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        def graph_and_eager(dtype, dim):
            with tf.device("/gpu:%d" % local_rank):
                tf.set_random_seed(1234)
                tensor = tf.random_uniform(
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
                return
            return max_difference, threshold

        with self.eager_mode:
            dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
            dims = [1, 2, 3]
            for dtype, dim in itertools.product(dtypes, dims):
                diff, threshold = graph_and_eager(dtype, dim)
                with self.subTest(msg='eager mode'):
                    self.assertTrue(
                        diff <= threshold,
                        "hvd.allreduce on GPU produces incorrect results")

        with self.test_session(config=self.config) as session:
            dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
            dims = [1, 2, 3]
            for dtype, dim in itertools.product(dtypes, dims):
                max_difference, threshold = graph_and_eager(dtype, dim)
                diff = session.run(max_difference)
                with self.subTest(msg='graph mode'):
                    self.assertTrue(
                        diff <= threshold,
                        "hvd.allreduce on GPU produces incorrect results")

    def test_horovod_allreduce_gpu_fused(self):
        """Test that the allreduce works on GPUs with Tensor Fusion.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        def graph_and_eager():
            dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
            dims = [1, 2, 3]
            tests = []
            for dtype, dim in itertools.product(dtypes, dims):
                with tf.device("/gpu:%d" % local_rank):
                    tf.set_random_seed(1234)
                    tensor = tf.random_uniform(
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
                    return

                test = max_difference <= threshold
                tests.append(test)
            return tests

        with self.eager_mode:
            tests = graph_and_eager()
            with self.subTest(msg='eager mode'):
                self.assertTrue(session.run(tf.reduce_all(tests)),
                                "hvd.allreduce produces incorrect results")

        with self.test_session(config=self.config) as session:
            tests = graph_and_eager()
            with self.subTest(msg='graph mode'):
                self.assertTrue(session.run(tf.reduce_all(tests)),
                                "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_multi_gpu(self):
        """Test that the allreduce works on multiple GPUs.

        This test will crash badly if used with an MPI implementation that does
        not support GPU memory transfers directly, as it will call MPI_Send on
        a GPU data pointer."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        gpu_ids = [local_rank * 2, local_rank * 2 + 1]

        def graph_and_eager(iter, dtype, dim):
            with tf.device("/gpu:%d" % gpu_ids[(iter + local_rank) % 2]):
                tf.set_random_seed(1234)
                tensor = tf.random_uniform(
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
                return
            return max_difference, threshold

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        with self.eager_mode:
            for iter, (dtype, dim) in enumerate(
                    itertools.product(dtypes, dims), start=1):
                diff, threshold = graph_and_eager(iter, dtype, dim)
                with self.subTest(msg='eager mode'):
                    self.assertTrue(
                        diff <= threshold,
                        "hvd.allreduce on GPU produces incorrect results")

        with self.test_session(config=self.config) as session:
            for iter, (dtype, dim) in enumerate(
                    itertools.product(dtypes, dims), start=1):
                max_difference, threshold = graph_and_eager(iter, dtype, dim)
                diff = session.run(max_difference)
                with self.subTest(msg='graph mode'):
                    self.assertTrue(
                        diff <= threshold,
                        "hvd.allreduce on GPU produces incorrect results")

    def test_horovod_allreduce_error(self):
        """Test that the allreduce raises an error if different ranks try to
        send tensors of different rank or dimension."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        def graph_and_eager(mode=context.GRAPH_MODE):
            # Same rank, different dimension
            tf.set_random_seed(1234)
            dims = [17 + rank] * 3
            tensor = tf.random_uniform(dims, -1.0, 1.0)
            if mode == context.EAGER_MODE:
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    hvd.allreduce(tensor)
            else:
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.allreduce(tensor))

            # Same number of elements, different rank
            tf.set_random_seed(1234)
            if rank == 0:
                dims = [17, 23 * 57]
            else:
                dims = [17, 23, 57]
            tensor = tf.random_uniform(dims, -1.0, 1.0)
            return tensor

        with self.eager_mode:
            tensor = graph_and_eager(context.EAGER_MODE)
            with self.subTest(msg='eager mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    hvd.allreduce(tensor)

        with self.test_session(config=self.config) as session:
            tensor = graph_and_eager()
            with self.subTest(msg='graph mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.allreduce(tensor))

    def test_horovod_allreduce_type_error(self):
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

        with self.eager_mode:
            tensor = tf.ones(dims,
                             dtype=tf.int32 if rank % 2 == 0 else tf.float32)
            with self.subTest(msg='eager mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    hvd.allreduce(tensor)

        with self.test_session(config=self.config) as session:
            tensor = tf.ones(dims,
                             dtype=tf.int32 if rank % 2 == 0 else tf.float32)
            with self.subTest(msg='graph mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.allreduce(tensor))

    def test_horovod_allreduce_cpu_gpu_error(self):
        """Test that the allreduce raises an error if different ranks try to
        perform reduction on CPU and GPU."""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            return

        hvd.init()
        local_rank = hvd.local_rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        device = "/gpu:%d" % local_rank if local_rank % 2 == 0 else "/cpu:0"
        # Same rank, different dimension
        dims = [17] * 3

        with self.eager_mode:
            with tf.device(device):
                tensor = tf.ones(dims, dtype=tf.int32)
                with self.subTest(msg='eager mode'):
                    with self.assertRaises(tf.errors.FailedPreconditionError):
                        hvd.allreduce(tensor)

        with self.test_session(config=self.config) as session:
            with tf.device(device):
                tensor = tf.ones(dims, dtype=tf.int32)
                with self.subTest(msg='graph mode'):
                    with self.assertRaises(tf.errors.FailedPreconditionError):
                        session.run(hvd.allreduce(tensor))

    def test_horovod_allreduce_grad(self):
        """Test the correctness of the allreduce gradient."""
        hvd.init()
        size = hvd.size()

        def graph_and_eager(dtype, dim, mode=context.GRAPH_MODE):
            with tf.device("/cpu:0"):
                tf.set_random_seed(1234)
                if mode == context.EAGER_MODE:
                    tensor = tf.Variable(tf.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        summed = hvd.allreduce(tensor, average=False)
                else:
                    tensor = tf.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, average=False)

            grad_ys = tf.ones([5] * dim)
            if mode == context.EAGER_MODE:
                grad_out = tape.gradient(summed, tensor, grad_ys)
            else:
                grad = tf.gradients(summed, tensor, grad_ys)[0]
                grad_out = session.run(grad)
            expected = np.ones([5] * dim) * size
            err = np.linalg.norm(expected - grad_out)
            return err, grad_out, expected

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]

        with self.eager_mode:
            for dtype, dim in itertools.product(dtypes, dims):
                err, grad_out, expected = graph_and_eager(
                    dtype, dim, mode=context.EAGER_MODE)
                with self.subTest(msg='graph mode'):
                    self.assertLess(
                        err, 0.00000001, "gradient %s differs from expected %s, "
                        "error: %s" %
                        (grad_out, expected, str(err)))

        with self.test_session(config=self.config) as session:
            for dtype, dim in itertools.product(dtypes, dims):
                err, grad_out, expected = graph_and_eager(dtype, dim)
                with self.subTest(msg='graph mode'):
                    self.assertLess(
                        err, 0.00000001, "gradient %s differs from expected %s, "
                        "error: %s" %
                        (grad_out, expected, str(err)))

    def test_horovod_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]

        def graph_and_eager(dtype, dim, mode=context.GRAPH_MODE):
            tensor = tf.ones([17] * dim) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            gathered = hvd.allgather(tensor)

            if mode == context.EAGER_MODE:
                gathered_tensor = gathered
            else:
                gathered_tensor = session.run(gathered)
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
                if mode == context.EAGER_MODE:
                    with self.subTest(msg='eager mode'):
                        self.assertTrue(
                            tf.reduce_all(
                                tf.equal(
                                    tf.cast(
                                        rank_tensor,
                                        tf.int32),
                                    value)),
                            "hvd.allgather produces incorrect gathered tensor")
                else:
                    with self.subTest(msg='graph mode'):
                        self.assertTrue(
                            session.run(
                                tf.reduce_all(
                                    tf.equal(
                                        tf.cast(
                                            rank_tensor,
                                            tf.int32),
                                        value))),
                            "hvd.allgather produces incorrect gathered tensor")

        with self.eager_mode:
            for dtype, dim in itertools.product(dtypes, dims):
                graph_and_eager(dtype, dim, mode=context.EAGER_MODE)
        with self.test_session(config=self.config) as session:
            for dtype, dim in itertools.product(dtypes, dims):
                graph_and_eager(dtype, dim)

    def test_horovod_allgather_variable_size(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        if size > 35:
            return

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        # Support tests up to MPI Size of 35
        tensor_sizes = [17, 32, 81, 12, 15, 23, 22] * 5
        tensor_sizes = tensor_sizes[:size]

        def graph_and_eager(dtype, dim, mode=context.GRAPH_MODE):
            tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
            if dtype == tf.bool:
                tensor = tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            gathered = hvd.allgather(tensor)

            if mode == context.EAGER_MODE:
                gathered_tensor = gathered
            else:
                gathered_tensor = session.run(gathered)

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
                if mode == context.EAGER_MODE:
                    with self.subTest(msg='eager mode'):
                        self.assertTrue(
                            tf.reduce_all(
                                tf.equal(
                                    tf.cast(
                                        rank_tensor,
                                        tf.int32),
                                    value)),
                            "hvd.allgather produces incorrect gathered tensor")
                else:
                    with self.subTest(msg='graph mode'):
                        self.assertTrue(
                            session.run(
                                tf.reduce_all(
                                    tf.equal(
                                        tf.cast(
                                            rank_tensor,
                                            tf.int32),
                                        value))),
                            "hvd.allgather produces incorrect gathered tensor")

        with self.eager_mode:
            for dtype, dim in itertools.product(dtypes, dims):
                graph_and_eager(dtype, dim, mode=context.EAGER_MODE)

        with self.test_session(config=self.config) as session:
            for dtype, dim in itertools.product(dtypes, dims):
                graph_and_eager(dtype, dim)

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

        with self.eager_mode:
            tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
            with self.subTest(msg='eager mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    hvd.allgather(tensor)

        with self.test_session(config=self.config) as session:
            tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
            with self.subTest(msg='graph mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.allgather(tensor))

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
        dtype = tf.int32 if rank % 2 == 0 else tf.float32

        with self.eager_mode:
            tensor = tf.ones(tensor_size, dtype=dtype) * rank
            with self.subTest(msg='eager mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    hvd.allgather(tensor)

        with self.test_session(config=self.config) as session:
            tensor = tf.ones(tensor_size, dtype=dtype) * rank
            with self.subTest(msg='graph mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.allgather(tensor))

    def test_horovod_allgather_grad(self):
        """Test the correctness of the allgather gradient."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]

        def graph_and_eager(dtype, dim, mode=context.GRAPH_MODE):
            tensor_sizes = [3, 2, 7, 4, 6, 8, 10] * 5
            tensor_sizes = tensor_sizes[:size]
            if mode == context.EAGER_MODE:
                with tf.GradientTape() as tape:
                    tensor = tf.Variable(
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
                tensor = tf.ones([tensor_sizes[rank]] +
                                 [17] * (dim - 1)) * rank
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
                grad_out = session.run(grad)

            expected = np.ones(
                [tensor_sizes[rank]] + [17] * (dim - 1)
            ) * rank * size
            err = np.linalg.norm(expected - grad_out)
            return err, grad_out, expected

        with self.eager_mode:
            for dtype, dim in itertools.product(dtypes, dims):
                err, grad_out, expected = graph_and_eager(
                    dtype, dim, mode=context.EAGER_MODE)
                with self.subTest(msg='eager mode'):
                    self.assertLess(err, 0.00000001,
                                    "gradient %s differs from expected %s, "
                                    "error: %s" %
                                    (grad_out, expected, str(err)))

        with self.test_session(config=self.config) as session:
            for dtype, dim in itertools.product(dtypes, dims):
                err, grad_out, expected = graph_and_eager(dtype, dim)
                with self.subTest(msg='graph mode'):
                    self.assertLess(err, 0.00000001,
                                    "gradient %s differs from expected %s, "
                                    "error: %s" %
                                    (grad_out, expected, str(err)))

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        def graph_and_eager(dtype, dim, root_rank):
            tensor = tf.ones([17] * dim) * rank
            root_tensor = tf.ones([17] * dim) * root_rank
            if dtype == tf.bool:
                tensor = tensor % 2
                root_tensor = root_tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            root_tensor = tf.cast(root_tensor, dtype=dtype)
            broadcasted_tensor = hvd.broadcast(tensor, root_rank)
            return root_tensor, broadcasted_tensor

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(range(size))

        with self.eager_mode:
            for dtype, dim, root_rank in itertools.product(
                    dtypes, dims, root_ranks):
                root_tensor, broadcasted_tensor = graph_and_eager(
                    dtype, dim, root_rank)
                with self.subTest(msg='eager mode'):
                    self.assertTrue(
                        tf.reduce_all(
                            tf.equal(
                                tf.cast(
                                    root_tensor,
                                    tf.int32),
                                tf.cast(
                                    broadcasted_tensor,
                                    tf.int32))),
                        "hvd.broadcast produces incorrect broadcasted tensor")

        with self.test_session(config=self.config) as session:
            for dtype, dim, root_rank in itertools.product(
                    dtypes, dims, root_ranks):
                root_tensor, broadcasted_tensor = graph_and_eager(
                    dtype, dim, root_rank)
                with self.subTest(msg='graph mode'):
                    self.assertTrue(
                        session.run(
                            tf.reduce_all(
                                tf.equal(
                                    tf.cast(
                                        root_tensor,
                                        tf.int32),
                                    tf.cast(
                                        broadcasted_tensor,
                                        tf.int32)))),
                        "hvd.broadcast produces incorrect broadcasted tensor")

    def test_horovod_broadcast_error(self):
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

        with self.test_session(config=self.config) as session:
            tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
            with self.subTest(msg='graph mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.broadcast(tensor, 0))

        with self.eager_mode:
            tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
            with self.subTest(msg='eager mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.broadcast(tensor, 0))

    def test_horovod_broadcast_type_error(self):
        """Test that the broadcast returns an error if the types being broadcasted
        differ among the processes"""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        tensor_size = [17] * 3
        dtype = tf.int32 if rank % 2 == 0 else tf.float32

        with self.eager_mode:
            tensor = tf.ones(tensor_size, dtype=dtype) * rank
            with self.subTest(msg='eager mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    hvd.broadcast(tensor, 0)

        with self.test_session(config=self.config) as session:
            tensor = tf.ones(tensor_size, dtype=dtype) * rank
            with self.subTest(msg='graph mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.broadcast(tensor, 0))

    def test_horovod_broadcast_rank_error(self):
        """Test that the broadcast returns an error if different ranks
        specify different root rank."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        with self.eager_mode:
            tensor = tf.ones([17] * 3, dtype=tf.float32)
            with self.subTest(msg='eager mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    hvd.broadcast(tensor, rank)

        with self.test_session(config=self.config) as session:
            tensor = tf.ones([17] * 3, dtype=tf.float32)
            with self.subTest(msg='graph mode'):
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.broadcast(tensor, rank))

    def test_horovod_broadcast_grad(self):
        """Test the correctness of the broadcast gradient."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        root_ranks = list(range(size))

        def graph_and_eager(dtype, dim, root_rank, session=None):
            if session:
                tensor = tf.ones([5] * dim) * rank
            else:
                tensor = tf.Variable(tf.ones([5] * dim) * rank)
            if dtype == tf.bool:
                tensor = tensor % 2
            if session:
                tensor = tf.cast(tensor, dtype=dtype)
                broadcasted_tensor = hvd.broadcast(tensor, root_rank)
                grad_ys = tf.ones([5] * dim)
                grad = tf.gradients(broadcasted_tensor, tensor, grad_ys)[0]
                grad_out = session.run(grad)
            else:
                with tf.GradientTape() as tape:
                    tensor = tf.cast(tensor, dtype=dtype)
                    broadcasted_tensor = hvd.broadcast(tensor, root_rank)
                grad_out = tape.gradient(broadcasted_tensor, tensor)
            c = size if rank == root_rank else 0
            expected = np.ones([5] * dim) * c
            err = np.linalg.norm(expected - grad_out)
            return err, grad_out, expected

        with self.eager_mode:
            for dtype, dim, root_rank in itertools.product(
                    dtypes, dims, root_ranks):
                err, grad_out, expected = graph_and_eager(
                    dtype, dim, root_rank)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

        with self.test_session(config=self.config) as session:
            for dtype, dim, root_rank in itertools.product(
                    dtypes, dims, root_ranks):
                err, grad_out, expected = graph_and_eager(
                    dtype, dim, root_rank, session)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_compression_fp16(self):
        valid_dtypes = [tf.float16, tf.float32, tf.float64]
        invalid_dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                          tf.int32, tf.int64, tf.bool]

        tensor_size = [17] * 3
        compression = hvd.Compression.fp16

        def graph_and_eager(session=None):
            for dtype in valid_dtypes:
                tensor = tf.ones(tensor_size, dtype=dtype)

                tensor_compressed, ctx = compression.compress(tensor)
                self.assertEqual(tensor_compressed.dtype, tf.float16)

                tensor_decompressed = compression.decompress(
                    tensor_compressed, ctx)
                self.assertEqual(tensor_decompressed.dtype, dtype)

                if session:
                    actual = session.run(tensor_decompressed)
                else:
                    actual = tensor_decompressed
                expected = np.ones(tensor_size)
                err = np.linalg.norm(expected - actual)
                self.assertLess(err, 0.00000001)

            for dtype in invalid_dtypes:
                if not session and dtype is tf.bool:
                    return
                tensor = tf.ones(tensor_size, dtype=dtype)

                tensor_compressed, ctx = compression.compress(tensor)
                self.assertEqual(tensor_compressed.dtype, dtype)

                tensor_decompressed = compression.decompress(
                    tensor_compressed, ctx)
                self.assertEqual(tensor_decompressed.dtype, dtype)

                if session:
                    actual = session.run(tensor_decompressed)
                else:
                    actual = tensor_decompressed
                expected = np.ones(tensor_size)
                err = np.linalg.norm(expected - actual)
                self.assertLess(err, 0.00000001)

        with self.eager_mode:
            graph_and_eager()
        with self.test_session(config=self.config) as session:
            graph_and_eager(session)


if __name__ == '__main__':
    tf.test.main()
