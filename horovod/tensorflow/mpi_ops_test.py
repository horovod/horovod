# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 Uber Technologies, Inc.
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

import os.path
import itertools

import tensorflow as tf

import horovod.tensorflow as hvd


def mpi_env_rank_and_size():
    """Get MPI rank and size from environment variables and return them as a
    tuple of integers.

    Most MPI implementations have an `mpirun` or `mpiexec` command that will
    run an MPI executable and set up all communication necessary between the
    different processors. As part of that set up, they will set environment
    variables that contain the rank and size of the MPI_COMM_WORLD
    communicator. We can read those environment variables from Python in order
    to ensure that `hvd.rank()` and `hvd.size()` return the expected values.

    Since MPI is just a standard, not an implementation, implementations
    typically choose their own environment variable names. This function tries
    to support several different implementation, but really it only needs to
    support whatever implementation we want to use for the TensorFlow test
    suite.

    If this is not running under MPI, then defaults of rank zero and size one
    are returned. (This is appropriate because when you call MPI_Init in an
    application not started with mpirun, it will create a new independent
    communicator with only one process in it.)
    """
    rank_env = "PMI_RANK OMPI_COMM_WORLD_RANK".split()
    size_env = "PMI_SIZE OMPI_COMM_WORLD_SIZE".split()

    for rank_var, size_var in zip(rank_env, size_env):
        rank = os.environ.get(rank_var)
        size = os.environ.get(size_var)
        if rank is not None and size is not None:
            return int(rank), int(size)

    # Default to rank zero and size one if there are no environment variables
    return 0, 1


class MPITests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """

    def test_horovod_rank(self):
        """Test that the rank returned by hvd.rank() is correct."""
        true_rank, _ = mpi_env_rank_and_size()
        hvd.init()
        rank = hvd.rank()
        self.assertEqual(true_rank, rank)

    def test_horovod_size(self):
        """Test that the size returned by hvd.size() is correct."""
        _, true_size = mpi_env_rank_and_size()
        hvd.init()
        size = hvd.size()
        self.assertEqual(true_size, size)

    def test_horovod_allreduce_cpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        with self.test_session() as session:
            dtypes = [tf.int32, tf.int64, tf.float32, tf.float64]
            dims = [1, 2, 3]
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
                if size <= 3:
                    threshold = 0
                elif size < 10:
                    threshold = 1e-4
                elif size < 15:
                    threshold = 5e-4
                else:
                    break

                diff = session.run(max_difference)
                self.assertTrue(diff <= threshold,
                                "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_cpu_fused(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors
        with Tensor Fusion."""
        hvd.init()
        size = hvd.size()
        with self.test_session() as session:
            dtypes = [tf.int32, tf.int64, tf.float32, tf.float64]
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
                if size <= 3:
                    threshold = 0
                elif size < 10:
                    threshold = 1e-4
                elif size < 15:
                    threshold = 5e-4
                else:
                    break

                test = max_difference <= threshold
                tests.append(test)
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

        one_gpu = tf.GPUOptions(visible_device_list=str(local_rank))
        gpu_config = tf.ConfigProto(gpu_options=one_gpu)
        with self.test_session(config=gpu_config) as session:
            dtypes = [tf.int32, tf.int64, tf.float32, tf.float64]
            dims = [1, 2, 3]
            for dtype, dim in itertools.product(dtypes, dims):
                with tf.device("/gpu:0"):
                    tf.set_random_seed(1234)
                    tensor = tf.random_uniform(
                        [17] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, average=False)
                multiplied = tensor * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

                # Threshold for floating point equality depends on number of
                # ranks, since we're comparing against precise multiplication.
                if size <= 3:
                    threshold = 0
                elif size < 10:
                    threshold = 1e-4
                elif size < 15:
                    threshold = 5e-4
                else:
                    return

                diff = session.run(max_difference)
                self.assertTrue(diff <= threshold,
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

        one_gpu = tf.GPUOptions(visible_device_list=str(local_rank))
        gpu_config = tf.ConfigProto(gpu_options=one_gpu)
        with self.test_session(config=gpu_config) as session:
            dtypes = [tf.int32, tf.int64, tf.float32, tf.float64]
            dims = [1, 2, 3]
            tests = []
            for dtype, dim in itertools.product(dtypes, dims):
                with tf.device("/gpu:0"):
                    tf.set_random_seed(1234)
                    tensor = tf.random_uniform(
                        [17] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, average=False)
                multiplied = tensor * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

                # Threshold for floating point equality depends on number of
                # ranks, since we're comparing against precise multiplication.
                if size <= 3:
                    threshold = 0
                elif size < 10:
                    threshold = 1e-4
                elif size < 15:
                    threshold = 5e-4
                else:
                    return

                test = max_difference <= threshold
                tests.append(test)
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

        iter = 0
        two_gpus = tf.GPUOptions(visible_device_list=(
            '%d,%d' % (local_rank * 2, local_rank * 2 + 1)))
        gpu_config = tf.ConfigProto(gpu_options=two_gpus)
        with self.test_session(config=gpu_config) as session:
            dtypes = [tf.int32, tf.int64, tf.float32, tf.float64]
            dims = [1, 2, 3]
            for dtype, dim in itertools.product(dtypes, dims):
                iter += 1
                with tf.device("/gpu:%d" % ((iter + local_rank) % 2)):
                    tf.set_random_seed(1234)
                    tensor = tf.random_uniform(
                        [17] * dim, -100, 100, dtype=dtype)
                    summed = hvd.allreduce(tensor, average=False)
                multiplied = tensor * size
                max_difference = tf.reduce_max(tf.abs(summed - multiplied))

                # Threshold for floating point equality depends on number of
                # ranks, since we're comparing against precise multiplication.
                if size <= 3:
                    threshold = 0
                elif size < 10:
                    threshold = 1e-4
                elif size < 15:
                    threshold = 5e-4
                else:
                    return

                diff = session.run(max_difference)
                self.assertTrue(diff <= threshold,
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

        with self.test_session() as session:
            # Same rank, different dimension
            tf.set_random_seed(1234)
            dims = [17 + rank] * 3
            tensor = tf.random_uniform(dims, -1.0, 1.0)
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(hvd.allreduce(tensor))

            # Same number of elements, different rank
            tf.set_random_seed(1234)
            if rank == 0:
                dims = [17, 23 * 57]
            else:
                dims = [17, 23, 57]
            tensor = tf.random_uniform(dims, -1.0, 1.0)
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

        with self.test_session() as session:
            # Same rank, different dimension
            dims = [17] * 3
            tensor = tf.ones(dims,
                             dtype=tf.int32 if rank % 2 == 0 else tf.float32)
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

        device = "/gpu:0" if local_rank % 2 == 0 else "/cpu:0"
        one_gpu = tf.GPUOptions(visible_device_list=str(local_rank))
        gpu_config = tf.ConfigProto(gpu_options=one_gpu)
        with self.test_session(config=gpu_config) as session:
            with tf.device(device):
                # Same rank, different dimension
                dims = [17] * 3
                tensor = tf.ones(dims, dtype=tf.int32)
                with self.assertRaises(tf.errors.FailedPreconditionError):
                    session.run(hvd.allreduce(tensor))

    def test_horovod_allgather(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        with self.test_session() as session:
            dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                      tf.int32, tf.int64, tf.float32, tf.float64,
                      tf.bool]
            dims = [1, 2, 3]
            for dtype, dim in itertools.product(dtypes, dims):
                tensor = tf.ones([17] * dim) * rank
                if dtype == tf.bool:
                    tensor = tensor % 2
                tensor = tf.cast(tensor, dtype=dtype)
                gathered = hvd.allgather(tensor)

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
                    self.assertTrue(
                        session.run(tf.reduce_all(
                            tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                        "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_variable_size(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors,
        even if those tensors have different sizes along the first dim."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        with self.test_session() as session:
            dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                      tf.int32, tf.int64, tf.float32, tf.float64,
                      tf.bool]
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
                gathered = hvd.allgather(tensor)

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
                    self.assertTrue(
                        session.run(tf.reduce_all(
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
            return

        with self.test_session() as session:
            tensor_size = [17] * 3
            tensor_size[1] = 10 * (rank + 1)
            tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
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

        with self.test_session() as session:
            tensor_size = [17] * 3
            dtype = tf.int32 if rank % 2 == 0 else tf.float32
            tensor = tf.ones(tensor_size, dtype=dtype) * rank
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(hvd.allgather(tensor))

    def test_horovod_broadcast(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        with self.test_session() as session:
            dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                      tf.int32, tf.int64, tf.float32, tf.float64,
                      tf.bool]
            dims = [1, 2, 3]
            root_ranks = list(range(size))
            for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
                try:
                    tensor = tf.ones([17] * dim) * rank
                    root_tensor = tf.ones([17] * dim) * root_rank
                    if dtype == tf.bool:
                        tensor = tensor % 2
                        root_tensor = root_tensor % 2
                    tensor = tf.cast(tensor, dtype=dtype)
                    root_tensor = tf.cast(root_tensor, dtype=dtype)
                    broadcasted_tensor = hvd.broadcast(tensor, root_rank)
                    self.assertTrue(
                        session.run(tf.reduce_all(tf.equal(
                            tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                        "hvd.broadcast produces incorrect broadcasted tensor")
                except Exception:
                    import traceback
                    traceback.print_exc()

    def test_horovod_broadcast_error(self):
        """Test that the broadcast returns an error if any dimension besides
        the first is different among the tensors being broadcasted."""
        hvd.init()
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            return

        with self.test_session() as session:
            tensor_size = [17] * 3
            tensor_size[1] = 10 * (rank + 1)
            tensor = tf.ones(tensor_size, dtype=tf.float32) * rank
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

        with self.test_session() as session:
            tensor_size = [17] * 3
            dtype = tf.int32 if rank % 2 == 0 else tf.float32
            tensor = tf.ones(tensor_size, dtype=dtype) * rank
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

        with self.test_session() as session:
            tensor = tf.ones([17] * 3, dtype=tf.float32)
            with self.assertRaises(tf.errors.FailedPreconditionError):
                session.run(hvd.broadcast(tensor, rank))


if __name__ == '__main__':
    tf.test.main()
