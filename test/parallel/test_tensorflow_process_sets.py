"""Tests for horovod.tensorflow.mpi_ops using multiple process sets.

With TensorFlow 2.9 and MPI the option HOROVOD_DYNAMIC_PROCESS_SETS has been observed to cause significant
slowdowns in all Horovod operations, especially on GPU-equipped AWS instances. For that reason we collect
tests for multiple process sets in this script that initializes Horovod with static process sets.
"""

from packaging import version

import itertools
import numpy as np
import platform
import tensorflow as tf
from horovod.tensorflow.util import _executing_eagerly

import horovod.tensorflow as hvd

from base_test_tensorflow import *

from horovod.runner.common.util.env import get_env_rank_and_size

_IS_TF2 = version.parse(tf.__version__) >= version.parse('2.0.0')
_is_mac = platform.system() == 'Darwin'


class TensorFlowProcessSetsTests(BaseTensorFlowTests):
    """
    Tests for ops in horovod.tensorflow using multiple process sets.
    """
    def __init__(self, *args, **kwargs):
        super(TensorFlowProcessSetsTests, self).__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        """Initializes Horovod with two process sets"""
        _, size = get_env_rank_and_size()

        cls.even_ranks = [rk for rk in range(0, size) if rk % 2 == 0]
        cls.odd_ranks = [rk for rk in range(0, size) if rk % 2 == 1]
        cls.even_set = hvd.ProcessSet(cls.even_ranks)
        cls.odd_set = hvd.ProcessSet(cls.odd_ranks)

        hvd.init(process_sets=[cls.even_set, cls.odd_set])

    def tearDown(self):
        """Prevent that one process shuts down Horovod too early"""
        with tf.device("/cpu:0"):
            b = hvd.allreduce(tf.constant([0.]), name="global_barrier_after_test")
            _ = self.evaluate(b)

    def test_horovod_size_op_process_set(self):
        """Test that the size returned by hvd.size_op(process_set_id) is correct."""
        # This test does not apply if there is only one worker.
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        size = self.evaluate(hvd.size_op(process_set_id=self.even_set.process_set_id))
        self.assertEqual(size, self.even_set.size(),
                        "hvd.size_op produces incorrect results for a process set")

    def test_horovod_process_set_included_op(self):
        """Test that the result of hvd.process_set_included_op(process_set_id) is correct."""
        # This test does not apply if there is only one worker.
        if hvd.size() == 1:
            self.skipTest("Only one worker available")

        included = self.evaluate(hvd.process_set_included_op(process_set_id=self.even_set.process_set_id))

        if hvd.rank() in self.even_ranks:
            self.assertEqual(included, 1)
        else:
            self.assertEqual(included, 0)

    def test_horovod_allreduce_cpu_process_sets(self):
        """ Test on CPU that allreduce correctly sums if restricted to non-global process sets"""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                even_rank_tensor = self.random_uniform([17] * dim, -100, 100)
                even_rank_tensor = tf.cast(even_rank_tensor, dtype=dtype)
                odd_rank_tensor = self.random_uniform([17] * dim, -100, 100)
                odd_rank_tensor = tf.cast(odd_rank_tensor, dtype=dtype)
                if rank in self.even_ranks:
                    summed = hvd.allreduce(even_rank_tensor, average=False, process_set=self.even_set)
                    multiplied = even_rank_tensor * len(self.even_ranks)
                if rank in self.odd_ranks:
                    summed = hvd.allreduce(odd_rank_tensor, average=False, process_set=self.odd_set)
                    multiplied = odd_rank_tensor * len(self.odd_ranks)
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            max_process_set_size = max(len(self.even_ranks), len(self.odd_ranks))
            if max_process_set_size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif max_process_set_size < 10:
                threshold = 1e-4
            elif max_process_set_size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_gpu_process_sets(self):
        """ Test on GPU that allreduce correctly sums if restricted to non-global process sets"""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        local_rank = hvd.local_rank()
        rank = hvd.rank()

        dtypes = [tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                even_rank_tensor = self.random_uniform([17] * dim, -100, 100)
                even_rank_tensor = tf.cast(even_rank_tensor, dtype=dtype)
                odd_rank_tensor = self.random_uniform([17] * dim, -100, 100)
                odd_rank_tensor = tf.cast(odd_rank_tensor, dtype=dtype)
                if rank in self.even_ranks:
                    summed = hvd.allreduce(even_rank_tensor, average=False, process_set=self.even_set)
                    multiplied = even_rank_tensor * len(self.even_ranks)
                if rank in self.odd_ranks:
                    summed = hvd.allreduce(odd_rank_tensor, average=False, process_set=self.odd_set)
                    multiplied = odd_rank_tensor * len(self.odd_ranks)
                difference = summed - multiplied
                difference = tf.cast(difference, tf.int32) if dtype == tf.uint8 else difference
                max_difference = tf.reduce_max(tf.abs(difference))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            max_process_set_size = max(len(self.even_ranks), len(self.odd_ranks))
            if max_process_set_size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif max_process_set_size < 10:
                threshold = 1e-4
            elif max_process_set_size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.allreduce produces incorrect results")

    def test_horovod_allreduce_process_set_id_error(self):
        """Test that allreduce raises an error if an invalid process set id
        is specified."""
        rank = hvd.rank()
        size = hvd.size()

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        with tf.device("/cpu:0"):
            tensor = tf.ones(4)
            if rank in self.even_ranks:
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    self.evaluate(hvd.allreduce(tensor, process_set=self.odd_set))
            else:
                with self.assertRaises(tf.errors.InvalidArgumentError):
                    self.evaluate(hvd.allreduce(tensor, process_set=self.even_set))
            with self.assertRaises(ValueError):
                fake_set = hvd.ProcessSet([0])
                fake_set.process_set_id = 10  # you should not do this
                self.evaluate(hvd.allreduce(tensor, process_set=fake_set))

    def test_horovod_allreduce_grad_cpu_process_sets(self):
        """Test the correctness of the allreduce gradient on CPU if restricted to non-global process sets."""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    self.even_rank_tensor = self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype))
                    odd_rank_tensor = self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        if rank in self.even_ranks:
                            summed = hvd.allreduce(self.even_rank_tensor, average=False,
                                                   process_set=self.even_set)
                        elif rank in self.odd_ranks:
                            summed = hvd.allreduce(odd_rank_tensor, average=False,
                                                   process_set=self.odd_set)
                else:
                    self.even_rank_tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    odd_rank_tensor = self.random_uniform([5] * dim, -100, 100, dtype=dtype)
                    if rank in self.even_ranks:
                        summed = hvd.allreduce(self.even_rank_tensor, average=False,
                                               process_set=self.even_set)
                    elif rank in self.odd_ranks:
                        summed = hvd.allreduce(odd_rank_tensor, average=False,
                                               process_set=self.odd_set)

                if rank in self.even_ranks:
                    tensor = self.even_rank_tensor
                    set_size = len(self.even_ranks)
                elif rank in self.odd_ranks:
                    tensor = odd_rank_tensor
                    set_size = len(self.odd_ranks)

                grad_ys = tf.ones([5] * dim)
                if _executing_eagerly():
                    grad_out = tape.gradient(summed, tensor, grad_ys)
                else:
                    grad = tf.gradients(summed, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            expected = np.ones([5] * dim) * set_size
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_allreduce_cpu_process_sets(self):
        """Test on CPU that the grouped allreduce correctly sums if restricted to non-global process sets"""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                even_rank_tensors = [tf.cast(self.random_uniform(
                    [17] * dim, -100, 100), dtype=dtype) for _ in range(5)]
                odd_rank_tensors = [tf.cast(self.random_uniform(
                    [17] * dim, -100, 100), dtype=dtype) for _ in range(5)]
                if rank in self.even_ranks:
                    summed = hvd.grouped_allreduce(even_rank_tensors, average=False, process_set=self.even_set)
                    multiplied = [tensor * len(self.even_ranks) for tensor in even_rank_tensors]
                elif rank in self.odd_ranks:
                    summed = hvd.grouped_allreduce(odd_rank_tensors, average=False, process_set=self.odd_set)
                    multiplied = [tensor * len(self.odd_ranks) for tensor in odd_rank_tensors]
            differences = [t1 - t2 for t1, t2 in zip(summed, multiplied)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            max_process_set_size = max(len(self.even_ranks), len(self.odd_ranks))
            if max_process_set_size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif max_process_set_size < 10:
                threshold = 1e-4
            elif max_process_set_size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_gpu_process_sets(self):
        """Test on GPU that the grouped allreduce correctly sums if restricted to non-global process sets"""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")
        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")
        rank = hvd.rank()
        local_rank = hvd.local_rank()

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                even_rank_tensors = [tf.cast(self.random_uniform(
                    [17] * dim, -100, 100), dtype=dtype) for _ in range(5)]
                odd_rank_tensors = [tf.cast(self.random_uniform(
                    [17] * dim, -100, 100), dtype=dtype) for _ in range(5)]
                if rank in self.even_ranks:
                    summed = hvd.grouped_allreduce(even_rank_tensors, average=False, process_set=self.even_set)
                    multiplied = [tensor * len(self.even_ranks) for tensor in even_rank_tensors]
                elif rank in self.odd_ranks:
                    summed = hvd.grouped_allreduce(odd_rank_tensors, average=False, process_set=self.odd_set)
                    multiplied = [tensor * len(self.odd_ranks) for tensor in odd_rank_tensors]
            differences = [t1 - t2 for t1, t2 in zip(summed, multiplied)]
            differences = [tf.cast(diff, tf.int32) if dtype == tf.uint8 else diff for diff in differences]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(diff)) for diff in differences])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            max_process_set_size = max(len(self.even_ranks), len(self.odd_ranks))
            if max_process_set_size <= 3 or dtype in [tf.uint8, tf.int8, tf.int32, tf.int64]:
                threshold = 0
            elif max_process_set_size < 10:
                threshold = 1e-4
            elif max_process_set_size < 15:
                threshold = 5e-4
            else:
                self.skipTest("Horovod cluster too large for precise multiplication comparison")

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold, "hvd.grouped_allreduce produces incorrect results")

    def test_horovod_grouped_allreduce_grad_cpu_process_sets(self):
        """Test the correctness of the grouped allreduce gradient on CPU
        if restricted to non-global process sets."""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    even_rank_tensors = [self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)) for _ in range(5)]
                    odd_rank_tensors = [self.tfe.Variable(self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype)) for _ in range(5)]
                    with tf.GradientTape(persistent=True) as tape:
                        if rank in self.even_ranks:
                            summed = hvd.grouped_allreduce(even_rank_tensors, average=False,
                                                           process_set=self.even_set)
                        elif rank in self.odd_ranks:
                            summed = hvd.grouped_allreduce(odd_rank_tensors, average=False,
                                                           process_set=self.odd_set)
                else:
                    even_rank_tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    odd_rank_tensors = [self.random_uniform(
                        [5] * dim, -100, 100, dtype=dtype) for _ in range(5)]
                    if rank in self.even_ranks:
                        summed = hvd.grouped_allreduce(even_rank_tensors, average=False,
                                                       process_set=self.even_set)
                    elif rank in self.odd_ranks:
                        summed = hvd.grouped_allreduce(odd_rank_tensors, average=False,
                                                       process_set=self.odd_set)

                if rank in self.even_ranks:
                    tensors = even_rank_tensors
                    set_size = len(self.even_ranks)
                elif rank in self.odd_ranks:
                    tensors = odd_rank_tensors
                    set_size = len(self.odd_ranks)

                grads_ys = [tf.ones([5] * dim, dtype=dtype) for _ in range(5)]
                if _executing_eagerly():
                    grads_out = [tape.gradient(s, t, g) for s, t, g in zip(summed, tensors, grads_ys)]
                else:
                    grads = [tf.gradients(s, t, g)[0] for s, t, g in zip(summed, tensors, grads_ys)]
                    grads_out = [self.evaluate(grad) for grad in grads]

            expected = np.ones([5] * dim) * set_size
            for grad_out in grads_out:
                err = np.linalg.norm(expected - grad_out)
                self.assertLess(err, 0.00000001,
                                "gradient %s differs from expected %s, "
                                "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_allgather_cpu_process_sets(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors if restricted to non-global process sets."""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        if rank in self.even_ranks:
            set_size = len(self.even_ranks)
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_size = len(self.odd_ranks)
            set_ranks = self.odd_ranks
            this_set = self.odd_set

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
                gathered = hvd.allgather(tensor, process_set=this_set)

            gathered_tensor = self.evaluate(gathered)
            self.assertEqual(list(gathered_tensor.shape),
                             [17 * set_size] + [17] * (dim - 1))

            for i in range(set_size):
                rank_tensor = tf.slice(gathered_tensor,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                self.assertEqual(list(rank_tensor.shape), [17] * dim)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = set_ranks[i]
                else:
                    value = set_ranks[i] % 2
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                    "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_gpu_process_sets(self):
        """Test that the allgather correctly gathers 1D, 2D, 3D tensors if restricted to non-global process sets."""

        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        rank = hvd.rank()
        local_rank = hvd.local_rank()

        if rank in self.even_ranks:
            set_size = len(self.even_ranks)
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_size = len(self.odd_ranks)
            set_ranks = self.odd_ranks
            this_set = self.odd_set

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
                gathered = hvd.allgather(tensor, process_set=this_set)

            gathered_tensor = self.evaluate(gathered)
            self.assertEqual(list(gathered_tensor.shape),
                             [17 * set_size] + [17] * (dim - 1))

            for i in range(set_size):
                rank_tensor = tf.slice(gathered_tensor,
                                       [i * 17] + [0] * (dim - 1),
                                       [17] + [-1] * (dim - 1))
                self.assertEqual(list(rank_tensor.shape), [17] * dim)
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = set_ranks[i]
                else:
                    value = set_ranks[i] % 2
                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(rank_tensor, tf.int32), value))),
                    "hvd.allgather produces incorrect gathered tensor")

    def test_horovod_allgather_grad_cpu_process_sets(self):
        """Test the correctness of the allgather gradient on CPU if restricted to non-global process sets."""
        rank = hvd.rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            tensor_sizes = [3, 2, 7, 4, 6, 8, 10] * 5
            tensor_sizes = tensor_sizes[:size]
            set_tensor_sizes = [tensor_sizes[rk] for rk in set_ranks]

            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    with tf.GradientTape() as tape:
                        tensor = self.tfe.Variable(
                            tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank)
                        if dtype == tf.bool:
                            tensor = tensor % 2
                        tensor = tf.cast(tensor, dtype=dtype)
                        gathered = hvd.allgather(tensor, process_set=this_set)
                        grad_list = []
                        for r, tensor_size in zip(set_ranks, set_tensor_sizes):
                            g = tf.ones([tensor_size] + [17] * (dim - 1)) * r
                            grad_list.append(g)
                        grad_ys = tf.concat(grad_list, axis=0)
                    grad_out = tape.gradient(gathered, tensor, grad_ys)
                else:
                    tensor = tf.ones([tensor_sizes[rank]] + [17] * (dim - 1)) * rank
                    if dtype == tf.bool:
                        tensor = tensor % 2
                    tensor = tf.cast(tensor, dtype=dtype)
                    gathered = hvd.allgather(tensor, process_set=this_set)

                    grad_list = []
                    for r, tensor_size in zip(set_ranks, set_tensor_sizes):
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

    def test_horovod_broadcast_cpu_process_sets(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on CPU
         if restricted to non-global process sets"""
        rank = hvd.rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(set_ranks)
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            with tf.device("/cpu:0"):
                tensor = tf.ones([17] * dim) * rank
                root_tensor = tf.ones([17] * dim) * root_rank
                if dtype == tf.bool:
                    tensor = tensor % 2
                    root_tensor = root_tensor % 2
                tensor = tf.cast(tensor, dtype=dtype)
                root_tensor = tf.cast(root_tensor, dtype=dtype)
                broadcasted_tensor = hvd.broadcast(tensor, root_rank, process_set=this_set)
            self.assertTrue(
                self.evaluate(tf.reduce_all(tf.equal(
                    tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                "hvd.broadcast produces incorrect broadcasted tensor")

    def test_horovod_broadcast_gpu_process_sets(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors on GPU
         if restricted to non-global process sets"""
        # Only do this test if there are GPUs available.
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        rank = hvd.rank()
        local_rank = hvd.local_rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64, tf.bool]
        dims = [1, 2, 3]
        root_ranks = list(set_ranks)
        for dtype, dim, root_rank in itertools.product(dtypes, dims, root_ranks):
            tensor = tf.ones([17] * dim) * rank
            root_tensor = tf.ones([17] * dim) * root_rank
            if dtype == tf.bool:
                tensor = tensor % 2
                root_tensor = root_tensor % 2
            tensor = tf.cast(tensor, dtype=dtype)
            root_tensor = tf.cast(root_tensor, dtype=dtype)
            with tf.device("/gpu:%d" % local_rank):
                broadcasted_tensor = hvd.broadcast(tensor, root_rank, process_set=this_set)
            self.assertTrue(
                self.evaluate(tf.reduce_all(tf.equal(
                    tf.cast(root_tensor, tf.int32), tf.cast(broadcasted_tensor, tf.int32)))),
                "hvd.broadcast produces incorrect broadcasted tensor")

    def test_broadcast_variables_process_sets(self):
        rank = hvd.rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set
        root_rank = set_ranks[0]

        with tf.device("/cpu:0"):
            var = tf.Variable(initial_value=[rank], dtype=tf.int32)
            if not hvd._executing_eagerly():
                init = tf.compat.v1.global_variables_initializer()
                self.evaluate(init)
            self.evaluate(
                hvd.broadcast_variables([var], root_rank=root_rank, process_set=this_set))
            value = self.evaluate(var)
        self.assertListEqual(list(value), [root_rank])

    def test_horovod_broadcast_grad_cpu_process_sets(self):
        """Test the correctness of the broadcast gradient on CPU if restricted to non-global process sets."""
        rank = hvd.rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        root_ranks = list(set_ranks)
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
                        broadcasted_tensor = hvd.broadcast(tensor, root_rank,
                                                           process_set=this_set)
                    grad_out = tape.gradient(broadcasted_tensor, tensor)
                else:
                    tensor = tf.cast(tensor, dtype=dtype)
                    broadcasted_tensor = hvd.broadcast(tensor, root_rank,
                                                       process_set=this_set)
                    grad_ys = tf.ones([5] * dim)
                    grad = tf.gradients(broadcasted_tensor, tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            c = 1 if rank == root_rank else 0
            expected = np.ones([5] * dim) * c
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_alltoall_cpu_process_sets(self):
        """Test that the alltoall on restricted process sets correctly distributes 1D, 2D, and 3D tensors."""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        if rank in self.even_ranks:
            set_size = len(self.even_ranks)
            set_ranks = self.even_ranks
        elif rank in self.odd_ranks:
            set_size = len(self.odd_ranks)
            set_ranks = self.odd_ranks

        dtypes = self.filter_supported_types([tf.uint8, tf.int8, tf.uint16, tf.int16,
                                              tf.int32, tf.int64, tf.float16, tf.float32,
                                              tf.float64])
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                vals = []
                for i in set_ranks:
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                splits = tf.convert_to_tensor([rank+1] * set_size, dtype=tf.int32)
                if rank in self.even_ranks:
                    collected, received_splits = hvd.alltoall(tensor, splits, process_set=self.even_set)
                elif rank in self.odd_ranks:
                    collected, received_splits = hvd.alltoall(tensor, splits, process_set=self.odd_set)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), sum(rk + 1 for rk in set_ranks) * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

                self.assertSequenceEqual(self.evaluate(received_splits).tolist(), [rk + 1 for rk in set_ranks],
                                         "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_gpu_process_sets(self):
        """Test that the GPU alltoall on restricted process sets correctly distributes 1D, 2D, and 3D tensors."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        # This test does not apply if NCCL version < 2.7.0
        if hvd.nccl_built() and hvd.nccl_built() < 2700:
            self.skipTest("NCCL-based Alltoall requires NCCL version >= 2.7.0.")

        rank = hvd.rank()
        local_rank = hvd.local_rank()

        if rank in self.even_ranks:
            set_size = len(self.even_ranks)
            set_ranks = self.even_ranks
        elif rank in self.odd_ranks:
            set_size = len(self.odd_ranks)
            set_ranks = self.odd_ranks

        dtypes = [tf.uint8, tf.int8, tf.uint16, tf.int16,
                  tf.int32, tf.int64, tf.float16, tf.float32,
                  tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/gpu:%s" % local_rank):
                vals = []
                for i in set_ranks:
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)
                splits = tf.convert_to_tensor([rank+1] * set_size, dtype=tf.int32)
                if rank in self.even_ranks:
                    collected, received_splits = hvd.alltoall(tensor, splits, process_set=self.even_set)
                elif rank in self.odd_ranks:
                    collected, received_splits = hvd.alltoall(tensor, splits, process_set=self.odd_set)

                self.assertTrue(
                    self.evaluate(tf.reduce_all(
                        tf.equal(tf.cast(collected, tf.int32), rank))),
                    "hvd.alltoall produces incorrect collected tensor")

                self.assertTrue(
                    self.evaluate(tf.equal(tf.size(collected), sum(rk + 1 for rk in set_ranks) * 2**(dim - 1))),
                    "hvd.alltoall collected wrong number of values")

                self.assertSequenceEqual(self.evaluate(received_splits).tolist(), [rk + 1 for rk in set_ranks],
                                         "hvd.alltoall returned incorrect received_splits")

    def test_horovod_alltoall_grad_cpu_process_sets(self):
        """Test the correctness of the alltoall gradient on CPU with restricted process sets."""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        if rank in self.even_ranks:
            set_size = len(self.even_ranks)
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_size = len(self.odd_ranks)
            this_set = self.odd_set

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float32, tf.float64]
        dims = [1, 2, 3]
        for dtype, dim in itertools.product(dtypes, dims):
            with tf.device("/cpu:0"):
                vals = []
                for i in range(set_size):
                  vals += [i] * (rank+1)
                tensor = tf.convert_to_tensor(vals, dtype=dtype)
                for _ in range(dim - 1):
                  tensor = tf.expand_dims(tensor, axis=1)
                  tensor = tf.concat([tensor, tensor], axis=1)

                if _executing_eagerly():
                    tensor = self.tfe.Variable(tensor)
                    splits = tf.convert_to_tensor([rank + 1] * set_size, dtype=tf.int32)
                    with tf.GradientTape() as tape:
                        collected, received_splits = hvd.alltoall(tensor, splits, process_set=this_set)
                else:
                    splits = tf.convert_to_tensor([rank + 1] * set_size, dtype=tf.int32)
                    collected, received_splits = hvd.alltoall(tensor, splits, process_set=this_set)

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

    def test_broadcast_object_process_sets(self):
        """ This should best be tested with more than two Horovod processes """
        rank = hvd.rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set
        root_rank = set_ranks[0]

        with tf.device("/cpu:0"):
            expected_even_obj = {
                'even': 123,
                0: [1, 2]
            }
            expected_odd_obj = {
                'odd': 456,
                1: [1, 2, 3, 4]
            }
            expected_obj = expected_even_obj if this_set == self.even_set else expected_odd_obj
            obj = expected_obj if hvd.rank() == root_rank else {}

            obj = hvd.broadcast_object(obj, root_rank=root_rank, process_set=this_set)
            self.assertDictEqual(obj, expected_obj)

    def test_broadcast_object_fn_process_sets(self):
        """ This should best be tested with more than two Horovod processes """
        if hvd._executing_eagerly() or _IS_TF2:
            # Only for TF 1.0 in graph mode
            return

        rank = hvd.rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set
        root_rank = set_ranks[0]

        with tf.device("/cpu:0"):
            expected_even_obj = {
                'even': 123,
                0: [1, 2]
            }
            expected_odd_obj = {
                'odd': 456,
                1: [1, 2, 3, 4]
            }
            expected_obj = expected_even_obj if this_set == self.even_set else expected_odd_obj
            obj = expected_obj if hvd.rank() == root_rank else {}

            bcast = hvd.broadcast_object_fn(root_rank=root_rank, process_set=this_set)
            obj = bcast(obj)
            self.assertDictEqual(obj, expected_obj)

    def test_allgather_object_process_sets(self):
        """ This should best be tested with more than two Horovod processes """

        rank = hvd.rank()
        size = hvd.size()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        # This test does not apply if there is only one worker.
        if size == 1:
            self.skipTest("Only one worker available")

        if rank in self.even_ranks:
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_ranks = self.odd_ranks
            this_set = self.odd_set

        with tf.device("/cpu:0"):
            d = {'metric_val_1': hvd.rank()}
            if this_set.rank() == 1:
                d['metric_val_2'] = 42 if this_set == self.even_set else 23

            results = hvd.allgather_object(d, process_set=this_set)

            expected = [{'metric_val_1': i} for i in set_ranks]
            if this_set.size() > 1:
                expected[1] = {'metric_val_1': set_ranks[1],
                               'metric_val_2': 42 if this_set == self.even_set else 23}

            self.assertEqual(len(results), this_set.size())
            self.assertListEqual(results, expected)

    def test_legacy_DistributedOptimizer_process_sets(self):
        """ Note that this test makes the most sense when running with > 2 processes. """
        if _executing_eagerly():
            # Legacy Optimizers only support graph mode.
            return

        resource_variables_by_default = tf.compat.v1.resource_variables_enabled()
        tf.compat.v1.disable_resource_variables()

        size = hvd.size()

        if size == 1:
            self.skipTest("Only one worker available")

        with self.test_session(use_gpu=False) as sess:
            class TestOptimizer(tf.compat.v1.train.Optimizer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.variable = tf.compat.v1.get_variable("dummy_var", initializer=[0.0],
                                                              use_resource=False)
                def compute_gradients(self, *args, **kwargs):
                    return [(tf.constant([float(hvd.rank())]), self.variable)]
                def _apply_dense(self, grad, var):
                    return var.assign_add(grad)

            opt = TestOptimizer(name="test", use_locking=False)
            opt = hvd.DistributedOptimizer(
                optimizer=opt,
                average_aggregated_gradients=True,
                process_set=self.even_set,
            )

            grads_and_vars = opt.compute_gradients()
            update_op = opt.apply_gradients(grads_and_vars)
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(update_op)

            computed_value = sess.run(opt._optimizer.variable.read_value())[0]
            if self.even_set.included():
                self.assertAlmostEqual(computed_value, sum(range(0, size, 2)) / self.even_set.size())
            else:
                self.assertAlmostEqual(computed_value, float(hvd.rank()))

        if resource_variables_by_default:
            tf.compat.v1.enable_resource_variables()

    def test_distributed_gradient_tape_process_sets(self):
        """ Note: test makes most sense with more than 2 nodes. """
        size = hvd.size()

        if size == 1:
            self.skipTest("Only one worker available")

        with tf.device("/cpu:0"):
            x = tf.constant(float(hvd.rank()))
            with tf.GradientTape() as g:
                g.watch(x)
                y = x * x
            dg = hvd.DistributedGradientTape(g, process_set=self.even_set)
            dy_dx = dg.gradient(y, [x])
        value, = self.evaluate(dy_dx)

        if self.even_set.included():
            self.assertAlmostEqual(value, 2. * sum(self.even_set.ranks) / self.even_set.size())
        else:
            self.assertAlmostEqual(value, 2. * hvd.rank())

    def test_horovod_reducescatter_cpu_process_sets(self):
        """Test on CPU that the reducescatter correctly sums or averages and scatters 1D, 2D, 3D tensors
        if restricted to non-global process sets."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        rank = hvd.rank()

        if rank in self.even_ranks:
            this_set = self.even_set
        else:
            this_set = self.odd_set
        process_set_size = this_set.size()
        process_set_rank = this_set.rank()

        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/cpu:0"):
                even_rank_tensor = self.random_uniform([process_set_size * 4] * dim, -100, 100, dtype=dtype)
                odd_rank_tensor = self.random_uniform([process_set_size * 4] * dim, -100, 100, dtype=dtype)
                if rank in self.even_ranks:
                    tensor = even_rank_tensor
                else:
                    tensor = odd_rank_tensor
                reduced = hvd.reducescatter(tensor, op=red_op, process_set=this_set)
            if red_op == hvd.Sum:
                expected = tf.cast(tensor[process_set_rank * 4:(process_set_rank + 1) * 4] * process_set_size,
                                   reduced.dtype)
            elif red_op == hvd.Average:
                expected = tf.cast(tensor[process_set_rank * 4:(process_set_rank + 1) * 4], reduced.dtype)
            max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif process_set_size < 10:
                threshold = 1e-4
            elif process_set_size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_reducescatter_gpu_process_sets(self):
        """Test that the reducescatter works on GPUs if restricted to non-global process sets."""
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest("No GPUs available")

        if int(os.environ.get('HOROVOD_MIXED_INSTALL', 0)):
            # Skip if compiled with CUDA but without HOROVOD_GPU_OPERATIONS.
            self.skipTest("Not compiled with HOROVOD_GPU_OPERATIONS")

        local_rank = hvd.local_rank()
        rank = hvd.rank()

        if rank in self.even_ranks:
            this_set = self.even_set
        else:
            this_set = self.odd_set
        process_set_size = this_set.size()
        process_set_rank = this_set.rank()

        dtypes = [tf.int32, tf.int64, tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/gpu:%d" % local_rank):
                even_rank_tensor = self.random_uniform([process_set_size * 4] * dim, -100, 100, dtype=dtype)
                odd_rank_tensor = self.random_uniform([process_set_size * 4] * dim, -100, 100, dtype=dtype)
                if rank in self.even_ranks:
                    tensor = even_rank_tensor
                else:
                    tensor = odd_rank_tensor
                reduced = hvd.reducescatter(tensor, op=red_op, process_set=this_set)
            if red_op == hvd.Sum:
                expected = tf.cast(tensor[process_set_rank * 4:(process_set_rank + 1) * 4] * process_set_size,
                                   reduced.dtype)
            elif red_op == hvd.Average:
                expected = tf.cast(tensor[process_set_rank * 4:(process_set_rank + 1) * 4], reduced.dtype)
            max_difference = tf.reduce_max(tf.abs(reduced - expected))

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif process_set_size < 10:
                threshold = 1e-4
            elif process_set_size < 15:
                threshold = 5e-4
            else:
                return

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter on GPU produces incorrect results")

    def test_horovod_reducescatter_grad_cpu_process_sets(self):
        """Test the correctness of the reducescatter gradient on CPU if restricted to non-global process sets."""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        rank = hvd.rank()

        # As of TensorFlow v1.9, gradients are not supported on
        # integer tensors
        dtypes = [tf.float16, tf.float32, tf.float64]
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/cpu:0"):
                if _executing_eagerly():
                    even_rank_tensor = self.tfe.Variable(self.random_uniform(
                        [len(self.even_ranks) * 4] * dim, -100, 100, dtype=dtype))
                    odd_rank_tensor = self.tfe.Variable(self.random_uniform(
                        [len(self.odd_ranks) * 4] * dim, -100, 100, dtype=dtype))
                    with tf.GradientTape() as tape:
                        if rank in self.even_ranks:
                            reduced = hvd.reducescatter(even_rank_tensor, op=red_op,
                                                       process_set=self.even_set)
                        elif rank in self.odd_ranks:
                            reduced = hvd.reducescatter(odd_rank_tensor, op=red_op,
                                                       process_set=self.odd_set)
                else:
                    even_rank_tensor = self.random_uniform([len(self.even_ranks) * 4] * dim, -100, 100, dtype=dtype)
                    odd_rank_tensor = self.random_uniform([len(self.odd_ranks) * 4] * dim, -100, 100, dtype=dtype)
                    if rank in self.even_ranks:
                        reduced = hvd.reducescatter(even_rank_tensor, op=red_op,
                                                   process_set=self.even_set)
                    elif rank in self.odd_ranks:
                        reduced = hvd.reducescatter(odd_rank_tensor, op=red_op,
                                                   process_set=self.odd_set)

                if rank in self.even_ranks:
                    tensor = even_rank_tensor
                    set_size = len(self.even_ranks)
                elif rank in self.odd_ranks:
                    tensor = odd_rank_tensor
                    set_size = len(self.odd_ranks)

                grad_ys = tf.ones([4] + [set_size * 4] * (dim - 1), dtype=tensor.dtype)
                if _executing_eagerly():
                    grad_out = tape.gradient(tf.cast(reduced, dtype=tensor.dtype), tensor, grad_ys)
                else:
                    grad = tf.gradients(tf.cast(reduced, dtype=tensor.dtype), tensor, grad_ys)[0]
                    grad_out = self.evaluate(grad)

            if red_op == hvd.Sum:
                expected = np.ones([set_size * 4] * dim) * set_size
            elif red_op == hvd.Average:
                expected = np.ones([set_size * 4] * dim)
            err = np.linalg.norm(expected - grad_out)
            self.assertLess(err, 0.00000001,
                            "gradient %s differs from expected %s, "
                            "error: %s" % (grad_out, expected, str(err)))

    def test_horovod_grouped_reducescatter_cpu_process_sets(self):
        """Test on CPU that the grouped reducescatter correctly sums if restricted to non-global process sets"""
        if hvd.ccl_built():
            self.skipTest("Reducescatter is not supported yet with oneCCL operations.")
        if _is_mac and hvd.gloo_built() and not hvd.mpi_built():
            self.skipTest("ReducescatterGloo is not supported on macOS")
        rank = hvd.rank()

        if rank in self.even_ranks:
            this_set = self.even_set
        else:
            this_set = self.odd_set
        process_set_size = this_set.size()
        process_set_rank = this_set.rank()

        dtypes = self.filter_supported_types([tf.int32, tf.int64, tf.float16, tf.float32, tf.float64])
        dims = [1, 2, 3]
        for red_op, dtype, dim in itertools.product([hvd.Sum, hvd.Average], dtypes, dims):
            with tf.device("/cpu:0"):
                even_rank_tensors = [self.random_uniform([process_set_size * 4] * dim, -100, 100, dtype=dtype)
                                     for _ in range(5)]
                odd_rank_tensors = [self.random_uniform([process_set_size * 4] * dim, -100, 100, dtype=dtype)
                                    for _ in range(5)]
                if rank in self.even_ranks:
                    tensors = even_rank_tensors
                else:
                    tensors = odd_rank_tensors
                reduced = hvd.grouped_reducescatter(tensors, op=red_op, process_set=this_set)
            if red_op == hvd.Sum:
                expected = [tf.cast(tensor[process_set_rank * 4:(process_set_rank + 1) * 4] * process_set_size,
                                    reduced[0].dtype) for tensor in tensors]
            elif red_op == hvd.Average:
                expected = [tf.cast(tensor[process_set_rank * 4:(process_set_rank + 1) * 4], reduced[0].dtype)
                            for tensor in tensors]
            max_difference = tf.reduce_max([tf.reduce_max(tf.abs(t1 - t2)) for t1, t2 in zip(reduced, expected)])

            # Threshold for floating point equality depends on number of
            # ranks, since we're comparing against precise multiplication.
            if dtype == tf.float16:
                threshold = .5
            elif dtype in [tf.int32, tf.int64]:
                threshold = 0
            elif process_set_size < 10:
                threshold = 1e-4
            elif process_set_size < 15:
                threshold = 5e-4
            else:
                break

            diff = self.evaluate(max_difference)
            self.assertTrue(diff <= threshold,
                            "hvd.reducescatter produces incorrect results")

    def test_horovod_grouped_allgather_cpu_process_sets(self):
        """Test that the grouped allgather correctly gathers 1D, 2D, 3D tensors
        if restricted to non-global process sets."""
        rank = hvd.rank()

        if hvd.ccl_built():
            self.skipTest("Multiple process sets currently do not support CCL.")

        if rank in self.even_ranks:
            set_size = len(self.even_ranks)
            set_ranks = self.even_ranks
            this_set = self.even_set
        elif rank in self.odd_ranks:
            set_size = len(self.odd_ranks)
            set_ranks = self.odd_ranks
            this_set = self.odd_set

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
                gathered = hvd.grouped_allgather(tensors, process_set=this_set)

            gathered_tensors = self.evaluate(gathered)
            for gathered_tensor in gathered_tensors:
                self.assertEqual(list(gathered_tensor.shape),
                                 [17 * set_size] + [17] * (dim - 1))

            for i in range(set_size):
                rank_tensors = [tf.slice(gathered_tensor,
                                         [i * 17] + [0] * (dim - 1),
                                         [17] + [-1] * (dim - 1))
                                for gathered_tensor in gathered_tensors]
                self.assertEqual([rank_tensor.shape for rank_tensor in rank_tensors], len(tensors) * [[17] * dim])
                # tf.equal() does not support tf.uint16 as of TensorFlow 1.2,
                # so need to cast rank_tensor to tf.int32.
                if dtype != tf.bool:
                    value = set_ranks[i]
                else:
                    value = set_ranks[i] % 2
                self.assertTrue(all(self.evaluate(tf.reduce_all(
                    tf.equal(tf.cast(rank_tensor, tf.int32), value))) for rank_tensor in rank_tensors),
                    "hvd.grouped_allgather produces incorrect gathered tensor")


from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
run_all_in_graph_and_eager_modes(TensorFlowProcessSetsTests)

if __name__ == '__main__':
    tf.test.main()
