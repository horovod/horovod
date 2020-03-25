# Copyright 2019 Microsoft. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from horovod.tensorflow.util import _executing_eagerly, _has_eager
import warnings
from datetime import datetime
import horovod.tensorflow as hvd
import math
import copy
import subprocess

def adasum_reference_operation(a,b):
    assert a.size == b.size
    assert a.size > 0 and b.size > 0
    assert a.dtype == b.dtype
    # Adasum logic in numpy
    anormsq = np.inner(a.ravel(), a.ravel())
    bnormsq = np.inner(b.ravel(), b.ravel())
    dotProduct = np.dot(a.ravel(), b.ravel())
    acoeff = 1.0
    bcoeff = 1.0
    if anormsq != 0:
        acoeff = 1.0 - dotProduct / anormsq * 0.5
    if bnormsq != 0:
        bcoeff = 1.0 - dotProduct / bnormsq * 0.5
    answer = acoeff * a + bcoeff * b
    return answer

def is_power2(num):
    return num != 0 and ((num & (num -1)) == 0)

def reference_tree_reduction(tensors, hvd_size):
    if hvd_size == 1:
        return tensors[0]
    temp = copy.copy(tensors)
    power_of_2 = int(math.log(hvd_size, 2))
    for level in range(power_of_2):
        for i in range(int(hvd_size / pow(2, level + 1))):
            answer = []
            for a,b in zip(temp[i * 2], temp[i * 2 + 1]):
                answer.append(adasum_reference_operation(a, b))
            temp[i] = copy.copy(answer)
    return temp[0]

if hasattr(tf, 'ConfigProto'):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

if hasattr(tf, 'config') and hasattr(tf.config, 'experimental') \
        and hasattr(tf.config.experimental, 'set_memory_growth'):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    if _has_eager:
        # Specifies the config to use with eager execution. Does not preclude
        # tests from running in the graph mode.
        tf.enable_eager_execution(config=config)
class MPITests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """
    def __init__(self, *args, **kwargs):
        super(MPITests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        if _has_eager:
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


    def test_horovod_adasum_multiple_allreduce_cpu(self):
        """Test on CPU that the Adasum correctly computes 2D tensors."""
        hvd.init()
        # TODO support non-MPI Adasum operation
        if not hvd.mpi_enabled():
            self.skipTest("MPI not enabled")

        size = hvd.size()
        # TODO support testing with non-power 2 ranks
        if not is_power2(size):
            self.skipTest("MPI rank is not power of 2")

        rank = hvd.rank()
        rank_tensors = []
        for _ in range(size):
            rank_tensors.append([np.random.random_sample((2,2)), np.random.random_sample((2,2))])
        answer = reference_tree_reduction(rank_tensors, size)

        for dtype in [tf.float16, tf.float32, tf.float64]:
            with tf.device("/cpu:0"):
                tensors = map(tf.constant, rank_tensors[rank])
                # cast to the corresponding dtype
                tensors = map(lambda tensor: tf.cast(tensor, dtype), tensors)
                # and away we go: do reduction
                reduced_tensors = [
                    self.evaluate(hvd.allreduce(tensor, op=hvd.Adasum))
                    for tensor in tensors
                ]
                # cast expected result to the type of the tensorflow values
                np_type = dtype.as_numpy_dtype
                tmp = [t.astype(np_type) for t in answer]
                self.assertAllCloseAccordingToType(tmp, reduced_tensors)

    def test_horovod_adasum_multiple_allreduce_gpu_nccl(self):
        """Test on GPU using NCCL that the Adasum correctly computes 2D tensors."""
        hvd.init()
        # TODO support non-MPI Adasum operation
        if not hvd.mpi_enabled() or not hvd.gpu_available('tensorflow') or not hvd.nccl_built():
            self.skipTest("MPI, GPU or NCCL not available")

        rank = hvd.rank()
        rank_tensors = []
        size = hvd.size()
        # TODO support testing with non-power 2 ranks
        if not is_power2(size):
            self.skipTest("MPI rank is not power of 2")

        local_size = hvd.local_size()

        # Only run on homogeneous cluster
        if not hvd.is_homogeneous():
            self.skipTest("Horovod cluster is not homogeneous")

        num_nodes = int(size / local_size)
        for _ in range(size):
            rank_tensors.append([np.random.random_sample((2,2)), np.random.random_sample((2,2))])
        sum_local_ranks_tensor = []
        for i in range(num_nodes):
            sum_local_ranks_tensor.append([np.zeros((2,2)), np.zeros((2,2))])
            for j in range(local_size):
                sum_local_ranks_tensor[i] = np.add(sum_local_ranks_tensor[i], rank_tensors[j])

        answer = reference_tree_reduction(sum_local_ranks_tensor, num_nodes)
        answer = np.true_divide(answer, local_size)
        for dtype in [tf.float16, tf.float32, tf.float64]:
            with tf.device("/gpu:{}".format(hvd.local_rank())):
                tensors = map(tf.constant, rank_tensors[rank])
                # cast to the corresponding dtype
                tensors = map(lambda tensor: tf.cast(tensor, dtype), tensors)
                # and away we go: do reduction
                reduced_tensors = [
                    self.evaluate(hvd.allreduce(tensor, op=hvd.Adasum))
                    for tensor in tensors
                ]
                # cast expected result to the type of the tensorflow values
                np_type = dtype.as_numpy_dtype
                tmp = [t.astype(np_type) for t in answer]
                self.assertAllCloseAccordingToType(tmp, reduced_tensors)

if __name__ == '__main__':
    tf.test.main()
