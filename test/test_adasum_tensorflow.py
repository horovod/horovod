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

class MPITests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """
    def __init__(self, *args, **kwargs):
        super(MPITests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        #self.config.gpu_options.visible_device_list = str(hvd.local_rank())

    def evaluate(self, tensors):
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=self.config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)


    def test_horovod_adasum_multiple_allreduce_cpu(self):
        """Test on CPU that the Adasum correctly computes 2D tensors."""
        hvd.init()
        size = hvd.size()
        # TODO support testing with non-power 2 ranks
        if not is_power2(size):
            return
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

    def test_horovod_adasum_multiple_allreduce_gpu(self):
        """Test on GPU that the Adasum correctly computes 2D tensors."""
        if not tf.test.is_gpu_available(cuda_only=True):
            return
        hvd.init()
        rank = hvd.rank()
        rank_tensors = []
        size = hvd.size()
        # TODO support testing with non-power 2 ranks
        if not is_power2(size):
            return

        local_size = hvd.local_size()
        is_homogeneous = size % local_size == 0

        # Only run on homogeneous cluster
        if(not is_homogeneous):
            return
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
