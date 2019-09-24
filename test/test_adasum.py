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
from horovod.tensorflow import AllreduceType

def adasum_reference_operation(a,b):
    assert a.size == b.size
    assert a.size > 0 and b.size > 0
    assert a.dtype == b.dtype
    # AdaSum logic in numpy
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

class MPITests(tf.test.TestCase):
    """
    Tests for ops in horovod.tensorflow.
    """
    def __init__(self, *args, **kwargs):
        super(MPITests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        hvd.init()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.visible_device_list = str(hvd.local_rank())

    def evaluate(self, tensors):
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=self.config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)


    def test_horovod_multiple_allreduce_cpu(self):
        """Test on CPU that the Adasum correctly computes 2D tensors."""
        size = hvd.size()
        rank0_tensors = [np.asarray([[1.0, 2.0], [3.0, 4.0]]), np.asarray([[9.0, 10.0], [11.0, 12.0]])]
        rank1_tensors = [np.asarray([[5.0, 6.0], [7.0, 8.0]]), np.asarray([[13.0, 14.0], [15.0, 16.0]])]

        expected = []
        for a,b in zip(rank0_tensors, rank1_tensors):
            answer = adasum_reference_operation(a, b)
            expected.append(answer)

        for dtype in [tf.float16, tf.float32, tf.float64]:
            with tf.device("/cpu:0"):
                tensors = map(tf.constant, rank0_tensors if hvd.rank() == 0 else rank1_tensors)
                # cast to the corresponding dtype
                tensors = map(lambda tensor: tf.cast(tensor, dtype), tensors)
                # and away we go: do reduction
                reduced_tensors = [
                    self.evaluate(hvd.allreduce(tensor, average=False, allreduce_type=AllreduceType.Adasum))
                    for tensor in tensors
                ]
                # cast expected result to the type of the tensorflow values
                np_type = dtype.as_numpy_dtype
                tmp = [t.astype(np_type) for t in expected]
                self.assertAllClose(tmp, reduced_tensors)

    def test_horovod_multiple_allreduce_gpu(self):
        """Test on GPU that the Adasum correctly computes 2D tensors."""
        size = hvd.size()
        print("Testing with {} ranks.".format(size))

        rank0_tensors = [np.asarray([[1.0, 2.0], [3.0, 4.0]]), np.asarray([[9.0, 10.0], [11.0, 12.0]])]
        rank1_tensors = [np.asarray([[1.0, 2.0], [3.0, 4.0]]), np.asarray([[9.0, 10.0], [11.0, 12.0]])]
        expected = []
        for a,b in zip(rank0_tensors, rank1_tensors):
            answer = adasum_reference_operation(a, b)
            expected.append(answer)
        rank_num = hvd.local_rank()
        for dtype in [tf.float16, tf.float32, tf.float64]:
            #with tf.device("/gpu:%d" % rank_num):
                tensors = map(tf.constant, rank0_tensors if rank_num == 0 else rank1_tensors)
                # cast to the corresponding dtype
                tensors = map(lambda tensor: tf.cast(tensor, dtype), tensors)
                # and away we go: do reduction
                reduced_tensors = [
                    self.evaluate(hvd.allreduce(tensor, average=False, allreduce_type=AllreduceType.Adasum))
                    for tensor in tensors
                ]
                # cast expected result to the type of the tensorflow values
                np_type = dtype.as_numpy_dtype
                tmp = [t.astype(np_type) for t in expected]
                self.assertAllClose(tmp, reduced_tensors)

if __name__ == '__main__':
    tf.test.main()
