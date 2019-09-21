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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

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
    def evaluate(self, tensors):
        sess = ops.get_default_session()
        if sess is None:
            with self.test_session(config=config) as sess:
                return sess.run(tensors)
        else:
            return sess.run(tensors)


    # def test_horovod_multiple_allreduce_cpu(self):
    #     """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
    #     hvd.init()
    #     size = hvd.size()
    #     rank0_tensors = [np.asarray([[1.0, 2.0], [3.0, 4.0]]), np.asarray([[9.0, 10.0], [11.0, 12.0]])]
    #     rank1_tensors = [np.asarray([[5.0, 6.0], [7.0, 8.0]]), np.asarray([[13.0, 14.0], [15.0, 16.0]])]

    #     expected = []
    #     for a,b in zip(rank0_tensors, rank1_tensors):
    #         answer = AdaSum_reference_operation(a, b)
    #         expected.append(answer)

    #     for dtype in [tf.float16, tf.float32, tf.float64]:
    #         with tf.device("/cpu:0"):
    #             tensors = map(tf.constant, rank0_tensors if hvd.rank() == 0 else rank1_tensors)
    #             # cast to the corresponding dtype
    #             tensors = map(lambda tensor: tf.cast(tensor, dtype), tensors)
    #             # and away we go: do reduction
    #             reduced_tensors = [
    #                 self.evaluate(hvd.allreduce(tensor, average=False, allreduce_type=AllreduceType.MsAllreduce))
    #                 for tensor in tensors
    #             ]
    #             # cast expected result to the type of the tensorflow values
    #             np_type = dtype.as_numpy_dtype
    #             tmp = [t.astype(np_type) for t in expected]
    #             self.assertAllClose(tmp, reduced_tensors)

    def test_horovod_multiple_allreduce_gpu(self):
        """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
        hvd.init()
        size = hvd.size()
        
        all_tensors = []
        for i in range(8):
            # all_tensors.append([np.asarray([[(1.0), (2.0)], [(3.0), (4.0)]]), np.asarray([[(5.0), (6.0)], [(7.0), (8.0)]])])
            # all_tensors.append([np.asarray([[(1.0+i), (2.0+i)], [(3.0+i), (4.0+i)]]), np.asarray([[(5.0+i), (6.0+i)], [(7.0+i), (8.0+i)]])])
            all_tensors.append([np.asarray([(1.0+i), (1.0+i)])])
            # all_tensors.append([np.asarray([[(1.0+i)*(i==0), (2.0+i)*(i==1)], [(3.0+i)*(i==2), (4.0+i)*(i==3)]]), np.asarray([[(5.0+i)*(i==0), (6.0+i)*(i==1)], [(7.0+i)*(i==2), (8.0+i)*(i==3)]])])


        # rank0_tensors = [np.asarray([[1.0, 2.0], [3.0, 4.0]]), np.asarray([[9.0, 10.0], [11.0, 12.0]])]
        # rank1_tensors = [np.asarray([[1.0, 2.0], [3.0, 4.0]]), np.asarray([[9.0, 10.0], [11.0, 12.0]])]
        # rank0_tensors = [np.asarray([[9.0, 10.0], [11.0, 12.0]])]
        # rank1_tensors = [np.asarray([[9.0, 10.0], [11.0, 12.0]])]

        expected = all_tensors[0]
        for i in [3, 2, 1, 5, 6, 7, 4]:
            answer0 = adasum_reference_operation(expected[0], all_tensors[i][0])
            expected = [answer0]
        rank_num = hvd.local_rank()
        for dtype in [tf.float32]:
            with tf.device("/gpu:{}".format(rank_num)):
                tensors = map(tf.constant, all_tensors[hvd.rank()])
                # tensors = map(tf.constant, rank0_tensors if hvd.rank() == 0 else rank1_tensors)
                # cast to the corresponding dtype
                tensors = map(lambda tensor: tf.cast(tensor, dtype), tensors)
                # and away we go: do reduction
                reduced_tensors = [
                    self.evaluate(hvd.allreduce(tensor, average=False, allreduce_type=AllreduceType.MsAllreduce))
                    for tensor in tensors
                ]
                # cast expected result to the type of the tensorflow values
                np_type = dtype.as_numpy_dtype
                tmp = [t.astype(np_type) for t in expected]
                self.assertAllClose(tmp, reduced_tensors)

    # def test_horovod_multiple_large_tensors_allreduce_cpu(self):
    #     """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
    #     hvd.init()
    #     size = hvd.size()
    #     base_dim = [16,32,64]
    #     dim_multipliers = [1, 4, 8, 16, 32, 64]
    #     #for multiplier in dim_multipliers:
    #     multiplier = dim_multipliers[5]
    #     true_dim = base_dim.copy()
    #     true_dim[2] = true_dim[2] * multiplier
    #     start_time = datetime.utcnow()
    #     rep = 1
    #     tensor_count = 100
    #     with tf.device("/cpu:0"):
    #         tf.set_random_seed(1234)
            
    #         for _ in range(rep):
    #             summed = []
    #             for _ in range(tensor_count):
    #                 tensor = tf.random_uniform(
    #                 true_dim, -100, 100, dtype=tf.float32)
    #                 summed.append(hvd.allreduce(tensor, average=False))
    #             result_sum = self.evaluate(summed)
    #             #print(result_sum)
    #     end_time = datetime.utcnow()
    #     time_delta = end_time - start_time
    #     tensor_size = np.prod(true_dim) / 256
    #     print("{} {}K tensors {} Cycles took {}".format(tensor_count, tensor_size, rep, time_delta.total_seconds()))

    # def test_horovod_single_large_tensor_allreduce_cpu(self):
    #     """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
    #     hvd.init()
    #     size = hvd.size()
    #     base_dim = [16,32,64]
    #     dim_multipliers = [1, 4, 8, 16, 32, 64]
    #     #for multiplier in dim_multipliers:
    #     multiplier = dim_multipliers[5]
    #     true_dim = base_dim.copy()
    #     true_dim[2] = true_dim[2] * multiplier
    #     with tf.device("/cpu:0"):
    #         tf.set_random_seed(1234)
    #         tensor = tf.random_uniform(
    #                 true_dim, -100, 100, dtype=tf.float32)
    #         start_time = datetime.utcnow()
           
    #         for _ in range(100):
    #             summed = 0
    #             summed = hvd.allreduce(tensor, average=False)
    #             result_sum = self.evaluate(summed)
    #             #print(result_sum)
    #         end_time = datetime.utcnow()
    #         time_delta = end_time - start_time
    #         tensor_size = np.prod(true_dim) / 256
    #         print("{}K tensor Cycle took {}".format(tensor_size,time_delta.total_seconds()))

    # def test_horovod_single_allreduce_cpu(self):
    #     """Test on CPU that the allreduce correctly sums 1D, 2D, 3D tensors."""
    #     hvd.init()
    #     size = hvd.size()
    #     with tf.device("/cpu:0"):
    #         if hvd.rank() == 0:
    #             tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    #         else:
    #             tensor = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    #         summed = hvd.allreduce(tensor, average=False)
    #     diff = self.evaluate(summed)
    #     print(diff)

    # def test_horovod_multithread_init(self):
    #     """Test thread pool init"""
    #     hvd.init()

if __name__ == '__main__':
    tf.test.main()
