import numpy as np
import os
import time

import tensorflow as tf
import horovod.tensorflow as hvd
hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.visible_device_list = str(hvd.local_rank())

from tensorflow.python.ops.nccl_ops import all_sum

#bytes = [2**i for i in range(23)]
bytes = [2**23]

with tf.device('/cpu:0'):
    tensors = [
        tf.get_variable("var%i" % i, shape=[size], dtype = tf.float32, initializer=tf.zeros_initializer(), use_resource=False, trainable=False)
        for i, size in enumerate(bytes)
    ]

ops = [
    hvd.allreduce(tensor)
    for tensor in tensors
]
#ops = all_sum(tensors)

with tf.Session(config=config) as session:
    session.run(tf.initializers.global_variables())
    session.run(ops)

    rept = 100
    st = time.time()
    for _ in range(rept):
        session.run(ops)
    ed = time.time()

    print((ed - st)/rept, flush= True)