from mpi4py import MPI
import horovod.tensorflow as hvd
import tensorflow as tf
import time
import os
import signal
from common import env

def test():
  signal.alarm(45)
  with env(HOROVOD_STALL_CHECK_TIME_SECONDS="2", 
    HOROVOD_STALL_SHUTDOWN_TIME_SECONDS="5"):
    config = tf.ConfigProto()
    tf.enable_eager_execution(config=config)
    hvd.init()

    with tf.device("/cpu:0"):
        tf.set_random_seed(1234)
        tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        if hvd.rank() != 0:
          time.sleep(10 * hvd.rank());
        try:
          hvd.allreduce(tensor, average=False)
        except:
          print "except"
        finally:
          print "finally"
          hvd.shutdown()

if __name__ == "__main__":
  test()