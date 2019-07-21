from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import horovod.torch as hvd
from horovod.common.util import env
import torch
import time
import signal


def test():
    signal.alarm(45)
    with env(HOROVOD_STALL_CHECK_TIME_SECONDS="2",
             HOROVOD_STALL_SHUTDOWN_TIME_SECONDS="5"):
        hvd.init()
        tensor = torch.IntTensor([[1, 2], [3, 4]])
        if hvd.rank() != 0:
            time.sleep(10 * hvd.rank());
        try:
            summed = hvd.allreduce(tensor, average=False)
        except:
            pass
        finally:
            hvd.shutdown()


if __name__ == "__main__":
    test()
