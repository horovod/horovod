from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion
import collections
import inspect
import itertools
import numpy as np
import os
import tempfile
import torch
import torch.nn.functional as F
import unittest
import warnings

import horovod.torch as hvd

class TorchTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_horovod_multiple_allreduce_cpu(self):
        hvd.init()
        size = hvd.size()
        if hvd.rank() == 0:
            tensors = [torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]]),torch.FloatTensor([[9.0, 10.0], [11.0, 12.0]])]
        else:
            tensors = [torch.FloatTensor([[5.0, 6.0], [7.0, 8.0]]), torch.FloatTensor([[13.0, 14.0], [15.0, 16.0]])]
        summed = 0
        for tensor in tensors:
            summed += hvd.allreduce(tensor, average=False)
        print(summed)

    def test_horovod_single_allreduce_cpu(self):
        hvd.init()
        size = hvd.size()
        if hvd.rank() == 0:
            tensor = torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]])
        else:
            tensor = torch.FloatTensor([[5.0, 6.0], [7.0, 8.0]])
        summed = hvd.allreduce(tensor, average=False)
        print(summed)

if __name__ == "__main__":
   unittest.main()