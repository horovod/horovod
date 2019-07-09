# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import time
import torch
import traceback
import unittest
import warnings

import horovod.torch as hvd
from horovod.run import run_func


class RunFuncTests(unittest.TestCase):
    """
    Tests for horovod.run.run_func().
    """
    def __init__(self, *args, **kwargs):
        super(RunFuncTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_happy_run(self):

        def fn():
            hvd.init()
            res = hvd.allgather(torch.tensor([hvd.rank()])) + hvd.rank() * 10
            return res.tolist(), hvd.rank()

        results = run_func(fn, num_proc=3, host='127.0.0.1:3',
                           env={'PATH': os.environ.get('PATH')})
        self.assertListEqual([([0, 1, 2], 0), ([10, 11, 12], 1), ([20, 21, 22], 2)], results)
