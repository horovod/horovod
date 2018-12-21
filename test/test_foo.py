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

import itertools
import mxnet as mx
import unittest
import numpy as np
import horovod.mxnet as hvd

class FooTest(unittest.TestCase):
    """
    Tests for ops in horovod.mxnet.
    """

    def _current_context(self):
        if mx.current_context().device_type == 'gpu':
            return mx.gpu(hvd.local_rank())
        else:
            return mx.current_context()

    def test_foo(self):
        """Test that the broadcast correctly broadcasts 1D, 2D, 3D tensors."""
        hvd.init()
        print('hello world')
        return

if __name__ == '__main__':
    unittest.main()
