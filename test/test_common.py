# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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
import unittest
import warnings

from horovod.common.util import _cache, extension_available, gloo_built, mpi_built


class CommonTests(unittest.TestCase):
    """
    Tests for horovod.common.
    """

    def __init__(self, *args, **kwargs):
        super(CommonTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_gloo_built(self):
        """Test that Gloo has been built if env is set."""
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))
        if gloo_rank >= 0:
            self.assertTrue(gloo_built())

    def test_mpi_built(self):
        """Test that MPI has been built if env is set."""
        gloo_rank = int(os.getenv('HOROVOD_RANK', -1))
        if gloo_rank == -1:
            self.assertTrue(mpi_built())

    def test_tensorflow_available(self):
        """Test that TensorFLow support has been built."""
        available = extension_available('tensorflow')
        try:
            self.assertTrue(available)
        except:
            self.assertFalse(available)

    def test_torch_available(self):
        """Test that PyTorch support has been built."""
        available = extension_available('torch')
        try:
            self.assertTrue(available)
        except:
            self.assertFalse(available)

    def test_mxnet_available(self):
        """Test that MXNet support has been built."""
        available = extension_available('mxnet')
        try:
            self.assertTrue(available)
        except:
            self.assertFalse(available)

    def test_cache(self):
        """Test that caching of expensive functions only computes values once."""
        state = {}

        @_cache
        def fn():
            return state['key']

        # Not yet cached
        state['key'] = 1
        value = fn()
        assert value == 1

        # Cached
        state['key'] = 2
        value = fn()
        assert value == 1
