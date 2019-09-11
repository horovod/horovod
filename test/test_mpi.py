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

import unittest
import warnings

from mpi4py import MPI

import horovod.torch as hvd


class MPITests(unittest.TestCase):
    """
    Tests for horovod.common.
    """

    def __init__(self, *args, **kwargs):
        super(MPITests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_mpi_built(self):
        """Test that MPI has been built successfully."""
        self.assertTrue(hvd.mpi_built())

    def test_mpi_enabled(self):
        """Test that MPI has been enabled following initialization."""
        hvd.init()
        self.assertTrue(hvd.mpi_enabled())

    def test_mpi_threads_supported(self):
        """Test the MPI threads are supported if MPI is enabled."""
        hvd.init()

        provided = MPI.Query_thread()
        threads_supported = provided == MPI.THREAD_MULTIPLE
        self.assertEqual(hvd.mpi_threads_supported(), threads_supported)
