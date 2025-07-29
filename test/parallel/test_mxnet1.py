# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import itertools
import unittest
from packaging import version

import pytest
import numpy as np

from base_test_mxnet import *


if HAS_MXNET:
    # MXNet 1.4.x will kill test MPI process if error occurs during operation enqueue. Skip
    # those tests for versions earlier than 1.5.0.
    _skip_enqueue_errors = version.parse(mx.__version__) < version.parse('1.5.0')
else:
    _skip_enqueue_errors = False

import atexit
import mxnet as mx
import horovod.mxnet as hvd


@atexit.register
def finalize_horovod():
    try:
        if hvd.is_initialized():
            # Ensure any pending ops are flushed
            mx.nd.waitall()

            try:
                # Try barrier, but timeout after N seconds to avoid hang
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Rank {} timed out in hvd.barrier()".format(hvd.rank()))

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # 10 second timeout for barrier

                hvd.barrier()
                signal.alarm(0)

            except TimeoutError as te:
                print(f"[Rank {hvd.rank()}] barrier timeout: {te}")

            try:
                hvd.shutdown()
            except Exception as e:
                print(f"[Rank {hvd.rank()}] hvd.shutdown() failed: {e}")
    except Exception as e:
        print(f"[Rank N/A] Finalization error: {e}")


@pytest.mark.skipif(not HAS_MXNET, reason='MXNet unavailable')
@pytest.mark.skipif(version.parse(mx.__version__).major != 1, reason='MXNet v1.x tests')
class MX1Tests(MXTests, unittest.TestCase):
    """
    Tests for ops in horovod.mxnet. This tests MXNet 1.x specifically.
    """

    @unittest.skipUnless(has_gpu, "no gpu detected")
    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_grouped_allreduce_cpu_gpu_error(self):
        """Test that the grouped allreduce raises an error if the input tensor
           list contains a mix of tensors on CPU and GPU."""
        super(MX1Tests, self).test_horovod_grouped_allreduce_cpu_gpu_error()

    @unittest.skipUnless(has_gpu, "no gpu detected")
    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_grouped_allgather_cpu_gpu_error(self):
        """Test that the grouped allgather raises an error if the input tensor
           list contains a mix of tensors on CPU and GPU."""
        super(MX1Tests, self).test_horovod_grouped_allgather_cpu_gpu_error()

    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_alltoall_equal_split_length_error(self):
        """Test that the alltoall with default splitting returns an error if the first dimension
        of tensor is not a multiple of the number of workers."""
        super(MX1Tests, self).test_horovod_alltoall_equal_split_length_error()

    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_alltoall_splits_error(self):
        """Test that the alltoall returns an error if the sum of the splits entries exceeds
        the first dimension of the input tensor."""
        super(MX1Tests, self).test_horovod_alltoall_splits_error()

    @pytest.mark.skipif(_skip_enqueue_errors,
                        reason="Skip enqueue errors for MXNet version < 1.5.0")
    def test_horovod_alltoall_splits_type_error(self):
        """Test that the alltoall returns an error if the splits tensor does not
           contain 32-bit integers."""
        super(MX1Tests, self).test_horovod_alltoall_splits_type_error()


if __name__ == '__main__':
    unittest.main()
