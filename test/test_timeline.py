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
import tempfile
import time
import torch
import unittest
import warnings

import horovod.torch as hvd

from common import env


class TimelineTests(unittest.TestCase):
    """
    Tests for ops in horovod.torch.
    """

    def __init__(self, *args, **kwargs):
        super(TimelineTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_timeline(self):
        with tempfile.NamedTemporaryFile() as t:
            with env(HOROVOD_TIMELINE=t.name, HOROVOD_TIMELINE_MARK_CYCLES='1'):
                hvd.init()

                # Perform a simple allreduce operation
                hvd.allreduce(torch.tensor([1, 2, 3]), name='test_allreduce')

                # Wait for it to register in the timeline.
                time.sleep(0.1)

                if hvd.rank() == 0:
                    with open(t.name, 'r') as tf:
                        timeline_text = tf.read()
                        assert 'allreduce.test_allreduce' in timeline_text, timeline_text
                        assert 'NEGOTIATE_ALLREDUCE' in timeline_text, timeline_text
                        assert 'ALLREDUCE' in timeline_text, timeline_text
                        assert 'CYCLE_START' in timeline_text, timeline_text
