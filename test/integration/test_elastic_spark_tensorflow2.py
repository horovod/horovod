# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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

from elastic_spark_common import BaseElasticSparkTests


class ElasticSparkTensorflow2Tests(BaseElasticSparkTests):
    __test__ = True

    def __init__(self, *args, **kwargs):
        training_script = os.path.join(os.path.dirname(__file__), 'data/elastic_tensorflow2_main.py')
        super(ElasticSparkTensorflow2Tests, self).__init__(training_script, *args, **kwargs)

    def test_fault_tolerance_hosts_added_and_removed(self):
        self.skipTest('This test fails due to https://github.com/horovod/horovod/issues/1994')

    def test_auto_scale_down_by_discovery(self):
        self.skipTest('This test fails due to https://github.com/horovod/horovod/issues/1994')
