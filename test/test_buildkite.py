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
from __future__ import print_function

import string
import unittest
import warnings

from horovod.run.common.util import tiny_shell_exec


class BuildKiteTests(unittest.TestCase):
    """
    Tests for .buildkite directory
    """

    def __init__(self, *args, **kwargs):
        super(BuildKiteTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    """
    Tests the generated buildkite pipeline.
    
    Compares output of .buildkite/gen_pipeline.sh with test/data/expected_buildkite_pipeline.yaml.
    To see the changes in the output, run the following in your Horovod project root:
    
        BUILDKITE_PIPELINE_SLUG=SLUG BUILDKITE_BRANCH=BRANCH .buildkite/gen-pipeline.sh > test/data/expected_buildkite_pipeline.yaml
    
    Then run `git diff` to see the changes in the generated pipeline YAML.
    Commit `test/data/expected_buildkite_pipeline.yaml` to get those changes into your PR.
    """
    def test_gen_pipeline(self):
        with open('data/expected_buildkite_pipeline.yaml', 'r') as f:
            lines = f.readlines()
            expected_pipeline = ''.join(lines)

        gen_pipeline_env = 'BUILDKITE_PIPELINE_SLUG=SLUG BUILDKITE_BRANCH=BRANCH'
        gen_pipeline_cmd = '{env} ../.buildkite/gen-pipeline.sh'.format(env=gen_pipeline_env)
        actual_pipeline, exit_code = tiny_shell_exec.execute(gen_pipeline_cmd)

        self.assertEqual(0, exit_code)
        self.assertEqual(expected_pipeline, actual_pipeline)
