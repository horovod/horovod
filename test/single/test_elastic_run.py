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

import os
import unittest
from tempfile import NamedTemporaryFile

import horovod
from horovod.common.util import gloo_built
from horovod.runner.common.util.env import get_env_rank_and_size


def train():
    return get_env_rank_and_size()


@unittest.skipIf(not gloo_built(), "Gloo is not available")
class ElasticRunTests(unittest.TestCase):
    """
    Tests for run api with elastic config.
    """

    def test_run_with_hosts(self):
        """Tests two usable hosts, two slots each in standard happy path."""
        hosts = 'localhost:2,127.0.0.1:2'
        results = horovod.run(train, num_proc=2, min_num_proc=2, max_num_proc=2, hosts=hosts)
        self.assertEqual([(0, 2), (1, 2)], results)

    def test_run_with_discovery_script(self):
        """Tests two usable hosts, two slots each via discovery script in standard happy path."""
        with NamedTemporaryFile(mode='w') as script:
            script.write('echo "localhost:2"\n')
            script.write('echo "127.0.0.1:2"\n')
            script.file.close()
            os.chmod(script.name, 0o700)

            results = horovod.run(train, num_proc=2, min_num_proc=2, max_num_proc=2, host_discovery_script=script.name)

        self.assertEqual([(0, 2), (1, 2)], results)


if __name__ == "__main__":
    unittest.main()
