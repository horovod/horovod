# Copyright 2022 G-Research, Inc. All Rights Reserved.
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
from distutils.version import LooseVersion
from itertools import islice
from typing import List
import tensorflow as tf

from horovod.runner.common.util import env
from horovod.runner.util.threads import in_thread
from horovod.tensorflow.data.compute_service import TfDataServiceConfig
from horovod.tensorflow.data.compute_worker import main

_PRE_TF_2_0_0 = LooseVersion(tf.__version__) < LooseVersion("2.0.0")


# this test is to be run via horovodrun -np 2, all processes have to run on the same machine
class ComputeWorkerTest(unittest.TestCase):
    # general timeout in this test
    timeout = 10

    # rank and size of this test
    rank, size = env.get_env_rank_and_size()

    def test_single_dispatcher(self):
        self.do_test_worker(1, reuse_dataset=False, round_robin=False,
                            expected_batches=[[[i for i in range(batch*128, (batch+1)*128)]
                                               for batch in range(8)]] * self.size)

    def test_single_dispatcher_reuse_dataset(self):
        self.do_test_worker(1, reuse_dataset=True, round_robin=False, expected_batches=[[[]]] * self.size)

    def test_single_dispatcher_reuse_dataset_round_robin(self):
        self.do_test_worker(1, reuse_dataset=True, round_robin=True, expected_batches=[[[]]] * self.size)

    def test_two_dispatchers_compute_side(self):
        self.do_test_worker(2, reuse_dataset=True, round_robin=True, expected_batches=[[[]]] * self.size)

    def do_test_worker(self,
                       dispatchers: int,
                       reuse_dataset: bool,
                       round_robin: bool,
                       expected_batches: List[List[List[int]]]):
        self.do_test_worker_compute_side(dispatchers, reuse_dataset=reuse_dataset, round_robin=round_robin,
                                         expected_batches=expected_batches)

    def do_test_worker_compute_side(self,
                                    dispatchers: int,
                                    reuse_dataset: bool,
                                    round_robin: bool,
                                    expected_batches: List[List[List[int]]]):
        # check expected batches have right size
        self.assertEqual(self.size, len(expected_batches), msg=f'expected batches must have length of {self.size}: {len(expected_batches)}')

        # the config file for this worker
        configfile = __file__ + '.config'
        if self.rank == 0 and os.path.exists(configfile):
            raise RuntimeError(f'Config file exists already, please delete first: {configfile}')

        try:
            # start the worker
            worker = in_thread(main, (dispatchers, 'compute', configfile, self.timeout), daemon=True)

            # read the config file
            compute_config = TfDataServiceConfig.read(configfile, wait_for_file_creation=True)

            try:
                # Allow tf.data service to pre-process the pipeline
                dataset = tf.data.Dataset.range(1024)
                if round_robin:
                    dataset = dataset.repeat()
                dataset = dataset.batch(128) \
                    .send_to_data_service(compute_config, self.rank, self.size,
                                          reuse_dataset=reuse_dataset,
                                          round_robin=round_robin) \
                    .prefetch(4)

                # fetch the batches
                it = dataset.as_numpy_iterator()
                if round_robin:
                    it = islice(it, 8)

                self.assertEqual(expected_batches[self.rank], [batch.tolist() for batch in it])

            finally:
                # shutdown compute service
                if self.rank == 0:
                    compute = compute_config.compute_client(verbose=2)
                    compute.shutdown()

            # wait for the worker to terminate
            worker.join(self.timeout)
            self.assertFalse(worker.is_alive())

        finally:
            if self.rank == 0:
                os.unlink(configfile)
