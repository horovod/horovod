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
from typing import List, Callable

import tensorflow as tf

import horovod.tensorflow as hvd
from horovod.runner.util.threads import in_thread
from horovod.tensorflow.data.compute_service import TfDataServiceConfig
from horovod.tensorflow.data.compute_worker import main

_PRE_TF_2_0_0 = LooseVersion(tf.__version__) < LooseVersion("2.0.0")


# this test is to be run via horovodrun -np 2, all processes have to run on the same machine
class ComputeWorkerTest(unittest.TestCase):
    # general timeout in this test
    timeout = 10

    # rank and size of this test
    hvd.init()
    rank, size = hvd.rank(), hvd.size()

    @property
    def expected_cluster_shape(self):
        return [(r, self.size) for r in range(self.size)]

    def test_single_dispatcher(self):
        def assert_batches(actuals: List[List[List[int]]]):
            # batches are not deterministic, all we can say is the shape of the batches
            # and each number in [0..15] appears once
            self.assertEqual(self.size, len(actuals))
            for actual in actuals:
                self.assertEqual(4, len(actual))
                self.assertEqual([4, 4, 4, 4], [len(batch) for batch in actual])
                self.assertEqual({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
                                 {i for batch in actual for i in batch})

        self.do_test_worker(1, reuse_dataset=False, round_robin=False, assert_batches=assert_batches)

    def test_single_dispatcher_reuse_dataset_fcfs(self):
        def assert_batches(actuals: List[List[List[int]]]):
            # batches are not deterministic, all we can say is all but the last batch have length 4,
            # the last has at most 4, all numbers in [0..15] appear once across all workers
            for actual in actuals:
                if len(actual) > 0:
                    self.assertEqual([4] * (len(actual) - 1), [len(batch) for batch in actual[:-1]])
                    self.assertTrue(len(actual[-1]) <= 4)

            actuals = [i for actual in actuals for batch in actual for i in batch]
            self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], sorted(actuals))

        self.do_test_worker(1, reuse_dataset=True, round_robin=False, assert_batches=assert_batches)

    def test_single_dispatcher_reuse_dataset_round_robin(self):
        def assert_batches(actuals: List[List[List[int]]]):
            # batches are not deterministic, all we can say is the shape of the batches
            # and each number in [0..15] appears once across all workers
            for actual in actuals:
                self.assertEqual(2, len(actual))
                self.assertEqual([4, 4], [len(batch) for batch in actual])

            actuals = [i for actual in actuals for batch in actual for i in batch]
            self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], sorted(actuals))

        self.do_test_worker(1, reuse_dataset=True, round_robin=True, assert_batches=assert_batches)

    def test_two_dispatchers(self):
        def assert_batches(actuals: List[List[List[int]]]):
            for actual in actuals:
                self.assertEqual([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], actual)

        self.do_test_worker(2, reuse_dataset=False, round_robin=False, assert_batches=assert_batches)

    def do_test_worker(self,
                       dispatchers: int,
                       reuse_dataset: bool,
                       round_robin: bool,
                       assert_batches: Callable[[List[List[List[int]]]], None]):
        self.do_test_worker_compute_side(dispatchers, reuse_dataset=reuse_dataset, round_robin=round_robin, assert_batches=assert_batches)

    def do_test_worker_compute_side(self,
                                    dispatchers: int,
                                    reuse_dataset: bool,
                                    round_robin: bool,
                                    assert_batches: Callable[[List[List[List[int]]]], None]):
        # synchronize with all processes
        print('waiting for all processes to get started')
        self.assertTrue(self.size > 1)
        cluster_shape = hvd.allgather_object((self.rank, self.size), name='test_start')
        self.assertEqual(self.expected_cluster_shape, cluster_shape)
        print('all processes started')

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
                dataset = tf.data.Dataset.range(16)
                if round_robin:
                    dataset = dataset.repeat()
                dataset = dataset.batch(4) \
                    .send_to_data_service(compute_config, self.rank, self.size,
                                          reuse_dataset=reuse_dataset,
                                          round_robin=round_robin)

                # sync again before fetching dataset
                hvd.allgather_object((self.rank, self.size), name='test_start')

                # fetch the batches
                it = dataset.as_numpy_iterator()
                if round_robin:
                    it = islice(it, 2)

                actual = list([batch.tolist() for batch in it])

                # synchronize with all processes
                print('waiting for all processes to finish')
                actuals = hvd.allgather_object(actual)
                print('all processes finished')

                # assert the provided batches
                self.assertEqual(self.size, len(actuals))
                assert_batches(actuals)

            finally:
                # shutdown compute service
                if self.rank == 0:
                    compute = compute_config.compute_client(verbose=2)
                    compute.shutdown()

                # wait for the worker to terminate
                worker.join(self.timeout)
                #self.assertFalse(worker.is_alive())

        finally:
            # remove the configfile as it will interfere with subsequent runs of this test
            if self.rank == 0:
                os.unlink(configfile)
