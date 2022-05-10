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
import logging
import os
import unittest
from distutils.version import LooseVersion
from itertools import islice
from typing import List, Callable

import tensorflow as tf

import horovod.tensorflow as hvd
from horovod.runner.util.threads import in_thread
from horovod.tensorflow.data.compute_service import TfDataServiceConfig, tf_data_service
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
        with self.subTest(msg='dispatcher on compute side'):
            self.do_test_worker_compute_side(dispatchers, reuse_dataset=reuse_dataset, round_robin=round_robin, assert_batches=assert_batches)
        with self.subTest(msg='dispatcher on training side'):
            self.do_test_worker_training_side(dispatchers, reuse_dataset=reuse_dataset, round_robin=round_robin, assert_batches=assert_batches)

    # keep this in-sync with do_test_worker_training_side
    def do_test_worker_compute_side(self,
                                    dispatchers: int,
                                    reuse_dataset: bool,
                                    round_robin: bool,
                                    assert_batches: Callable[[List[List[List[int]]]], None]):
        # the config file for this worker
        configfile = __file__ + '.config'
        if self.rank == 0 and os.path.exists(configfile):
            raise RuntimeError(f'Config file exists already, please delete first: {configfile}')

        # synchronize with all processes
        self.assertTrue(self.size > 1)
        logging.debug('waiting for all processes to get started')
        cluster_shape = hvd.allgather_object((self.rank, self.size), name='test_start')
        self.assertEqual(self.expected_cluster_shape, cluster_shape)
        logging.debug('all processes started')

        try:
            # start the worker
            logging.debug('starting worker process')
            worker = in_thread(main, (dispatchers, 'compute', configfile, self.timeout), daemon=True)
            # this runs 'main' as a separated process
            #command = f'{sys.executable} -m horovod.tensorflow.data.compute_worker --dispatchers {dispatchers} --dispatcher-side compute {configfile}'
            #worker = in_thread(safe_shell_exec.execute, (command, None, sys.stdout, sys.stderr), daemon=True)
            logging.debug('worker process started')

            # read the config file
            compute_config = TfDataServiceConfig.read(configfile, wait_for_file_creation=True)

            try:
                # Allow tf.data service to pre-process the pipeline
                dataset = tf.data.Dataset.range(1024)
                if reuse_dataset and round_robin:
                    dataset = dataset.repeat()
                dataset = dataset.batch(128) \
                    .send_to_data_service(compute_config, self.rank, self.size,
                                          reuse_dataset=reuse_dataset,
                                          round_robin=round_robin)

                # fetch the batches
                it = dataset.as_numpy_iterator()
                if round_robin:
                    it = islice(it, 8)
                actual = list([batch.tolist() for batch in it])

                # synchronize with all processes
                logging.debug('waiting for all processes to finish')
                actuals = hvd.allgather_object(actual)
                logging.debug(actuals)
                logging.debug('all processes finished')

                # assert the provided batches
                # the batches are not deterministic, so we cannot assert them here too thoroughly
                # that would test tf.data service anyway, all we assert here is that worker and send_to_data_service
                # work together nicely and produce a consumable dataset
                self.assertEqual(self.size, len(actuals), msg="one 'actual batches' from each process")
                self.assertEqual([True] * self.size, [len(actual) > 0 for actual in actuals], msg='each process has at least one batch')
                for actual in actuals:
                    self.assertEqual([True] * len(actual), [0 < len(batch) <= 128 for batch in actual], msg=f'all batches are at most 128 in size: {[len(batch) for batch in actual]}')
                    for batch in actual:
                        self.assertEqual([True] * len(batch), [0 <= i < 1024 for i in batch], msg=f'values in batch must be within [0..1024): {batch}')

            finally:
                # shutdown compute service
                if self.rank == 0:
                    logging.debug('sending shutdown request')
                    compute = compute_config.compute_client(verbose=2)
                    compute.shutdown()
                    logging.debug('shutdown request sent')

                # wait for the worker to terminate
                logging.debug('waiting for worker to terminate')
                worker.join(self.timeout)

                # in round robin mode, the worker process does not terminate once stopped until some high timeout
                if reuse_dataset and round_robin:
                    logging.debug(f'worker still alive: {worker.is_alive()}')
                else:
                    self.assertFalse(worker.is_alive())
                    logging.debug('worker terminated')

        finally:
            # remove the configfile as it will interfere with subsequent runs of this test
            if self.rank == 0 and os.path.exists(configfile):
                os.unlink(configfile)

    # keep this in-sync with do_test_worker_compute_side
    def do_test_worker_training_side(self,
                                     dispatchers: int,
                                     reuse_dataset: bool,
                                     round_robin: bool,
                                     assert_batches: Callable[[List[List[List[int]]]], None]):
        # the config file for this worker
        configfile = __file__ + '.config'
        if self.rank == 0 and os.path.exists(configfile):
            raise RuntimeError(f'Config file exists already, please delete first: {configfile}')

        # synchronize with all processes
        self.assertTrue(self.size > 1)
        logging.debug('waiting for all processes to get started')
        cluster_shape = hvd.allgather_object((self.rank, self.size), name='test_start')
        self.assertEqual(self.expected_cluster_shape, cluster_shape)
        logging.debug('all processes started')

        try:
            # start the worker
            logging.debug('starting worker process')
            worker = in_thread(main, (dispatchers, 'training', configfile, self.timeout), daemon=True)
            # this runs 'main' as a separated process
            #command = f'{sys.executable} -m horovod.tensorflow.data.compute_worker --dispatchers {dispatchers} --dispatcher-side compute {configfile}'
            #worker = in_thread(safe_shell_exec.execute, (command, None, sys.stdout, sys.stderr), daemon=True)
            logging.debug('worker process started')

            # read the config file
            compute_config = TfDataServiceConfig.read(configfile, wait_for_file_creation=True)

            try:
                with tf_data_service(compute_config, hvd.rank()) as dispatcher_address:
                    # Allow tf.data service to pre-process the pipeline
                    dataset = tf.data.Dataset.range(1024)
                    if reuse_dataset and round_robin:
                        dataset = dataset.repeat()
                    dataset = dataset.batch(128) \
                        .apply(tf.data.experimental.service.distribute(
                            service=dispatcher_address,
                            processing_mode="distributed_epoch",
                            job_name='job' if reuse_dataset else None,
                            consumer_index=hvd.rank() if round_robin else None,
                            num_consumers=hvd.size() if round_robin else None))

                    # fetch the batches
                    it = dataset.as_numpy_iterator()
                    if round_robin:
                        it = islice(it, 8)
                    actual = list([batch.tolist() for batch in it])

                    # synchronize with all processes
                    logging.debug('waiting for all processes to finish')
                    actuals = hvd.allgather_object(actual)
                    logging.debug(actuals)
                    logging.debug('all processes finished')

                    # assert the provided batches
                    # the batches are not deterministic, so we cannot assert them here too thoroughly
                    # that would test tf.data service anyway, all we assert here is that worker and send_to_data_service
                    # work together nicely and produce a consumable dataset
                    self.assertEqual(self.size, len(actuals), msg="one 'actual batches' from each process")
                    self.assertEqual([True] * self.size, [len(actual) > 0 for actual in actuals], msg='each process has at least one batch')
                    for actual in actuals:
                        self.assertEqual([True] * len(actual), [0 < len(batch) <= 128 for batch in actual], msg=f'all batches are at most 128 in size: {[len(batch) for batch in actual]}')
                        for batch in actual:
                            self.assertEqual([True] * len(batch), [0 <= i < 1024 for i in batch], msg=f'values in batch must be within [0..1024): {batch}')

            finally:
                # shutdown compute service
                if self.rank == 0:
                    logging.debug('sending shutdown request')
                    compute = compute_config.compute_client(verbose=2)
                    compute.shutdown()
                    logging.debug('shutdown request sent')

                # wait for the worker to terminate
                logging.debug('waiting for worker to terminate')
                worker.join(self.timeout)

                # in round robin mode, the worker process does not terminate once stopped until some high timeout
                if reuse_dataset and round_robin:
                    logging.debug(f'worker still alive: {worker.is_alive()}')
                else:
                    self.assertFalse(worker.is_alive())
                    logging.debug('worker terminated')

        finally:
            # remove the configfile as it will interfere with subsequent runs of this test
            if self.rank == 0 and os.path.exists(configfile):
                os.unlink(configfile)
