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
import unittest
from packaging import version
from queue import Queue

import tensorflow as tf

from horovod.runner.common.service.compute_service import ComputeService, ComputeClient
from horovod.runner.common.util import secret
from horovod.runner.common.util.timeout import TimeoutException
from horovod.runner.util.threads import in_thread

_PRE_TF_2_0_0 = version.parse(tf.__version__) < version.parse("2.0.0")


class ComputeServiceTest(unittest.TestCase):

    @staticmethod
    def wait_for_dispatcher(client, dispatcher_id, queue):
        def _wait():
            queue.put((dispatcher_id, client.wait_for_dispatcher_registration(dispatcher_id, 10)))
        return in_thread(_wait, daemon=True)

    @staticmethod
    def wait_for_dispatcher_workers(client, dispatcher_id, queue):
        def _wait():
            client.wait_for_dispatcher_worker_registration(dispatcher_id, 10)
            queue.put(dispatcher_id)
        return in_thread(_wait, daemon=True)

    @staticmethod
    def wait_for_shutdown(client, queue):
        def _wait():
            client.wait_for_shutdown()
            queue.put(True)
        return in_thread(_wait, daemon=True)

    @staticmethod
    def get_all(queue):
        while not queue.empty():
            yield queue.get_nowait()

    def test_good_path(self):
        for dispatchers_num, workers_per_dispatcher in [
            (1, 1), (1, 2), (1, 4),
            (2, 1), (2, 2), (2, 4),
            (32, 16), (1, 512)
        ]:
            with self.subTest(dispatchers=dispatchers_num, workers_per_dispatcher=workers_per_dispatcher):
                key = secret.make_secret_key()
                service = ComputeService(dispatchers_num, workers_per_dispatcher, key, nics=None)
                try:
                    client = ComputeClient(service.addresses(), key, verbose=2)

                    # create thread waiting for shutdown
                    shutdown = Queue()
                    shutdown_thread = self.wait_for_shutdown(client, shutdown)

                    # dispatcher registration
                    # start threads that wait for dispatchers
                    threads = []
                    dispatchers = Queue()
                    for id in range(dispatchers_num):
                        threads.append(self.wait_for_dispatcher(client, id, dispatchers))

                    # register dispatchers
                    for id in range(dispatchers_num):
                        client.register_dispatcher(id, f'grpc://localhost:{10000+id}')

                    # check threads terminate
                    for thread in threads:
                        thread.join(10)
                        self.assertFalse(thread.is_alive(), msg="threads waiting for dispatchers did not terminate")

                    # check reported dispatcher addresses
                    self.assertEqual([(id, f'grpc://localhost:{10000+id}') for id in range(dispatchers_num)],
                                     sorted(self.get_all(dispatchers)))

                    # worker registration
                    # start threads to wait for dispatcher worker registration
                    threads = []
                    dispatchers = Queue()
                    for id in range(dispatchers_num):
                        threads.append(self.wait_for_dispatcher_workers(client, id, dispatchers))

                    # register dispatcher workers
                    for id in range(dispatchers_num * workers_per_dispatcher):
                        client.register_worker_for_dispatcher(dispatcher_id=id // workers_per_dispatcher, worker_id=id)

                    # check threads terminate
                    for thread in threads:
                        thread.join(10)
                        self.assertFalse(thread.is_alive(), msg="threads waiting for dispatchers' workers did not terminate")

                    # check reported dispatcher success
                    self.assertEqual(sorted(range(dispatchers_num)), sorted(self.get_all(dispatchers)))

                    # shutdown and wait for shutdown
                    self.assertTrue(shutdown_thread.is_alive(), msg="thread waiting for shutdown, terminated early")
                    client.shutdown()
                    shutdown_thread.join(10)
                    self.assertFalse(shutdown_thread.is_alive(), msg="thread waiting for shutdown did not terminate")
                    self.assertEqual([True], list(self.get_all(shutdown)))
                finally:
                    service.shutdown()

    def test_invalid_dispatcher_ids(self):
        key = secret.make_secret_key()
        service = ComputeService(2, 4, key, nics=None)
        try:
            client = ComputeClient(service.addresses(), key, verbose=2)

            with self.assertRaises(IndexError):
                client.register_dispatcher(-1, 'grpc://localhost:10000')
            with self.assertRaises(IndexError):
                client.register_dispatcher(2, 'grpc://localhost:10000')

            with self.assertRaises(IndexError):
                client.wait_for_dispatcher_registration(-1, 0.1)
            with self.assertRaises(IndexError):
                client.wait_for_dispatcher_registration(2, 0.1)

            with self.assertRaises(IndexError):
                client.register_worker_for_dispatcher(-1, 0)
            with self.assertRaises(IndexError):
                client.register_worker_for_dispatcher(2, 0)

            with self.assertRaises(IndexError):
                client.wait_for_dispatcher_worker_registration(-1, 0.1)
            with self.assertRaises(IndexError):
                client.wait_for_dispatcher_worker_registration(2, 0.1)
        finally:
            service.shutdown()

    def test_register_dispatcher_duplicate(self):
        key = secret.make_secret_key()
        service = ComputeService(2, 1, key, nics=None)
        try:
            client = ComputeClient(service.addresses(), key, verbose=2)
            client.register_dispatcher(0, 'grpc://localhost:10000')
            with self.assertRaisesRegex(ValueError, 'Dispatcher with id 0 has already been registered under '
                                                    'different address grpc://localhost:10000: grpc://localhost:10001'):
                client.register_dispatcher(0, 'grpc://localhost:10001')
        finally:
            service.shutdown()

    def test_register_dispatcher_replay(self):
        key = secret.make_secret_key()
        service = ComputeService(2, 1, key, nics=None)
        try:
            client = ComputeClient(service.addresses(), key, verbose=2)
            client.register_dispatcher(0, 'grpc://localhost:10000')
            client.wait_for_dispatcher_registration(0, timeout=2)

            # register the same dispatcher again should not interfere the registration of second dispatcher
            client.register_dispatcher(0, 'grpc://localhost:10000')
            with self.assertRaises(TimeoutException):
                client.wait_for_dispatcher_registration(1, timeout=2)

            # registering the second dispatcher completes registration
            client.register_dispatcher(1, 'grpc://localhost:10001')
            client.wait_for_dispatcher_registration(0, timeout=2)
            client.wait_for_dispatcher_registration(1, timeout=2)
        finally:
            service.shutdown()

    def test_register_dispatcher_worker_replay(self):
        key = secret.make_secret_key()
        service = ComputeService(1, 2, key, nics=None)
        try:
            client = ComputeClient(service.addresses(), key, verbose=2)
            client.register_dispatcher(0, 'grpc://localhost:10000')
            client.register_worker_for_dispatcher(0, 0)

            # register the same worker again should not complete the registration
            client.register_worker_for_dispatcher(0, 0)
            with self.assertRaises(TimeoutException):
                client.wait_for_dispatcher_worker_registration(0, timeout=2)

            # registering the second dispatcher completes registration
            client.register_worker_for_dispatcher(0, 1)
            client.wait_for_dispatcher_worker_registration(0, timeout=2)
        finally:
            service.shutdown()

    def test_register_dispatcher_timeout(self):
        key = secret.make_secret_key()
        service = ComputeService(1, 1, key, nics=None)
        try:
            client = ComputeClient(service.addresses(), key, verbose=2)
            with self.assertRaisesRegex(TimeoutException,
                                        expected_regex='Timed out waiting for dispatcher 0 to register. '
                                                       'Try to find out what takes the dispatcher so long '
                                                       'to register or increase timeout. Timeout after 0.1 seconds.'):
                client.wait_for_dispatcher_registration(0, timeout=0.1)
        finally:
            service.shutdown()

    def test_register_dispatcher_worker_timeout(self):
        key = secret.make_secret_key()
        service = ComputeService(1, 1, key, nics=None)
        try:
            client = ComputeClient(service.addresses(), key, verbose=2)
            with self.assertRaisesRegex(TimeoutException,
                                        expected_regex='Timed out waiting for workers for dispatcher 0 to register. '
                                                       'Try to find out what takes the workers so long '
                                                       'to register or increase timeout. Timeout after 0.1 seconds.'):
                client.wait_for_dispatcher_worker_registration(0, timeout=0.1)
        finally:
            service.shutdown()
