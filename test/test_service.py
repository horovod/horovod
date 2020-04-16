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
import threading
import time
import warnings

from six.moves import queue

import pytest
import mock
from mock import MagicMock

import horovod
from horovod.run.common.service.task_service import BasicTaskClient, BasicTaskService
from horovod.run.common.util import secret


class NetworkTests(unittest.TestCase):
    """
    Tests for horovod.run.common.service.
    """

    def __init__(self, *args, **kwargs):
        super(NetworkTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_concurrent_requests(self):
        """
        Tests the BasicTaskService can handle concurrent requests, as well as
        the BasicTaskClient can sent requests from multiple threads.
        A client executes a long running command and waits for termination.
        That wait is a long running request. During wait a separate thread
        polls the service for termination state.
        """
        def check_terminated(client, events):
            (started, terminated, shutdown) = events
            requests = 0
            command_terminated = None

            while not shutdown.is_set():
                requests += 1
                command_terminated = client.command_terminated()
                if requests > 10:
                    started.set()
                if command_terminated:
                    terminated.set()
                    break

            print('client contacted the service {} times'.format(requests))
            self.assertTrue(command_terminated, msg='command should have terminated')
            self.assertGreater(requests, 10, msg='we should have done at least some requests')
            self.assertTrue(shutdown.wait(10.0), msg='main thread should have finished by now')

        for multi_threaded_client in [False, True]:
            key = secret.make_secret_key()
            service_name = 'test-service'
            service = BasicTaskService(service_name, key, nics=None, verbose=2)

            client = BasicTaskClient(service_name, service.addresses(), key, verbose=2, attempts=1)
            parallel_client = client if multi_threaded_client \
                else BasicTaskClient(service_name, service.addresses(), key, verbose=2, attempts=1)

            started = threading.Event()
            terminated = threading.Event()
            shutdown = threading.Event()
            events = (started, terminated, shutdown)
            thread = threading.Thread(target=check_terminated, args=(parallel_client, events))
            thread.start()
            self.assertTrue(started.wait(1.0), msg='thread should have started by now')

            try:
                client.run_command('sleep 1', env={})
                client.wait_for_command_termination()
                self.assertTrue(terminated.wait(1.0), msg='thread should have recognized termination')
                self.assertTrue(thread.is_alive(), msg='thread should still be alive, check exceptions')
            finally:
                shutdown.set()
                thread.join(1.0)

            self.assertFalse(thread.is_alive(), msg='thread did not exit on term event')

    def test_shutdown_during_request(self):
        result_queue = queue.Queue(1)

        def wait_for_exit_code(client, queue):
            queue.put(client.wait_for_command_exit_code())

        key = secret.make_secret_key()
        service_name = 'test-service'
        service = BasicTaskService(service_name, key, nics=None, verbose=2)
        client = BasicTaskClient(service_name, service.addresses(), key, verbose=2, attempts=1)
        thread = threading.Thread(target=wait_for_exit_code, args=(client, result_queue))

        start = time.time()
        thread.start()  # wait for command exit code
        client.run_command('sleep 2', {})  # execute command
        time.sleep(0.5)  # give the thread some time to connect before shutdown
        service.shutdown()  # shutdown should wait on request to finish

        # we cannot call after shutdown
        with pytest.raises(Exception, match='^\\[Errno 104\\] Connection reset by peer$'):
            client.command_result()

        # but still our long running request succeeded
        thread.join(1)
        duration = time.time() - start
        self.assertGreater(duration, 2)

    def test_exit_code(self):
        """test non-zero exit code"""
        key = secret.make_secret_key()
        service_name = 'test-service'
        service = BasicTaskService(service_name, key, nics=None, verbose=2)
        client = BasicTaskClient(service_name, service.addresses(), key, verbose=2, attempts=1)

        client.run_command('false', {})
        res = client.wait_for_command_exit_code()
        self.assertEqual(1, res)
