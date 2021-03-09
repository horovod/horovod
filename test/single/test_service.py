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

import io
import queue
import threading
import time
import unittest
import warnings

import pytest

from horovod.runner.common.service.task_service import BasicTaskClient, BasicTaskService
from horovod.runner.common.util import network, secret
from horovod.runner.util.threads import in_thread
from horovod.runner.util.streams import Pipe


class SleepRequest(object):
    pass


class TestSleepService(network.BasicService):
    def __init__(self, key, duration):
        super(TestSleepService, self).__init__('test sleep service', key, nics=None)
        self._duration = duration

    def _handle(self, req, client_address):
        if isinstance(req, SleepRequest):
            print('{}: sleeping for client {}'.format(time.time(), client_address))
            time.sleep(self._duration)
            return network.AckResponse()

        return super(TestSleepService, self)._handle(req, client_address)


class TestSleepClient(network.BasicClient):
    def __init__(self, service_addresses, key, attempts=1):
        super(TestSleepClient, self).__init__('test sleep service',
                                              service_addresses,
                                              key,
                                              verbose=2,
                                              attempts=attempts)

    def sleep(self):
        self._send(SleepRequest())


class TestStreamService(network.BasicService):
    def __init__(self, key, duration):
        super(TestStreamService, self).__init__('test stream service', key, nics=None)
        self._duration = duration

    def _handle(self, req, client_address):
        if isinstance(req, SleepRequest):
            pipe = Pipe()

            def sleep():
                time.sleep(self._duration)
                pipe.write('slept {}'.format(self._duration))
                pipe.close()

            in_thread(sleep)

            return network.AckStreamResponse(), pipe

        return super(TestStreamService, self)._handle(req, client_address)


class TestStreamClient(network.BasicClient):
    def __init__(self, service_addresses, key, attempts=1):
        super(TestStreamClient, self).__init__('test stream service',
                                               service_addresses,
                                               key,
                                               verbose=2,
                                               attempts=attempts)

    def sleep(self, stream):
        self._send(SleepRequest(), stream)


class NetworkTests(unittest.TestCase):
    """
    Tests for horovod.runner.common.service.
    """

    def __init__(self, *args, **kwargs):
        super(NetworkTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_concurrent_requests_basic(self):
        sleep = 2.0
        key = secret.make_secret_key()
        service = TestSleepService(key, duration=sleep)
        try:
            client = TestSleepClient(service.addresses(), key, attempts=1)
            start = time.time()
            threads = list([in_thread(client.sleep, daemon=False) for _ in range(1)])
            for thread in threads:
                thread.join(sleep + 1.0)
                self.assertFalse(thread.is_alive(), 'thread should have terminated by now')
            duration = time.time() - start
            print('concurrent requests completed in {} seconds'.format(duration))
        finally:
            service.shutdown()

        self.assertGreaterEqual(duration, sleep, 'sleep requests should have been completed')
        self.assertLess(duration, sleep + 1.0, 'sleep requests should have been concurrent')

    def test_shutdown_during_request_basic(self):
        sleep = 2.0
        key = secret.make_secret_key()
        service = TestSleepService(key, duration=sleep)
        try:
            client = TestSleepClient(service.addresses(), key, attempts=1)
            start = time.time()
            threads = list([in_thread(client.sleep, name='request {}'.format(i+1), daemon=False) for i in range(5)])
            time.sleep(sleep / 2.0)
        finally:
            service.shutdown()

        duration = time.time() - start
        print('shutdown completed in {} seconds'.format(duration))
        self.assertGreaterEqual(duration, sleep, 'sleep requests should have been completed')
        self.assertLess(duration, sleep + 1.0, 'sleep requests should have been concurrent')

        for thread in threads:
            thread.join(0.1)
            self.assertFalse(thread.is_alive(), 'thread should have terminated by now')

    def test_shutdown_during_request_basic_task(self):
        result_queue = queue.Queue(1)

        def wait_for_exit_code(client, queue):
            queue.put(client.wait_for_command_exit_code())

        key = secret.make_secret_key()
        service_name = 'test-service'
        service = BasicTaskService(service_name, 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient(service_name, service.addresses(), key, verbose=2, attempts=1)
            thread = threading.Thread(target=wait_for_exit_code, args=(client, result_queue))

            start = time.time()
            thread.start()  # wait for command exit code
            client.run_command('sleep 2', {})  # execute command
            time.sleep(0.5)  # give the thread some time to connect before shutdown
        finally:
            service.shutdown()  # shutdown should wait on request to finish

        duration = time.time() - start
        self.assertGreaterEqual(duration, 2)

        # we cannot call after shutdown
        with pytest.raises(Exception, match=r'^(\[[Ee]rrno 104\] Connection reset by peer)'
                                            r'|(\[[Ee]rrno 111\] Connection refused)$'):
            client.command_result()

        # but still our long running request succeeded
        thread.join(1.0)
        self.assertFalse(thread.is_alive())

    def test_exit_code(self):
        """test non-zero exit code"""
        key = secret.make_secret_key()
        service_name = 'test-service'
        service = BasicTaskService(service_name, 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient(service_name, service.addresses(), key, verbose=2, attempts=1)
            client.run_command('false', {})
            res = client.wait_for_command_exit_code()
            self.assertEqual(1, res)
        finally:
            service.shutdown()

    def test_stream(self):
        sleep = 2.0
        key = secret.make_secret_key()
        service = TestStreamService(key, duration=sleep)
        try:
            client = TestStreamClient(service.addresses(), key, attempts=1)

            start = time.time()
            stream = io.StringIO()
            client.sleep(stream)
            duration = time.time() - start

            self.assertEqual(f'slept {sleep}', stream.getvalue())
            self.assertGreaterEqual(duration, 2)
        finally:
            service.shutdown()
