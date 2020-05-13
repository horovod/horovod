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
import sys
import warnings

from distutils.version import LooseVersion
from six.moves import queue

import pytest

from horovod.run.common.service.task_service import BasicTaskClient, BasicTaskService
from horovod.run.common.util import network, secret
from horovod.run.util.threads import in_thread


class SleepRequest(object):
    pass


class TestSleepService(network.BasicService):
    def __init__(self, key, duration):
        super(TestSleepService, self).__init__('test service', key, nics=None)
        self._duration = duration

    def _handle(self, req, client_address):
        if isinstance(req, SleepRequest):
            print('{}: sleeping for client {}'.format(time.time(), client_address))
            time.sleep(self._duration)
            return network.AckResponse()

        return super(TestSleepService, self)._handle(req, client_address)


class TestSleepClient(network.BasicClient):
    def __init__(self, service_addresses, key, attempts=1):
        super(TestSleepClient, self).__init__('test service',
                                              service_addresses,
                                              key,
                                              verbose=2,
                                              attempts=attempts)

    def sleep(self):
        self._send(SleepRequest())


class NetworkTests(unittest.TestCase):
    """
    Tests for horovod.run.common.service.
    """

    def __init__(self, *args, **kwargs):
        super(NetworkTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_concurrent_requests_basic(self):
        sleep = 2.0
        key = secret.make_secret_key()
        service = TestSleepService(key, duration=sleep)
        client = TestSleepClient(service.addresses(), key, attempts=1)

        start = time.time()
        threads = list([in_thread(client.sleep, daemon=False) for _ in range(1)])
        for thread in threads:
            thread.join(sleep + 1.0)
            self.assertFalse(thread.is_alive(), 'thread should have terminated by now')
        duration = time.time() - start
        print('concurrent requests completed in {} seconds'.format(duration))

        self.assertGreaterEqual(duration, sleep, 'sleep requests should have been completed')
        self.assertLess(duration, sleep + 1.0, 'sleep requests should have been concurrent')

    @pytest.mark.skipif(sys.version_info < (3,0),
                        reason='block on shutdown is supported in Spark 3.0 and above')
    def test_shutdown_during_request_basic(self):
        sleep = 2.0
        key = secret.make_secret_key()
        service = TestSleepService(key, duration=sleep)
        client = TestSleepClient(service.addresses(), key, attempts=1)

        start = time.time()
        threads = list([in_thread(client.sleep, name='request {}'.format(i+1), daemon=False) for i in range(5)])
        time.sleep(sleep / 2.0)
        service.shutdown()
        duration = time.time() - start
        print('shutdown completed in {} seconds'.format(duration))
        self.assertGreaterEqual(duration, sleep, 'sleep requests should have been completed')
        self.assertLess(duration, sleep + 1.0, 'sleep requests should have been concurrent')

        for thread in threads:
            thread.join(0.1)
            self.assertFalse(thread.is_alive(), 'thread should have terminated by now')

    @pytest.mark.skipif(sys.version_info < (3,0),
                        reason='block on shutdown is supported in Spark 3.0 and above')
    def test_shutdown_during_request_basic_task(self):
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
        service = BasicTaskService(service_name, key, nics=None, verbose=2)
        client = BasicTaskClient(service_name, service.addresses(), key, verbose=2, attempts=1)

        client.run_command('false', {})
        res = client.wait_for_command_exit_code()
        self.assertEqual(1, res)
