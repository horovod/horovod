# Copyright 2021 Uber Technologies, Inc. All Rights Reserved.
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
import re
import unittest

from horovod.runner.common.service.task_service import BasicTaskService, BasicTaskClient
from horovod.runner.common.util import secret


class FaultyStream:
    """This stream raises an exception after some text has been written."""
    def __init__(self, stream):
        self.stream = stream
        self.raised = False

    def write(self, b):
        if not self.raised and len(self.stream.getvalue()) > 1024:
            self.raised = True
            raise RuntimeError()
        self.stream.write(b)

    def close(self):
        pass


class TaskServiceTest(unittest.TestCase):

    cmd = 'for i in {1..10000}; do echo "a very very useful log line #$i"; done'
    cmd_single_line = f'{cmd} | wc'

    @staticmethod
    def cmd_with(stdout, stderr):
        return f"bash -c '{stderr} >&2 & {stdout}'"

    def test_run_command(self):
        key = secret.make_secret_key()
        service = BasicTaskService('test service', 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient('test service', service.addresses(), key, verbose=2, attempts=1)
            client.run_command(self.cmd_with(self.cmd_single_line, self.cmd_single_line), {})
            exit = client.wait_for_command_exit_code()
            self.assertEqual(0, exit)
            self.assertEqual((True, 0), client.command_result())
        finally:
            service.shutdown()

    def test_stream_command_output(self):
        self.do_test_stream_command_output(
            self.cmd_with(self.cmd, self.cmd),
            capture_stdout=True, capture_stderr=True,
            prefix_output_with_timestamp=True
        )

    def test_stream_command_output_stdout(self):
        self.do_test_stream_command_output(
            self.cmd_with(self.cmd, self.cmd_single_line),
            capture_stdout=True, capture_stderr=False,
            prefix_output_with_timestamp=True
        )

    def test_stream_command_output_stderr(self):
        self.do_test_stream_command_output(
            self.cmd_with(self.cmd_single_line, self.cmd),
            capture_stdout=False, capture_stderr=True,
            prefix_output_with_timestamp=True
        )

    def test_stream_command_output_neither(self):
        self.do_test_stream_command_output(
            self.cmd_with(self.cmd_single_line, self.cmd_single_line),
            capture_stdout=False, capture_stderr=False,
            prefix_output_with_timestamp=True
        )

    def test_stream_command_output_un_prefixed(self):
        self.do_test_stream_command_output(
            self.cmd_with(self.cmd, self.cmd),
            capture_stdout=True, capture_stderr=True,
            prefix_output_with_timestamp=False
        )

    def do_test_stream_command_output(self,
                                      command,
                                      capture_stdout, capture_stderr,
                                      prefix_output_with_timestamp):
        stdout = io.StringIO()
        stderr = io.StringIO()

        key = secret.make_secret_key()
        service = BasicTaskService('test service', 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient('test service', service.addresses(), key, verbose=2, attempts=1)
            stdout_t, stderr_t = client.stream_command_output(stdout, stderr)
            client.run_command(command, {},
                               capture_stdout=capture_stdout, capture_stderr=capture_stderr,
                               prefix_output_with_timestamp=prefix_output_with_timestamp)
            client.wait_for_command_termination(delay=0.2)
            self.assertEqual((True, 0), client.command_result())

            if stdout_t is not None:
                stdout_t.join(1.0)
                self.assertEqual(False, stdout_t.is_alive())
            if stderr_t is not None:
                stderr_t.join(1.0)
                self.assertEqual(False, stderr_t.is_alive())
        finally:
            service.shutdown()

        stdout = stdout.getvalue()
        stderr = stderr.getvalue()

        # remove timestamps from each line in outputs
        if prefix_output_with_timestamp:
            stdout_no_ts = re.sub('^[^[]+', '', stdout, flags=re.MULTILINE)
            stderr_no_ts = re.sub('^[^[]+', '', stderr, flags=re.MULTILINE)
            # test we are removing something (hopefully timestamps)
            if capture_stdout:
                self.assertNotEqual(stdout_no_ts, stdout)
            if capture_stderr:
                self.assertNotEqual(stderr_no_ts, stderr)
            stdout = stdout_no_ts
            stderr = stderr_no_ts

        # remove prefix
        stdout_no_prefix = re.sub('\[0\]<stdout>:', '', stdout, flags=re.MULTILINE)
        stderr_no_prefix = re.sub('\[0\]<stderr>:', '', stderr, flags=re.MULTILINE)
        # test we are removing something (hopefully prefixes)
        if capture_stdout:
            self.assertNotEqual(stdout_no_prefix, stdout)
        if capture_stderr:
            self.assertNotEqual(stderr_no_prefix, stderr)
        stdout = stdout_no_prefix
        stderr = stderr_no_prefix

        if capture_stdout and capture_stderr:
            # both streams should be equal
            self.assertEqual(stdout, stderr)

        # streams should have meaningful number of lines and characters
        if capture_stdout:
            self.assertTrue(len(stdout) > 1024)
            self.assertTrue(len(stdout.splitlines()) > 10)
        if capture_stderr:
            self.assertTrue(len(stderr) > 1024)
            self.assertTrue(len(stderr.splitlines()) > 10)

    def test_stream_command_output_reconnect(self):
        self.do_test_stream_command_output_reconnect(attempts=3, succeeds=True)

    def test_stream_command_output_no_reconnect(self):
        self.do_test_stream_command_output_reconnect(attempts=1, succeeds=None)

    def do_test_stream_command_output_reconnect(self, attempts, succeeds):
        key = secret.make_secret_key()
        stdout = io.StringIO()
        stderr = io.StringIO()

        stdout_s = FaultyStream(stdout)
        stderr_s = FaultyStream(stderr)
        service = BasicTaskService('test service', 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient('test service', service.addresses(), key, verbose=2, attempts=attempts)
            stdout_t, stderr_t = client.stream_command_output(stdout_s, stderr_s)
            client.run_command(self.cmd_with(self.cmd, self.cmd), {},
                               capture_stdout=True, capture_stderr=True,
                               prefix_output_with_timestamp=False)
            client.wait_for_command_termination(delay=0.2)
            terminated, exit = client.command_result()
            self.assertEqual(True, terminated)

            if succeeds is not None:
                self.assertEqual(succeeds, exit == 0)

            if stdout_t is not None:
                stdout_t.join(1.0)
                self.assertEqual(False, stdout_t.is_alive())
            if stderr_t is not None:
                stderr_t.join(1.0)
                self.assertEqual(False, stderr_t.is_alive())
        finally:
            service.shutdown()

        stdout = stdout.getvalue()
        stderr = stderr.getvalue()

        # we are likely to loose some lines, so output is hard to evaluate
        if succeeds:
            self.assertGreaterEqual(len(stdout), 1024)
            self.assertGreater(len(stdout.splitlines()), 10)
            self.assertTrue(stdout_s.raised)

            self.assertGreaterEqual(len(stderr), 1024)
            self.assertGreater(len(stderr.splitlines()), 10)
            self.assertTrue(stderr_s.raised)

            # assert stdout and stderr similarity (how many lines both have in common)
            stdout = re.sub('\[0\]<stdout>:', '', stdout, flags=re.MULTILINE)
            stderr = re.sub('\[0\]<stderr>:', '', stderr, flags=re.MULTILINE)
            stdout_set = set(stdout.splitlines())
            stderr_set = set(stderr.splitlines())
            intersect = stdout_set.intersection(stderr_set)
            self.assertGreater(len(intersect) / min(len(stdout_set), len(stderr_set)), 0.90)
        else:
            # we might have retrieved data only for one of stdout and stderr
            # so we expect some data for at least one of them
            self.assertGreaterEqual(len(stdout) + len(stderr), 1024)
            self.assertGreater(len(stdout.splitlines()) + len(stderr.splitlines()), 10)
            self.assertTrue(stdout_s.raised or stderr_s.raised)
