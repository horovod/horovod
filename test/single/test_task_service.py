import io
import re
import unittest

from horovod.runner.common.service.task_service import BasicTaskService, BasicTaskClient
from horovod.runner.common.util import secret


class TaskServiceTest(unittest.TestCase):

    def test_run_command(self):
        key = secret.make_secret_key()
        service = BasicTaskService('test service', 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient('test service', service.addresses(), key, verbose=2, attempts=1)
            client.run_command('find ../.. | wc', {})
            exit = client.wait_for_command_exit_code()
            self.assertEqual(0, exit)
            self.assertEqual((True, 0), client.command_result())
        finally:
            service.shutdown()

    cmd_with_stdout = 'find ../.. | sort'
    cmd_with_stdout_and_stderr = f'bash -c "{cmd_with_stdout} >&2 & {cmd_with_stdout}"'

    def test_stream_command_output(self):
        key = secret.make_secret_key()
        output = io.StringIO()
        service = BasicTaskService('test service', 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient('test service', service.addresses(), key, verbose=2, attempts=1)
            thread = client.stream_command_output(output)
            client.run_command(self.cmd_with_stdout_and_stderr, {}, capture=True)
            client.wait_for_command_termination(delay=0.2)
            thread.join(1.0)
            self.assertEqual(False, thread.is_alive())
            self.assertEqual((True, 0), client.command_result())
        finally:
            service.shutdown()

        # remove timestamps from each line in output
        output = re.sub('^[^[]+', '', output.getvalue(), flags=re.MULTILINE)
        # split output into stdout and stderr
        stdout = '\n'.join([line[12:]
                            for line in output.splitlines()
                            if line.startswith('[0]<stdout>')])
        stderr = '\n'.join([line[12:]
                            for line in output.splitlines()
                            if line.startswith('[0]<stderr>')])

        # both streams should be equal
        self.assertEqual(stdout, stderr)
        # streams should have meaningful number of lines and characters
        self.assertTrue(len(stdout) > 1024)
        self.assertTrue(len(stdout.splitlines()) > 10)

    def test_stream_command_output_reconnect(self):
        key = secret.make_secret_key()
        output = io.StringIO()

        class Stream:
            """This stream raises an exception after some text has been written."""
            raised = False

            def write(self, b):
                output.write(b)
                if not self.raised and len(output.getvalue()) > 1024:
                    print('raise exception')
                    self.raised = True
                    raise RuntimeError()

        stream = Stream()
        service = BasicTaskService('test service', 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient('test service', service.addresses(), key, verbose=2, attempts=3)
            thread = client.stream_command_output(stream)
            client.run_command(self.cmd_with_stdout_and_stderr, {}, capture=True)
            client.wait_for_command_termination(delay=0.2)
            thread.join(1.0)
            self.assertEqual(False, thread.is_alive())
            self.assertEqual((True, 0), client.command_result())
        finally:
            service.shutdown()

        # we are likely to loose some lines, so output is hard to evaluate
        output = output.getvalue()
        self.assertTrue(len(output) > 1024)
        self.assertTrue(len(output.splitlines()) > 10)
        self.assertTrue(stream.raised)

    def test_stream_command_output_stdout_no_reconnect(self):
        self.do_test_stream_command_output_no_reconnect(self.cmd_with_stdout)

    def test_stream_command_output_stdboth_no_reconnect(self):
        self.do_test_stream_command_output_no_reconnect(self.cmd_with_stdout_and_stderr)

    def do_test_stream_command_output_no_reconnect(self, command):
        key = secret.make_secret_key()
        output = io.StringIO()

        class Stream:
            """This stream raises an exception after some text has been written."""
            raised = False

            def write(self, b):
                output.write(b)
                if not self.raised and len(output.getvalue()) > 1024:
                    self.raised = True
                    raise RuntimeError()

        stream = Stream()
        service = BasicTaskService('test service', 0, key, nics=None, verbose=2)
        try:
            client = BasicTaskClient('test service', service.addresses(), key, verbose=2, attempts=1)
            thread = client.stream_command_output(stream)
            client.run_command(command, {}, capture=True)
            client.wait_for_command_termination(delay=0.2)
            thread.join(1.0)
            self.assertEqual(False, thread.is_alive())
            terminated, exit = client.command_result()
            self.assertEqual(True, terminated)
            self.assertTrue(exit != 0)
        finally:
            service.shutdown()

        # we are likely to loose some lines, so output is hard to evaluate
        output = output.getvalue()
        self.assertTrue(len(output) > 1024)
        self.assertTrue(len(output.splitlines()) > 10)
        self.assertTrue(stream.raised)
