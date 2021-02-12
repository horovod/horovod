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

import copy
import io
import itertools
import logging
import multiprocessing
import os
import subprocess
import sys
import threading
import time
import unittest
import warnings

import mock
import psutil
import pytest
from mock import MagicMock

import horovod
from horovod.runner import _HorovodArgs
from horovod.runner.common.util import config_parser, hosts, safe_shell_exec, secret, \
    settings as hvd_settings, timeout
from horovod.runner.common.util.host_hash import _hash, host_hash
from horovod.runner.common.util.hosts import SlotInfo, get_host_assignments, parse_hosts
from horovod.runner.gloo_run import gloo_run
from horovod.runner.js_run import js_run, generate_jsrun_rankfile
from horovod.runner.launch import gloo_built, parse_args, run_controller, _run
from horovod.runner.mpi_run import _get_mpi_implementation, _get_mpi_implementation_flags, \
    _LARGE_CLUSTER_THRESHOLD as large_cluster_threshold, mpi_available, mpi_run, \
    _OMPI_IMPL, _SMPI_IMPL, _MPICH_IMPL, _IMPI_IMPL, _UNKNOWN_IMPL, _MISSING_IMPL
from horovod.runner.util.threads import in_thread, on_event

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import is_built, lsf_and_jsrun, override_args, override_env, temppath, delay, wait


class RunTests(unittest.TestCase):
    """
    Tests for horovod.runner.
    """

    def __init__(self, *args, **kwargs):
        super(RunTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def test_params_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--fusion-threshold-mb', '10',
                           '--cycle-time-ms', '20',
                           '--cache-capacity', '512',
                           '--hierarchical-allreduce',
                           '--hierarchical-allgather'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_FUSION_THRESHOLD), str(10 * 1024 * 1024))
            self.assertEqual(env.get(config_parser.HOROVOD_CYCLE_TIME), '20.0')
            self.assertEqual(env.get(config_parser.HOROVOD_CACHE_CAPACITY), '512')
            self.assertEqual(env.get(config_parser.HOROVOD_HIERARCHICAL_ALLREDUCE), '1')
            self.assertEqual(env.get(config_parser.HOROVOD_HIERARCHICAL_ALLGATHER), '1')

    def test_autotune_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--autotune',
                           '--autotune-log-file', '/tmp/autotune.txt',
                           '--autotune-warmup-samples', '1',
                           '--autotune-steps-per-sample', '5',
                           '--autotune-bayes-opt-max-samples', '10',
                           '--autotune-gaussian-process-noise', '0.2'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_AUTOTUNE), '1')
            self.assertEqual(env.get(config_parser.HOROVOD_AUTOTUNE_LOG), '/tmp/autotune.txt')
            self.assertEqual(env.get(config_parser.HOROVOD_AUTOTUNE_WARMUP_SAMPLES), '1')
            self.assertEqual(env.get(config_parser.HOROVOD_AUTOTUNE_STEPS_PER_SAMPLE), '5')
            self.assertEqual(env.get(config_parser.HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES), '10')
            self.assertEqual(env.get(config_parser.HOROVOD_AUTOTUNE_GAUSSIAN_PROCESS_NOISE), '0.2')

    def test_autotuning_with_fixed_param(self):
        with override_args('horovodrun', '-np', '2',
                           '--autotune',
                           '--cache-capacity', '1024',
                           '--no-hierarchical-allgather'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertNotIn(config_parser.HOROVOD_FUSION_THRESHOLD, env)
            self.assertNotIn(config_parser.HOROVOD_CYCLE_TIME, env)
            self.assertEqual(env.get(config_parser.HOROVOD_CACHE_CAPACITY), '1024')
            self.assertNotIn(config_parser.HOROVOD_HIERARCHICAL_ALLREDUCE, env)
            self.assertEqual(env.get(config_parser.HOROVOD_HIERARCHICAL_ALLGATHER), '0')

    def test_timeline_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--timeline-filename', '/tmp/timeline.json',
                           '--timeline-mark-cycles'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_TIMELINE), '/tmp/timeline.json')
            self.assertEqual(env.get(config_parser.HOROVOD_TIMELINE_MARK_CYCLES), '1')

    def test_stall_check_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--no-stall-check'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_STALL_CHECK_DISABLE), '1')

        with override_args('horovodrun', '-np', '2',
                           '--stall-check-warning-time-seconds', '10',
                           '--stall-check-shutdown-time-seconds', '20'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertNotIn(config_parser.HOROVOD_STALL_CHECK_DISABLE, env)
            self.assertEqual(env.get(config_parser.HOROVOD_STALL_CHECK_TIME_SECONDS), '10')
            self.assertEqual(env.get(config_parser.HOROVOD_STALL_SHUTDOWN_TIME_SECONDS), '20')

    def test_library_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--mpi-threads-disable',
                           '--num-nccl-streams', '2',
                           '--thread-affinity', '1',
                           '--gloo-timeout-seconds', '60'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_MPI_THREADS_DISABLE), '1')
            self.assertEqual(env.get(config_parser.HOROVOD_NUM_NCCL_STREAMS), '2')
            self.assertEqual(env.get(config_parser.HOROVOD_THREAD_AFFINITY), '1')
            self.assertEqual(env.get(config_parser.HOROVOD_GLOO_TIMEOUT_SECONDS), '60')

    def test_library_env_override(self):
        """Tests that environment variables override arg defaults."""
        with override_args('horovodrun', '-np', '2'):
            args = parse_args()
            env = {
                'HOROVOD_GLOO_TIMEOUT_SECONDS': '1800',
            }
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_GLOO_TIMEOUT_SECONDS), '1800')

    def test_logging_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--log-level', 'INFO',
                           '--log-hide-timestamp'):
            args = parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_LOG_LEVEL), 'INFO')
            self.assertEqual(env.get(config_parser.HOROVOD_LOG_HIDE_TIME), '1')

    def test_config_file(self):
        config_filename = os.path.join(os.path.dirname(__file__), 'data/config.test.yaml')
        with override_args('horovodrun', '-np', '2',
                           '--config-file', config_filename):
            args = parse_args()

            self.assertTrue(args.use_gloo)

            # Params
            self.assertEqual(args.fusion_threshold_mb, 32)
            self.assertEqual(args.cycle_time_ms, 10)
            self.assertEqual(args.cache_capacity, 2048)
            self.assertTrue(args.hierarchical_allreduce)
            self.assertTrue(args.hierarchical_allgather)

            # Autotune
            self.assertTrue(args.autotune)
            self.assertEqual(args.autotune_log_file, 'horovod_autotune_log.txt')
            self.assertEqual(args.autotune_warmup_samples, 5)
            self.assertEqual(args.autotune_steps_per_sample, 20)
            self.assertEqual(args.autotune_bayes_opt_max_samples, 50)
            self.assertEqual(args.autotune_gaussian_process_noise, 0.9)

            # Timeline
            self.assertEqual(args.timeline_filename, 'horovod_timeline.json')
            self.assertTrue(args.timeline_mark_cycles)

            # Stall Check
            self.assertFalse(args.no_stall_check)
            self.assertEqual(args.stall_check_warning_time_seconds, 120)
            self.assertEqual(args.stall_check_shutdown_time_seconds, 240)

            # Library Options
            self.assertTrue(args.mpi_threads_disable)
            self.assertEqual(args.num_nccl_streams, 2)
            self.assertEqual(args.thread_affinity, 1)
            self.assertEqual(args.gloo_timeout_seconds, 60)

            # Logging
            self.assertEqual(args.log_level, 'INFO')
            self.assertFalse(args.log_with_timestamp)

    def test_config_file_override_args(self):
        config_filename = os.path.join(os.path.dirname(__file__), 'data/config.test.yaml')
        with override_args('horovodrun', '-np', '2',
                           '--fusion-threshold-mb', '256',
                           '--config-file', config_filename,
                           '--cycle-time-ms', '20',):
            args = parse_args()
            self.assertEqual(args.fusion_threshold_mb, 256)
            self.assertEqual(args.cycle_time_ms, 20)

    def test_validate_config_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--fusion-threshold-mb', '-1'):
            with pytest.raises(ValueError):
                parse_args()

    # test_on_event tests in_thread as well, but it does not test args
    def test_in_thread_args(self):
        fn = mock.Mock()
        thread = in_thread(fn, args=(1,))
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once_with(1)

        fn = mock.Mock()
        thread = in_thread(fn, args=(1, 2))
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once_with(1, 2)

        fn = mock.Mock()
        thread = in_thread(fn, args=(1, 2), silent=True)
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once_with(1, 2)

        fn = mock.Mock()
        with pytest.raises(ValueError, match="^args must be a tuple, not <(class|type) 'int'>, "
                                             "for a single argument use \\(arg,\\)$"):
            in_thread(fn, args=1)
        fn.assert_not_called()

    def test_on_event(self):
        # a happy run without args and stop event
        event = threading.Event()
        fn = mock.Mock()
        thread = on_event(event, fn)
        fn.assert_not_called()
        event.set()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once()

        # a happy run with args but without stop event
        event = threading.Event()
        fn = mock.Mock()
        thread = on_event(event, fn, ('a', 1))
        fn.assert_not_called()
        event.set()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once()
        fn.assert_called_once_with('a', 1)

        # a happy run with stop event but unused
        event = threading.Event()
        stop = threading.Event()
        fn = mock.Mock()
        thread = on_event(event, fn, stop=stop, check_stop_interval_s=0.01)
        fn.assert_not_called()
        event.set()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once()
        stop.set()
        time.sleep(0.1)
        fn.assert_called_once()

        # stop the thread before we set the event
        event = threading.Event()
        stop = threading.Event()
        fn = mock.Mock()
        thread = on_event(event, fn, stop=stop, check_stop_interval_s=0.01)
        fn.assert_not_called()
        stop.set()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_not_called()
        event.set()
        time.sleep(0.1)
        fn.assert_not_called()

        # test with exception
        def exception():
            raise Exception("Test Exception")

        event = threading.Event()
        fn = mock.Mock(side_effect=exception)
        thread = on_event(event, fn)
        fn.assert_not_called()
        event.set()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once()

        # test with exception but silent
        event = threading.Event()
        fn = mock.Mock(side_effect=exception)
        thread = on_event(event, fn)
        fn.assert_not_called()
        event.set()
        thread.join(1.0)
        self.assertFalse(thread.is_alive())
        fn.assert_called_once()

        # test None event
        event = None
        fn = mock.Mock()
        with pytest.raises(ValueError, match="^Event must not be None$"):
            on_event(event, fn)
        fn.assert_not_called()

        # test non-tuple args
        event = threading.Event()
        fn = mock.Mock()
        with pytest.raises(ValueError, match="^args must be a tuple, not <(class|type) 'int'>, "
                                             "for a single argument use \\(arg,\\)$"):
            on_event(event, fn, args=1)
        fn.assert_not_called()

        # test None stop and non-daemon
        event = threading.Event()
        fn = mock.Mock()
        with pytest.raises(ValueError, match="^Stop event must be given for non-daemon event thread$"):
            on_event(event, fn, stop=None, daemon=False)
        fn.assert_not_called()

    def test_prefix_connection(self):
        string = 'first line\nsecond line\nmore lines\n'
        self.do_test_prefix_connection(string, prefix='prefix', index=123,
                                       expected='[123]<prefix>:first line\n'
                                                '[123]<prefix>:second line\n'
                                                '[123]<prefix>:more lines\n')

    def test_prefix_connection_with_unicode(self):
        string = '∀∁∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑\n⌚⌛⛄✅…\n'
        self.do_test_prefix_connection(string, prefix='prefix', index=123,
                                       expected='[123]<prefix>:∀∁∂∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑\n'
                                                '[123]<prefix>:⌚⌛⛄✅…\n')

    def test_prefix_connection_with_multibyte_unicode(self):
        block = '█▓▒░▒▓'*1000
        string = block + '\n' + block + '\n'
        self.do_test_prefix_connection(string, prefix='prefix', index=123,
                                       expected='[123]<prefix>:' + block + '\n' +
                                                '[123]<prefix>:' + block + '\n')

    def test_prefix_connection_with_timestamp(self):
        string = 'first line\nsecond line\nmore lines\n'
        self.do_test_prefix_connection_with_timestamp(
            string, prefix='prefix', index=123,
            expected='Mon Jan 20 12:00:01 2020[123]<prefix>:first line\n'
                     'Mon Jan 20 12:00:02 2020[123]<prefix>:second line\n'
                     'Mon Jan 20 12:00:03 2020[123]<prefix>:more lines\n'
        )

    def test_prefix_connection_with_timestamp_without_index(self):
        string = 'first line\nsecond line\nmore lines\n'
        self.do_test_prefix_connection_with_timestamp(string, prefix=None, index=None,
                                                      expected=string)

    def test_prefix_connection_without_trailing_newline(self):
        string = 'first line\nsecond line\nmore lines'
        self.do_test_prefix_connection(string, prefix='prefix', index=123,
                                       expected='[123]<prefix>:first line\n'
                                            '[123]<prefix>:second line\n'
                                            '[123]<prefix>:more lines')

    def test_prefix_connection_without_index(self):
        string = 'first line\nsecond line\nmore lines\n'
        self.do_test_prefix_connection(string, prefix='prefix', index=None, expected=string)

    def test_prefix_connection_without_prefix(self):
        string = 'first line\nsecond line\nmore lines\n'
        self.do_test_prefix_connection(string, prefix=None, index=123, expected=string)

    def test_prefix_connection_with_carriage_return(self):
        string = 'first line\rfirst line again\nsecond line\n'
        self.do_test_prefix_connection(string, prefix='prefix', index=123,
                                       expected='[123]<prefix>:first line\r'
                                            '[123]<prefix>:first line again\n'
                                            '[123]<prefix>:second line\n')

    def test_prefix_connection_with_carriage_return_without_index(self):
        string = 'first line\rfirst line again\nsecond line\n'
        self.do_test_prefix_connection(string, prefix='prefix', index=None, expected=string)

    def test_prefix_connection_with_carriage_return_without_prefix(self):
        string = 'first line\rfirst line again\nsecond line\n'
        self.do_test_prefix_connection(string, prefix=None, index=123, expected=string)

    def do_test_prefix_connection(self, string, prefix, index, expected, timestamp=False):
        # create a Pipe Connection and populate it with string
        (connection, w) = multiprocessing.get_context('spawn').Pipe(duplex=False)
        with os.fdopen(w.fileno(), 'wt', encoding='utf8', newline='', closefd=False) as stream:
            stream.write(string)
        w.close()

        dst = io.StringIO()
        safe_shell_exec.prefix_connection(connection, dst, prefix=prefix, index=index,
                                          prefix_output_with_timestamp=timestamp)
        self.assertEqual(expected, dst.getvalue())

    def do_test_prefix_connection_with_timestamp(self, string_or_connection, prefix, index, expected):
        # control the time used to prepend the timestamp
        class MockTime:
            def __init__(self):
                # start at 2020-01-20 12:00:00
                self._time = 1579521600

            def time(self, seconds):
                from time import gmtime
                self._time = self._time + 1
                return gmtime(self._time)

        with mock.patch('horovod.runner.common.util.safe_shell_exec.time.localtime',
                        side_effect=MockTime().time):
            self.do_test_prefix_connection(string_or_connection, prefix, index, expected, timestamp=True)

    def test_prefix_connection_does_stream(self):
        index = 123
        prefix = 'prefix'
        expected = '[123]<prefix>:first line\r' \
                   '[123]<prefix>:first line again\n' \
                   '[123]<prefix>:second line\n' \
                   '[123]<prefix>:more lines'

        timeout = 0.2
        barrier = multiprocessing.Barrier(2, timeout=timeout)

        def writer(write_connection):
            def write(stream, text):
                stream.write(text)
                logging.info('wrote: {}'.format(text))
                stream.flush()

                # are we expecting the reader to read something
                logging.info('waiting for reader')
                if '\r' in text or '\n' in text:
                    # yes, barrier should not timeout
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        self.fail('reader side should have read something from the stream')
                else:
                    # no, barrier should timeout
                    try:
                        barrier.wait()
                        self.fail('reader side should not have read anything from the stream')
                    except threading.BrokenBarrierError:
                        logging.info('reader did not read')
                        barrier.reset()
                        pass
                logging.info('continuing writing')

            try:
                with os.fdopen(write_connection.fileno(), 'wt', encoding='utf8', newline='', closefd=False) as stream:
                    for text in ['first line\r',
                                 'first ', 'line ', 'again\n',
                                 'second line\nmore ', 'lines']:
                        write(stream, text)
            finally:
                write_connection.close()

        actual = []
        do_read = True

        def reader(read_connection):
            try:
                while do_read:
                    text = os.read(read_connection.fileno(), 1000)
                    if not text:
                        break
                    text = text.decode('utf8')
                    actual.append(text)
                    logging.info('read: {}'.format(text))
                    logging.info('waiting for writer')
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError as e:
                        if do_read:
                            raise e
                        break
                    logging.info('continuing reading')
            finally:
                read_connection.close()

        # one thread writes into the w side of this pipe
        # prefix_connection reads on the other end of this pipe
        (connection, w) = multiprocessing.get_context('spawn').Pipe(duplex=False)
        writer_thread = in_thread(writer, (w,))

        # prefix_connection writes to the write side of this Pipe (opened as a text stream)
        # another thread reads from the r side of this pipe
        (r, dst_con) = multiprocessing.get_context('spawn').Pipe(duplex=False)
        reader_thread = in_thread(reader, (r,))

        with os.fdopen(dst_con.fileno(), 'wt', encoding='utf8', newline='', closefd=False) as dst:
            safe_shell_exec.prefix_connection(connection, dst, prefix=prefix, index=index,
                                              prefix_output_with_timestamp=False)

        writer_thread.join(2*timeout)
        do_read = False
        reader_thread.join(2*timeout)

        self.assertEqual(expected, ''.join(actual))

    def test_safe_shell_exec_captures_stdout(self):
        self.do_test_safe_shell_exec('echo hello', 0, 'hello\n', '')

    def test_safe_shell_exec_captures_stderr(self):
        self.do_test_safe_shell_exec('echo hello >&2', 0, '', 'hello\n')

    def test_safe_shell_exec_captures_last_line_wo_eol(self):
        cmd = 'bash -c "echo -e -n \\"hello\nstdout\\"; echo -e -n \\"hello\nstderr\\" >&2"'
        self.do_test_safe_shell_exec(cmd, 0, 'hello\nstdout', 'hello\nstderr')

    def test_safe_shell_exec_returns_exit_code(self):
        self.do_test_safe_shell_exec('false', 1, '', '')

    @pytest.mark.skip(reason='https://github.com/horovod/horovod/issues/1993')
    def test_safe_shell_exec_interrupts_on_event(self):
        # interrupt execute in one second
        interrupt = threading.Event()
        interrupt_delay = 1.0
        delay(lambda: interrupt.set(), interrupt_delay)

        sleep = interrupt_delay + safe_shell_exec.GRACEFUL_TERMINATION_TIME_S + 2.0
        start = time.time()
        self.do_test_safe_shell_exec('sleep {}'.format(sleep), 143, '', None, interrupt)
        duration = time.time() - start

        self.assertGreaterEqual(duration, interrupt_delay)
        self.assertLess(duration, sleep - 1.0, 'sleep should not finish')

    def test_safe_shell_exec_interrupts_on_parent_shutdown(self):
        sleep = 20
        parent_script = os.path.join(os.path.dirname(__file__), 'data/run_safe_shell_exec.py')
        child_script = os.path.join(os.path.dirname(__file__), 'data/sleep.py')

        def get_pid(logfile):
            # Wait until the script has written its PID to the logfile
            wait(lambda: os.path.exists(logfile), timeout=5)
            with open(logfile, 'r') as f:
                return int(f.read())

        with temppath() as parent_logfile, temppath() as child_logfile:
            # It's important that this executes in an entirely different interpreter with as little shared
            # state as possible, to avoid issues with the semaphore tracker.
            cmd = ' '.join([sys.executable, parent_script, parent_logfile, child_script, str(sleep), child_logfile])
            p = subprocess.Popen(cmd, shell=True)

            parent = psutil.Process(get_pid(parent_logfile))
            child = psutil.Process(get_pid(child_logfile))

            self.assertTrue(parent.is_running())
            self.assertTrue(child.is_running())

            # Hard kill the parent process
            parent.kill()
            parent.wait(timeout=safe_shell_exec.GRACEFUL_TERMINATION_TIME_S)
            p.wait()

            # Child process will exit when pipe breaks
            child.wait(timeout=2 * safe_shell_exec.GRACEFUL_TERMINATION_TIME_S + 1)

            self.assertFalse(parent.is_running())
            self.assertFalse(child.is_running())

    def do_test_safe_shell_exec(self, cmd, expected_exit_code, expected_stdout, expected_stderr, event=None):
        stdout = io.StringIO()
        stderr = io.StringIO()
        res = safe_shell_exec.execute(cmd, stdout=stdout, stderr=stderr, events=[event] if event else None)
        self.assertEqual(expected_exit_code, res)
        if expected_stdout is not None:
            self.assertEqual(expected_stdout, stdout.getvalue())
        if expected_stderr is not None:
            self.assertEqual(expected_stderr, stderr.getvalue())

    def test_hash(self):
        hash = _hash("test string")
        self.assertEqual(hash, '6f8db599de986fab7a21625b7916589c')

    def test_host_hash(self):
        hash = host_hash()
        salted = host_hash('salt')
        empty_salted = host_hash('')

        self.assertNotEqual(salted, hash)
        self.assertEqual(empty_salted, hash)

    def test_get_mpi_implementation(self):
        def test(output, expected, exit_code=0):
            ret = (output, exit_code) if output is not None else None
            env = {'VAR': 'val'}
            with mock.patch("horovod.runner.mpi_run.tiny_shell_exec.execute", return_value=ret) as m:
                implementation = _get_mpi_implementation(env)
                self.assertEqual(expected, implementation)
                m.assert_called_once_with('mpirun --version', env)

        test(("mpirun (Open MPI) 2.1.1\n"
              "Report bugs to http://www.open-mpi.org/community/help/\n"), _OMPI_IMPL)

        test("OpenRTE", _OMPI_IMPL)

        test("IBM Spectrum MPI", _SMPI_IMPL)

        test(("HYDRA build details:\n"
              "    Version:           3.3a2\n"
              "    Configure options: 'MPICHLIB_CFLAGS=-g -O2'\n"), _MPICH_IMPL)
        
        test("Intel(R) MPI", _IMPI_IMPL)

        test("Unknown MPI v1.00", _UNKNOWN_IMPL)

        test("output", exit_code=1, expected=_MISSING_IMPL)

        test(None, _MISSING_IMPL)

    def test_run_controller(self):
        def test(use_gloo, use_mpi, use_js,
                 gloo_is_built, mpi_is_built,
                 lsf_exists, jsrun_installed,
                 expected, exception):
            gloo_run = MagicMock()
            mpi_run = MagicMock()
            js_run = MagicMock()

            with is_built(gloo_is_built, mpi_is_built):
                with lsf_and_jsrun(lsf_exists, jsrun_installed):
                    if exception is not None:
                        with pytest.raises(ValueError, match=exception) as e:
                            run_controller(use_gloo, gloo_run, use_mpi, mpi_run, use_js, js_run, verbosity=2)
                        return
                    run_controller(use_gloo, gloo_run, use_mpi, mpi_run, use_js, js_run, verbosity=2)

            if expected == "gloo":
                gloo_run.assert_called_once()
                mpi_run.assert_not_called()
                js_run.assert_not_called()
            elif expected == "mpi":
                gloo_run.assert_not_called()
                mpi_run.assert_called_once()
                js_run.assert_not_called()
            elif expected == "js":
                gloo_run.assert_not_called()
                mpi_run.assert_not_called()
                js_run.assert_called_once()
            else:
                raise ValueError("unsupported framework: {}".format(expected))

        bool_values = [False, True]
        bool_values_and_none = [None, False, True]

        for use_gloo, use_mpi, use_js, \
            gloo_is_built, mpi_is_built, \
            lsf_exists, jsrun_installed in \
            itertools.product(bool_values_and_none, bool_values_and_none, bool_values_and_none,
                              bool_values, bool_values,
                              bool_values, bool_values):

            expected = exception = None
            if use_gloo:
                if gloo_is_built:
                    expected = 'gloo'
                else:
                    exception = r'^Gloo support has not been built\.  If this is not expected, ensure CMake is installed ' \
                                r'and reinstall Horovod with HOROVOD_WITH_GLOO=1 to debug the build error\.$'
            elif use_mpi:
                if mpi_is_built:
                    expected = 'mpi'
                else:
                    exception = r'^MPI support has not been built\.  If this is not expected, ensure MPI is installed ' \
                                r'and reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error\.$'
            elif use_js:
                if mpi_is_built:
                    if lsf_exists:
                        expected = 'js'
                    else:
                        exception = 'Horovod did not detect an LSF job.  The jsrun launcher can only be used in that environment. ' \
                                    'Please, pick a different launcher for other environments.'
                else:
                    exception = r'^MPI support has not been built\.  If this is not expected, ensure MPI is installed ' \
                                r'and reinstall Horovod with HOROVOD_WITH_MPI=1 to debug the build error\.$'
            elif mpi_is_built:
                if lsf_exists and jsrun_installed:
                    expected = 'js'
                else:
                    expected = 'mpi'
            elif gloo_is_built:
                expected = 'gloo'
            else:
                exception = r'Neither MPI nor Gloo support has been built\. Try reinstalling Horovod ensuring that ' \
                            r'either MPI is installed \(MPI\) or CMake is installed \(Gloo\)\.'

            test(use_gloo, use_mpi, use_js,
                 gloo_is_built, mpi_is_built,
                 lsf_exists, jsrun_installed,
                 expected, exception)

    """
    Minimal mpi_run settings for tests.
    """
    minimal_settings = hvd_settings.Settings(
        verbose=0,
        num_proc=2,
        hosts='localhost:2',
        run_func_mode=True
    )

    """
    Tests mpi_run with minimal settings.
    """
    def test_mpi_run_minimal(self):
        if not mpi_available():
            self.skipTest("MPI is not available")

        cmd = ['cmd']
        settings = self.minimal_settings

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"], None

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", return_value=0) as execute:
                mpi_run(settings, None, {}, cmd)

                # call the mocked _get_mpi_implementation_flags method
                mpi_flags, binding_args, mpi = horovod.runner.mpi_run._get_mpi_implementation_flags(False)
                self.assertIsNotNone(mpi_flags)
                if _IMPI_IMPL == mpi:
                    expected_cmd = ('mpirun '
                                    '-l -n 2 -hosts localhost '
                                    '{binding_args} '
                                    '{mpi_flags}       '
                                    'cmd').format(binding_args=' '.join(binding_args), mpi_flags=' '.join(mpi_flags))
                else:
                    expected_cmd = ('mpirun '
                                    '--allow-run-as-root --tag-output '
                                    '-np 2 -H localhost:2 '
                                    '{binding_args} '
                                    '{mpi_flags}       '
                                    'cmd').format(binding_args=' '.join(binding_args), mpi_flags=' '.join(mpi_flags))

                # remove PYTHONPATH from execute's env
                # we cannot know the exact value of that env variable
                # we test right handling of PYTHONPATH in test_mpi_run_*pythonpath* below
                self.assertIn('env', execute.call_args.kwargs)
                if 'PYTHONPATH' in execute.call_args.kwargs['env']:
                    execute.call_args.kwargs['env'].pop('PYTHONPATH')

                expected_env = {'PATH': os.environ.get('PATH')}
                execute.assert_called_once_with(expected_cmd, env=expected_env, stdout=None, stderr=None)

    """
    Tests mpi_run on a large cluster.
    """
    def test_mpi_run_on_large_cluster(self):
        if not mpi_available():
            self.skipTest("MPI is not available")

        cmd = ['cmd']
        settings = copy.copy(self.minimal_settings)
        settings.hosts = ','.join(['localhost:1'] * large_cluster_threshold)

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"], None

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", return_value=0) as execute:
                mpi_run(settings, None, {}, cmd)

                # call the mocked _get_mpi_implementation_flags method
                mpi_flags, binding_args, mpi = horovod.runner.mpi_run._get_mpi_implementation_flags(False)
                self.assertIsNotNone(mpi_flags)
                if _IMPI_IMPL == mpi:
                    expected_cmd = ('mpirun '
                                    '-l -n 2 -ppn 1 -hosts {hosts} '
                                    '{binding_args} '
                                    '{mpi_flags}       '
                                    'cmd').format(hosts=settings.hosts, binding_args=' '.join(binding_args),
                                                  mpi_flags=' '.join(mpi_flags))
                else:
                    mpi_flags.append('-mca plm_rsh_no_tree_spawn true')
                    mpi_flags.append('-mca plm_rsh_num_concurrent {}'.format(large_cluster_threshold))
                    expected_cmd = ('mpirun '
                                    '--allow-run-as-root --tag-output '
                                    '-np 2 -H {hosts} '
                                    '{binding_args} '
                                    '{mpi_flags}       '
                                    'cmd').format(hosts=settings.hosts, binding_args=' '.join(binding_args),
                                                  mpi_flags=' '.join(mpi_flags))

                # remove PYTHONPATH from execute's env
                # we cannot know the exact value of that env variable
                # we test right handling of PYTHONPATH in test_mpi_run_*pythonpath* below
                self.assertIn('env', execute.call_args.kwargs)
                if 'PYTHONPATH' in execute.call_args.kwargs['env']:
                    execute.call_args.kwargs['env'].pop('PYTHONPATH')

                expected_env = {'PATH': os.environ.get('PATH')}
                execute.assert_called_once_with(expected_cmd, env=expected_env, stdout=None, stderr=None)

    """
    Tests mpi_run with full settings.
    """
    def test_mpi_run_full(self):
        if not mpi_available():
            self.skipTest("MPI is not available")

        cmd = ['cmd', 'arg1', 'arg2']
        nics = ['eth0', 'eth1']
        env = {'env1': 'val1', 'env2': 'val2'}
        stdout = '<stdout>'
        stderr = '<stderr>'
        tmout = timeout.Timeout(5, message='Timed out waiting for something.')
        settings = hvd_settings.Settings(
            verbose=0,
            ssh_port=1022,
            extra_mpi_args='>mpi-extra args go here<',
            binding_args='>binding args go here<',
            key=secret.make_secret_key(),
            start_timeout=tmout,
            num_proc=1,
            hosts='localhost:1',
            output_filename='>output filename goes here<',
            run_func_mode=True
        )

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], [], None

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags) as impl:
            with mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", return_value=0) as execute:
                mpi_run(settings, nics, env, cmd, stdout=stdout, stderr=stderr)

                # assert call on _get_mpi_implementation_flags
                impl.assert_called_once_with(None, env=env)

                # call the mocked _get_mpi_implementation_flags method ourselves
                mpi_flags, _, mpi = horovod.runner.mpi_run._get_mpi_implementation_flags(False)
                self.assertIsNotNone(mpi_flags)
                if _IMPI_IMPL == mpi:
                    expected_command = ('mpirun '
                                        '-l -n 1 -hosts {hosts} '
                                        '>binding args go here< '
                                        '{mpi_flags} '
                                        '-bootstrap-exec-args "-p 1022" '
                                        '-genv NCCL_SOCKET_IFNAME=eth0,eth1 '
                                        '-outfile-pattern >output pattern goes here< '
                                        '-genvlist env1,env2 '
                                        '>mpi-extra args go here< '
                                        'cmd arg1 arg2').format(hosts=settings.hosts,
                                                                mpi_flags=' '.join(mpi_flags))
                else:
                    expected_command = ('mpirun '
                                        '--allow-run-as-root --tag-output '
                                        '-np 1 -H {hosts} '
                                        '>binding args go here< '
                                        '{mpi_flags} '
                                        '-mca plm_rsh_args "-p 1022" '
                                        '-mca btl_tcp_if_include eth0,eth1 -x NCCL_SOCKET_IFNAME=eth0,eth1 '
                                        '--output-filename >output filename goes here< '
                                        '-x env1 -x env2 '
                                        '>mpi-extra args go here< '
                                        'cmd arg1 arg2').format(hosts=settings.hosts,
                                                                mpi_flags=' '.join(mpi_flags))

                # remove PYTHONPATH from execute's env
                # we cannot know the exact value of that env variable
                # we test right handling of PYTHONPATH in test_mpi_run_*pythonpath* below
                self.assertIn('env', execute.call_args.kwargs)
                if 'PYTHONPATH' in execute.call_args.kwargs['env']:
                    execute.call_args.kwargs['env'].pop('PYTHONPATH')

                expected_env = {'env1': 'val1', 'env2': 'val2', 'PATH': os.environ.get('PATH')}
                execute.assert_called_once_with(expected_command, env=expected_env, stdout=stdout, stderr=stderr)

    """
    Tests mpi_run without PYTHONPATH set.
    """
    def test_mpi_run_without_pythonpath(self):
        self.do_test_mpi_run_env_override({}, {}, 'PYTHONPATH', None)

    """
    Tests mpi_run with PYTHONPATH set in sys.
    """
    def test_mpi_run_with_sys_pythonpath(self):
        self.do_test_mpi_run_env_override({'PYTHONPATH': 'ppath'}, {}, 'PYTHONPATH', 'ppath')

    """
    Tests mpi_run with PYTHONPATH set in env.
    """
    def test_mpi_run_with_env_pythonpath(self):
        self.do_test_mpi_run_env_override({}, {'PYTHONPATH': 'ppath'}, 'PYTHONPATH', 'ppath')

    """
    Tests mpi_run with both PYTHONPATH set.
    """
    def test_mpi_run_with_both_pythonpaths(self):
        self.do_test_mpi_run_env_override({'PYTHONPATH': 'sys-ppath'}, {'PYTHONPATH': 'env-ppath'}, 'PYTHONPATH', 'env-ppath')

    """
    Tests mpi_run without PATH set.
    """
    def test_mpi_run_without_path(self):
        self.do_test_mpi_run_env_override({}, {}, 'PATH', None)

    """
    Tests mpi_run with PATH set in sys.
    """
    def test_mpi_run_with_sys_path(self):
        self.do_test_mpi_run_env_override({'PATH': 'ppath'}, {}, 'PATH', 'ppath')

    """
    Tests mpi_run with PATH set in env.
    """
    def test_mpi_run_with_env_path(self):
        self.do_test_mpi_run_env_override({}, {'PATH': 'ppath'}, 'PATH', 'ppath')

    """
    Tests mpi_run with both PATH set.
    """
    def test_mpi_run_with_both_paths(self):
        self.do_test_mpi_run_env_override({'PATH': 'sys-path'}, {'PATH': 'env-path'}, 'PATH', 'env-path')

    """
    Actually tests mpi_run overrides arg env with sys env.
    """
    def do_test_mpi_run_env_override(self, sysenv, argenv, env_var, expected):
        if not mpi_available():
            self.skipTest("MPI is not available")

        cmd = ['cmd']
        settings = self.minimal_settings

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"], None

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags),\
             mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", return_value=0) as execute,\
             override_env(sysenv):
            mpi_run(settings, None, argenv, cmd)

            # assert the env variable in the execute's env
            self.assertIn('env', execute.call_args.kwargs)
            self.assertEqual(execute.call_args.kwargs['env'].get(env_var), expected)

    def test_mpi_run_with_non_zero_exit(self):
        if not mpi_available():
            self.skipTest("MPI is not available")

        cmd = ['cmd']
        settings = self.minimal_settings

        def mpi_impl_flags(tcp, env=None):
            return [], [], None

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", return_value=1):
                with pytest.raises(RuntimeError, match="^mpirun failed with exit code 1$"):
                    mpi_run(settings, None, {}, cmd)

    """
    Tests mpi_run with os.environ.
    """
    def test_mpi_run_with_os_environ(self):
        if not mpi_available():
            self.skipTest("MPI is not available")

        cmd = ['cmd']
        settings = self.minimal_settings

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], ["--mock-mpi-binding-args"]

        with mock.patch("horovod.runner.mpi_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.runner.mpi_run.safe_shell_exec.execute", return_value=0):
                with pytest.raises(Exception, match="^env argument must be a dict, not <class 'os._Environ'>: "):
                    mpi_run(settings, None, os.environ, cmd)

    """
    Tests gloo_run with minimal settings.
    """
    def test_gloo_run_minimal(self):
        if not gloo_built:
            self.skipTest("Gloo is not available")

        cmd = ['whoami']
        settings = self.minimal_settings
        gloo_run(settings, ['lo'], {}, '127.0.0.1', cmd)

    """
    Tests gloo_run with os.environ.
    """
    def test_gloo_run_with_os_environ(self):
        if not gloo_built:
            self.skipTest("Gloo is not available")

        cmd = ['whoami']
        settings = self.minimal_settings
        gloo_run(settings, ['lo'], os.environ, '127.0.0.1', cmd)

    def test_horovodrun_hostfile(self):
        with temppath() as host_filename:
            with open(host_filename, 'w+') as fp:
                fp.write('172.31.32.7 slots=8\n')
                fp.write('172.31.33.9 slots=8\n')

            hostnames = hosts.parse_host_files(host_filename)
            self.assertEqual(hostnames, '172.31.32.7:8,172.31.33.9:8')

    """
    Tests js_run.
    """
    @mock.patch('horovod.runner.js_run.is_jsrun_installed', MagicMock(return_value=True))
    @mock.patch('horovod.runner.js_run.generate_jsrun_rankfile', MagicMock(return_value='/tmp/rankfile'))
    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_gpus', MagicMock(return_value=2))
    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_cores', MagicMock(return_value=2))
    def test_js_run(self):
        if _get_mpi_implementation_flags(False)[0] is None:
            self.skipTest("MPI is not available")

        cmd = ['cmd', 'arg1', 'arg2']
        env = {'env1': 'val1', 'env2': 'val2'}
        stdout = '<stdout>'
        stderr = '<stderr>'
        settings = hvd_settings.Settings(
            verbose=0,
            extra_mpi_args='>mpi-extra args go here<',
            num_proc=4,
            hosts='localhost:2,127.0.0.1:2',
            output_filename='>output filename goes here<',
            run_func_mode=True
        )

        def mpi_impl_flags(tcp, env=None):
            return ["--mock-mpi-impl-flags"], []

        with mock.patch("horovod.runner.js_run._get_mpi_implementation_flags", side_effect=mpi_impl_flags):
            with mock.patch("horovod.runner.js_run.safe_shell_exec.execute", return_value=0) as execute:
                js_run(settings, None, env, cmd, stdout=stdout, stderr=stderr)

                # call the mocked _get_mpi_implementation_flags method
                mpi_flags, _ = horovod.runner.js_run._get_mpi_implementation_flags(False)
                self.assertIsNotNone(mpi_flags)
                expected_command = ('jsrun '
                                    '--erf_input /tmp/rankfile '
                                    '--stdio_stderr >output filename goes here< '
                                    '--stdio_stdout >output filename goes here< '
                                    '--smpiargs \'{mpi_args} >mpi-extra args go here<\' '
                                    'cmd arg1 arg2').format(mpi_args=' '.join(mpi_flags))
                expected_env = {'env1': 'val1', 'env2': 'val2'}
                execute.assert_called_once_with(expected_command, env=expected_env, stdout=stdout, stderr=stderr)

    """
    Tests generate_jsrun_rankfile.
    """
    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_gpus', MagicMock(return_value=4))
    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_cores', MagicMock(return_value=4))
    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_threads', MagicMock(return_value=4))
    def test_generate_jsrun_rankfile(self):
        settings = hvd_settings.Settings(
            num_proc=5,
            hosts='host1:4,host2:4,host3:4',
        )

        with temppath() as rankfile_path:
            rankfile_path = generate_jsrun_rankfile(settings, rankfile_path)

            with open(rankfile_path, 'r') as file:
                gen_rankfile = file.read()

            expected_rankfile = (
"""overlapping_rs: allow
cpu_index_using: logical

rank: 0: { hostname: host1; cpu: {0-3} ; gpu: * ; mem: * }
rank: 1: { hostname: host1; cpu: {4-7} ; gpu: * ; mem: * }
rank: 2: { hostname: host1; cpu: {8-11} ; gpu: * ; mem: * }
rank: 3: { hostname: host1; cpu: {12-15} ; gpu: * ; mem: * }

rank: 4: { hostname: host2; cpu: {0-3} ; gpu: * ; mem: * }
""")

            self.assertMultiLineEqual(gen_rankfile, expected_rankfile)

    """
    Tests horovod.runner.launch._run with jsrun
    """
    @mock.patch('horovod.runner.util.lsf.LSFUtils.using_lsf', MagicMock(return_value=True))
    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_compute_hosts', MagicMock(return_value=['host1', 'host2']))
    @mock.patch('horovod.runner.util.lsf.LSFUtils.get_num_gpus', MagicMock(return_value=2))
    @mock.patch('horovod.runner.util.network.filter_local_addresses', MagicMock(return_value=['host1', 'host2']))
    @mock.patch('horovod.runner.launch._check_all_hosts_ssh_successful', MagicMock())
    @mock.patch('horovod.runner.launch.run_controller')
    def test_run_with_jsrun(self, mocked_run_controller):
        hargs = _HorovodArgs()
        _run(hargs)
        mocked_run_controller.assert_called_once()

    def test_get_host_assignments(self):
        hosts = parse_hosts('worker-0:2,worker-1:2')
        np = 4
        assignments = get_host_assignments(hosts, np)

        sizes = dict(size=4, local_size=2, cross_size=2)
        expected = [SlotInfo(hostname='worker-0', rank=0, local_rank=0, cross_rank=0, **sizes),
                    SlotInfo(hostname='worker-0', rank=1, local_rank=1, cross_rank=0, **sizes),
                    SlotInfo(hostname='worker-1', rank=2, local_rank=0, cross_rank=1, **sizes),
                    SlotInfo(hostname='worker-1', rank=3, local_rank=1, cross_rank=1, **sizes)]
        self.assertListEqual(assignments, expected)

    def test_get_host_assignments_elastic(self):
        hosts = parse_hosts('worker-0:2,worker-1:2')
        min_np = 1
        max_np = 2
        assignments = get_host_assignments(hosts, min_np=min_np, max_np=max_np)

        sizes = dict(size=2, local_size=2, cross_size=1)
        expected = [SlotInfo(hostname='worker-0', rank=0, local_rank=0, cross_rank=0, **sizes),
                    SlotInfo(hostname='worker-0', rank=1, local_rank=1, cross_rank=0, **sizes)]
        self.assertListEqual(assignments, expected)

    def test_get_host_assignments_heterogeneous(self):
        hosts = parse_hosts('worker-0:1,worker-1:2')
        np = 3
        assignments = get_host_assignments(hosts, np)

        expected = [SlotInfo(hostname='worker-0', rank=0, local_rank=0, cross_rank=0,
                             size=3, local_size=1, cross_size=2),
                    SlotInfo(hostname='worker-1', rank=1, local_rank=0, cross_rank=1,
                             size=3, local_size=2, cross_size=2),
                    SlotInfo(hostname='worker-1', rank=2, local_rank=1, cross_rank=0,
                             size=3, local_size=2, cross_size=1)]
        self.assertListEqual(assignments, expected)
