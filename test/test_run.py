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

import contextlib
import os
import sys
import unittest
import warnings

import pytest

from horovod.run import run
from horovod.run.common.util import config_parser


@contextlib.contextmanager
def override_args(tool=None, *args):
    old = sys.argv[:]
    try:
        if tool:
            sys.argv[0] = tool
        sys.argv[1:] = args
        yield
    finally:
        sys.argv = old


class RunTests(unittest.TestCase):
    """
    Tests for horovod.run.
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
            args = run.parse_args()
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
            args = run.parse_args()
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
            args = run.parse_args()
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
            args = run.parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_TIMELINE), '/tmp/timeline.json')
            self.assertEqual(env.get(config_parser.HOROVOD_TIMELINE_MARK_CYCLES), '1')

    def test_stall_check_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--no-stall-check'):
            args = run.parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_STALL_CHECK_DISABLE), '1')

        with override_args('horovodrun', '-np', '2',
                           '--stall-check-warning-time-seconds', '10',
                           '--stall-check-shutdown-time-seconds', '20'):
            args = run.parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertNotIn(config_parser.HOROVOD_STALL_CHECK_DISABLE, env)
            self.assertEqual(env.get(config_parser.HOROVOD_STALL_CHECK_TIME_SECONDS), '10')
            self.assertEqual(env.get(config_parser.HOROVOD_STALL_SHUTDOWN_TIME_SECONDS), '20')

    def test_library_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--mpi-threads-disable',
                           '--num-nccl-streams', '2',
                           '--mlsl-bgt-affinity', '1'):
            args = run.parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_MPI_THREADS_DISABLE), '1')
            self.assertEqual(env.get(config_parser.HOROVOD_NUM_NCCL_STREAMS), '2')
            self.assertEqual(env.get(config_parser.HOROVOD_MLSL_BGT_AFFINITY), '1')

    def test_logging_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--log-level', 'INFO',
                           '--log-hide-timestamp'):
            args = run.parse_args()
            env = {}
            config_parser.set_env_from_args(env, args)

            self.assertEqual(env.get(config_parser.HOROVOD_LOG_LEVEL), 'INFO')
            self.assertEqual(env.get(config_parser.HOROVOD_LOG_HIDE_TIME), '1')

    def test_config_file(self):
        config_filename = os.path.join(os.path.dirname(__file__), 'data/config.test.yaml')
        with override_args('horovodrun', '-np', '2',
                           '--config-file', config_filename):
            args = run.parse_args()

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
            self.assertEqual(args.mlsl_bgt_affinity, 1)

            # Logging
            self.assertEqual(args.log_level, 'INFO')
            self.assertTrue(args.log_hide_timestamp)

    def test_config_file_override_args(self):
        config_filename = os.path.join(os.path.dirname(__file__), 'data/config.test.yaml')
        with override_args('horovodrun', '-np', '2',
                           '--fusion-threshold-mb', '128',
                           '--config-file', config_filename,
                           '--cycle-time-ms', '20',):
            args = run.parse_args()
            self.assertEqual(args.fusion_threshold_mb, 128)
            self.assertEqual(args.cycle_time_ms, 20)

    def test_validate_config_args(self):
        with override_args('horovodrun', '-np', '2',
                           '--fusion-threshold-mb', '-1'):
            with pytest.raises(ValueError):
                run.parse_args()
