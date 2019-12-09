# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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
import pytest
import re
import subprocess
import time
import unittest
import warnings

import mock
import torch

from mock import MagicMock

from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.sql.types import ArrayType, BooleanType, DoubleType, FloatType, IntegerType, NullType, \
    StructField, StructType

import horovod.spark
import horovod.torch as hvd

from horovod.run.common.util import secret
from horovod.run.mpi_run import _get_mpi_implementation_flags
from horovod.spark.common import constants, util
from horovod.spark.task.task_service import SparkTaskService, SparkTaskClient

from spark_common import spark_session, create_test_data_from_schema, create_xor_data, local_store

from common import tempdir


@contextlib.contextmanager
def os_environ(env):
    old = os.environ
    try:
        os.environ = env
        yield
    finally:
        os.environ = old


class SparkTests(unittest.TestCase):
    """
    Tests for horovod.spark.run().
    """

    def __init__(self, *args, **kwargs):
        super(SparkTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')

    def run(self, result=None):
        if os.environ.get('OMPI_COMM_WORLD_RANK', '0') != '0':
            # Running in MPI as a rank > 0, ignore.
            return

        if 'Open MPI' not in str(subprocess.check_output('mpirun --version', shell=True)):
            return

        super(SparkTests, self).run(result)

    def test_happy_run(self):
        def fn():
            hvd.init()
            res = hvd.allgather(torch.tensor([hvd.rank()])).tolist()
            return res, hvd.rank()

        with spark_session('test_happy_run'):
            res = horovod.spark.run(fn, env={'PATH': os.environ.get('PATH')}, verbose=0)
            self.assertListEqual([([0, 1], 0), ([0, 1], 1)], res)

    def test_timeout(self):
        with spark_session('test_timeout'):
            with pytest.raises(Exception, match='^Timed out waiting for Spark tasks to start.'):
                horovod.spark.run(None, num_proc=4, start_timeout=5,
                                  env={'PATH': os.environ.get('PATH')},
                                  verbose=0)

    def test_mpirun_not_found(self):
        start = time.time()
        with spark_session('test_mpirun_not_found'):
            with pytest.raises(Exception, match='^mpirun failed with exit code 127$'):
                horovod.spark.run(None, env={'PATH': '/nonexistent'}, verbose=0)
        self.assertLessEqual(time.time() - start, 10, 'Failure propagation took too long')

    """
    Test that horovod.spark.run invokes mpi_run properly.
    """
    def test_spark_run_func(self):
        env = {'env1': 'val1', 'env2': 'val2'}
        expected_env = '-x env1 -x env2'
        extra_mpi_args = '<extra args go here>'
        self.do_test_spark_run_func(num_proc=2, extra_mpi_args=extra_mpi_args,
                                    env=env, stdout='<stdout>', stderr='<stderr>',
                                    cores=4, expected_np=2, expected_env=expected_env)

    """
    Test that horovod.spark.run defaults num_proc to spark parallelism.
    """
    def test_spark_run_func_defaults_num_proc_to_spark_cores(self):
        self.do_test_spark_run_func(num_proc=None, cores=2, expected_np=2)

    """
    Test that horovod.spark.run defaults env to the full system env.
    """
    def test_spark_run_func_defaults_env_to_os_env(self):
        env = {'env1': 'val1', 'env2': 'val2'}
        expected_env = '-x env1 -x env2'

        with os_environ(env):
            self.do_test_spark_run_func(env=None, expected_env=expected_env)

    """
    Test that horovod.spark.run raises and exception on non-zero exit code of mpi_run.
    """
    def test_spark_run_func_with_non_zero_exit(self):
        run_func = MagicMock(return_value=1)

        def fn():
            return 1

        with spark_session('test_spark_run_func', cores=4):
            with pytest.raises(Exception, match='^mpirun failed with exit code 1$') as e:
                horovod.spark.run(fn, verbose=0, run_func=run_func)

    """
    Performs the actual horovod.spark.run test.
    """
    def do_test_spark_run_func(self, args=(), kwargs={}, num_proc=1, extra_mpi_args=None, env={},
                               stdout=None, stderr=None, verbose=0,
                               cores=2, expected_np=1, expected_env=''):
        def fn():
            return 1

        run_func = MagicMock(return_value=0)

        with spark_session('test_spark_run_func', cores=cores):
            with pytest.raises(Exception) as e:
                # we need to timeout horovod because our mocked run_func will block spark otherwise
                # this raises above exception, but allows us to catch run_func arguments
                horovod.spark.run(fn, args=args, kwargs=kwargs,
                                  num_proc=num_proc, start_timeout=1,
                                  extra_mpi_args=extra_mpi_args, env=env,
                                  stdout=stdout, stderr=stderr, verbose=verbose,
                                  run_func=run_func)

        self.assertFalse(str(e.value).startswith('Timed out waiting for Spark tasks to start.'),
                         'Spark timed out before mpi_run was called, test setup is broken.')
        self.assertEqual(str(e.value), 'Spark job has failed, see the error above.')

        mpi_flags = _get_mpi_implementation_flags()
        self.assertIsNotNone(mpi_flags)
        expected_command = ('mpirun '
                            '--allow-run-as-root --tag-output '
                            '-np {expected_np} -H [^ ]+ '
                            '-bind-to none -map-by slot '
                            '{mpi_flags}  '
                            '-mca btl_tcp_if_include [^ ]+ -x NCCL_SOCKET_IFNAME=[^ ]+  '
                            '-x _HOROVOD_SECRET_KEY {expected_env}'
                            '{extra_mpi_args} '
                            '-x NCCL_DEBUG=INFO '
                            r'-mca plm_rsh_agent "[^"]+python[\d]* -m horovod.spark.driver.mpirun_rsh [^ ]+ [^ ]+" '
                            r'[^"]+python[\d]* -m horovod.spark.task.mpirun_exec_fn [^ ]+ [^ ]+'.format(
                                expected_np=expected_np,
                                expected_env=expected_env + ' ' if expected_env else '',
                                mpi_flags=' '.join(mpi_flags),
                                extra_mpi_args=extra_mpi_args if extra_mpi_args else ''))

        run_func.assert_called_once()
        run_func_args, run_func_kwargs = run_func.call_args
        actual_command = run_func_kwargs.get('command')
        actual_env = run_func_kwargs.get('env')
        actual_stdout = run_func_kwargs.get('stdout')
        actual_stderr = run_func_kwargs.get('stderr')
        actual_secret = actual_env.pop('_HOROVOD_SECRET_KEY', None)

        # for better comparison replace sections in actual_command that change across runs / hosts
        for replacement in ('-H [^ ]+', '-mca btl_tcp_if_include [^ ]+', '-x NCCL_SOCKET_IFNAME=[^ ]+',
                            r'"[^"]+python[\d]*', r' [^"]+python[\d]*',
                            '-m horovod.spark.driver.mpirun_rsh [^ ]+ [^ ]+"',
                            '-m horovod.spark.task.mpirun_exec_fn [^ ]+ [^ ]+'):
            actual_command = re.sub(replacement, replacement, actual_command, 1)

        self.assertEqual(run_func_args, ())
        self.assertEqual(actual_command, expected_command)
        if env:
            self.assertEqual(actual_env, env)
        else:
            self.assertIsNotNone(actual_env)
        self.assertIsNotNone(actual_secret)
        self.assertTrue(len(actual_secret) > 0)
        self.assertEqual(actual_stdout, stdout)
        self.assertEqual(actual_stderr, stderr)

    def test_df_cache(self):
        # Clean the cache before starting the test
        util.clear_training_cache()
        util._training_cache.get = mock.Mock(side_effect=util._training_cache.get)

        with spark_session('test_df_cache') as spark:
            with local_store() as store:
                df = create_xor_data(spark)

                key = util._training_cache.create_key(df, store, None)
                assert not util._training_cache.is_cached(key, store)

                train_rows, val_rows, metadata, avg_row_size = \
                    util.prepare_data(num_processes=2,
                                      store=store,
                                      df=df,
                                      feature_columns=['features'],
                                      label_columns=['y'])

                util._training_cache.get.assert_not_called()
                assert len(util._training_cache._entries) == 1
                assert util._training_cache.is_cached(key, store)

                train_rows_cached, val_rows_cached, metadata_cached, avg_row_size_cached = \
                    util.prepare_data(num_processes=2,
                                      store=store,
                                      df=df,
                                      feature_columns=['features'],
                                      label_columns=['y'])

                util._training_cache.get.assert_called()
                assert train_rows == train_rows_cached
                assert val_rows == val_rows_cached
                assert metadata == metadata_cached
                assert avg_row_size == avg_row_size_cached

                bad_key = util._training_cache.create_key(df, store, 0.1)
                assert not util._training_cache.is_cached(bad_key, store)

    def test_get_col_info(self):
        with spark_session('test_get_col_info') as spark:
            data = [[
                0,
                0.0,
                None,
                [1, 1],
                DenseVector([1.0, 1.0]),
                SparseVector(2, {1: 1.0}),
                DenseVector([1.0, 1.0])
            ], [
                1,
                None,
                None,
                [1, 1],
                DenseVector([1.0, 1.0]),
                SparseVector(2, {1: 1.0}),
                SparseVector(2, {1: 1.0})
            ]]

            schema = StructType([
                StructField('int', IntegerType()),
                StructField('float', FloatType()),
                StructField('null', NullType()),
                StructField('array', ArrayType(IntegerType())),
                StructField('dense', VectorUDT()),
                StructField('sparse', VectorUDT()),
                StructField('mixed', VectorUDT())
            ])

            df = create_test_data_from_schema(spark, data, schema)
            all_col_types, col_shapes, col_max_sizes = util._get_col_info(df)

            expected = [
                ('int', {int}, 1, 1),
                ('float', {float, NullType}, 1, 1),
                ('null', {NullType}, 1, 1),
                ('array', {list}, 2, 2),
                ('dense', {DenseVector}, 2, 2),
                ('sparse', {SparseVector}, 2, 1),
                ('mixed', {DenseVector, SparseVector}, 2, 2)
            ]

            for expected_col_info in expected:
                col_name, col_types, col_shape, col_size = expected_col_info
                assert all_col_types[col_name] == col_types, col_name
                assert col_shapes[col_name] == col_shape, col_name
                assert col_max_sizes[col_name] == col_size, col_name

    def test_get_col_info_error_bad_shape(self):
        with spark_session('test_get_col_info_error_bad_shape') as spark:
            data_bad_shape = [
                [SparseVector(2, {0: 1.0})],
                [SparseVector(1, {0: 1.0})]
            ]
            schema = StructType([StructField('data', VectorUDT())])
            df = create_test_data_from_schema(spark, data_bad_shape, schema)

            with pytest.raises(ValueError):
                util._get_col_info(df)

    def test_get_col_info_error_bad_size(self):
        with spark_session('test_get_col_info_error_bad_size') as spark:
            data_bad_size = [
                [DenseVector([1.0, 1.0])],
                [DenseVector([1.0])]
            ]
            schema = StructType([StructField('data', VectorUDT())])
            df = create_test_data_from_schema(spark, data_bad_size, schema)

            with pytest.raises(ValueError):
                util._get_col_info(df)

    def test_train_val_split_ratio(self):
        with spark_session('test_train_val_split_ratio') as spark:
            data = [
                [1.0], [1.0], [1.0], [1.0], [1.0]
            ]
            schema = StructType([StructField('data', FloatType())])
            df = create_test_data_from_schema(spark, data, schema)
            metadata = util._get_metadata(df)

            validation = 0.2
            train_df, val_df, validation_ratio = util._train_val_split(df, metadata, validation)

            # Only check validation ratio, as we can't rely on random splitting to produce an exact
            # result of 4 training and 1 validation samples.
            assert validation_ratio == validation

    def test_train_val_split_col_integer(self):
        with spark_session('test_train_val_split_col_integer') as spark:
            data = [
                [1.0, 0], [1.0, 0], [1.0, 0], [1.0, 0], [1.0, 1]
            ]
            schema = StructType([StructField('data', FloatType()), StructField('val', IntegerType())])
            df = create_test_data_from_schema(spark, data, schema)
            metadata = util._get_metadata(df)

            validation = 'val'
            train_df, val_df, validation_ratio = util._train_val_split(df, metadata, validation)

            # Only check counts as validation ratio cannot be guaranteed due to approx calculation
            assert train_df.count() == 4
            assert val_df.count() == 1

    def test_train_val_split_col_boolean(self):
        with spark_session('test_train_val_split_col_boolean') as spark:
            data = [
                [1.0, False], [1.0, False], [1.0, False], [1.0, False], [1.0, True]
            ]
            schema = StructType([StructField('data', FloatType()), StructField('val', BooleanType())])
            df = create_test_data_from_schema(spark, data, schema)
            metadata = util._get_metadata(df)

            validation = 'val'
            train_df, val_df, validation_ratio = util._train_val_split(df, metadata, validation)

            # Only check counts as validation ratio cannot be guaranteed due to approx calculation
            assert train_df.count() == 4
            assert val_df.count() == 1

    def test_get_metadata(self):
        expected_metadata = \
            {
                'float': {
                    'spark_data_type': FloatType,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.NOCHANGE,
                    'max_size': 1,
                    'shape': 1
                },
                'dense': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.ARRAY,
                    'max_size': 2,
                    'shape': 2
                },
                'sparse': {
                    'spark_data_type': SparseVector,
                    'is_sparse_vector_only': True,
                    'intermediate_format': constants.CUSTOM_SPARSE,
                    'max_size': 1,
                    'shape': 2
                },
                'mixed': {
                    'spark_data_type': DenseVector,
                    'is_sparse_vector_only': False,
                    'intermediate_format': constants.ARRAY,
                    'max_size': 2,
                    'shape': 2
                },
            }

        with spark_session('test_get_metadata') as spark:
            data = [
                [1.0, DenseVector([1.0, 1.0]), SparseVector(2, {0: 1.0}), DenseVector([1.0, 1.0])],
                [1.0, DenseVector([1.0, 1.0]), SparseVector(2, {1: 1.0}), SparseVector(2, {1: 1.0})]
            ]
            schema = StructType([
                StructField('float', FloatType()),
                StructField('dense', VectorUDT()),
                StructField('sparse', VectorUDT()),
                StructField('mixed', VectorUDT())
            ])
            df = create_test_data_from_schema(spark, data, schema)

            metadata = util._get_metadata(df)
            self.assertDictEqual(metadata, expected_metadata)

    def test_check_shape_compatibility(self):
        feature_columns = ['x1', 'x2', 'features']
        label_columns = ['y1', 'y_embedding']

        schema = StructType([StructField('x1', DoubleType()),
                             StructField('x2', IntegerType()),
                             StructField('features', VectorUDT()),
                             StructField('y1', FloatType()),
                             StructField('y_embedding', VectorUDT())])
        data = [[1.0, 1, DenseVector([1.0] * 12), 1.0, DenseVector([1.0] * 12)]] * 10

        with spark_session('test_df_cache') as spark:
                df = create_test_data_from_schema(spark, data, schema)
                metadata = util._get_metadata(df)

                input_shapes = [[1], [1], [-1, 3, 4]]
                output_shapes = [[1], [-1, 3, 4]]
                util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                               input_shapes, output_shapes)

                input_shapes = [[1], [1], [3, 2, 2]]
                output_shapes = [[1, 1], [-1, 2, 3, 2]]
                util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                               input_shapes, output_shapes)

                bad_input_shapes = [[1], [1], [-1, 3, 5]]
                with pytest.raises(ValueError):
                    util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                                   bad_input_shapes, output_shapes)

                bad_input_shapes = [[2], [1], [-1, 3, 4]]
                with pytest.raises(ValueError):
                    util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                                   bad_input_shapes, output_shapes)

                bad_output_shapes = [[7], [-1, 3, 4]]
                with pytest.raises(ValueError):
                    util.check_shape_compatibility(metadata, feature_columns, label_columns,
                                                   input_shapes, bad_output_shapes)

    def test_spark_task_service_env(self):
        key = secret.make_secret_key()
        service_env = dict([(key, '{} value'.format(key))
                            for key in SparkTaskService.SERVICE_ENV_KEYS])
        service_env.update({"other": "value"})
        with os_environ(service_env):
            service = SparkTaskService(1, key, None)
            client = SparkTaskClient(1, service.addresses(), key, 3)

            with tempdir() as d:
                file = '{}/env'.format(d)
                command = "env | grep -v '^PWD='> {}".format(file)
                command_env = {"test": "value"}

                try:
                    client.run_command(command, command_env)
                    client.wait_for_command_termination()
                finally:
                    service.shutdown()

                with open(file) as f:
                    env = sorted([line.strip() for line in f.readlines()])
                    expected = ['HADOOP_TOKEN_FILE_LOCATION=HADOOP_TOKEN_FILE_LOCATION value', 'test=value']
                    self.assertEqual(env, expected)
