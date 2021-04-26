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

import contextlib
import os
import platform
import stat
import sys

from tempfile import TemporaryDirectory

import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

from horovod.runner.common.util import secret
from horovod.spark.common.store import LocalStore
from horovod.spark.driver.driver_service import SparkDriverService, SparkDriverClient
from horovod.spark.task.task_service import SparkTaskService, SparkTaskClient

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, 'utils'))

from common import tempdir, temppath

# Spark will fail to initialize correctly locally on Mac OS without this
if platform.system() == 'Darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'


class CallbackBackend(object):
    def run(self, fn, args=(), kwargs={}, env={}):
        return [fn(*args, **kwargs)] * self.num_processes()

    def num_processes(self):
        return 1


@contextlib.contextmanager
def local_store():
    with tempdir() as tmp:
        store = LocalStore(tmp)
        yield store


@contextlib.contextmanager
def spark_session(app, cores=2, gpus=0, max_failures=1, *args):
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    with TemporaryDirectory() as tmpdir:
        metastore_path = os.path.join(tmpdir, 'metastore')

        # start a single worker with given cores when gpus are present
        # max failures are ignored when gpus in that case
        master = 'local-cluster[1,{},1024]'.format(cores) if gpus > 0 \
            else 'local[{},{}]'.format(cores, max_failures)
        conf = SparkConf().setAppName(app).setMaster(master)
        conf = conf.setAll([
            ('spark.ui.showConsoleProgress', 'false'),
            ('spark.test.home', os.environ.get('SPARK_HOME')),
            ('spark.locality.wait', '0'),
            ('spark.unsafe.exceptionOnMemoryLeak', 'true'),
            ('spark.ui.enabled', 'false'),
            ('spark.local.dir', os.path.join(tmpdir, 'tmp')),
            ('spark.sql.warehouse.dir', os.path.join(tmpdir, 'warehouse')),
            ('javax.jdo.option.ConnectionURL',
             f'jdbc:derby:;databaseName={metastore_path};create=true'),
        ])

        with temppath() as temp_filename:
            if gpus > 0:
                with open(temp_filename, 'wb') as temp_file:
                    addresses = ', '.join('\\"{}\\"'.format(i) for i in range(gpus))
                    temp_file.write(b'echo {\\"name\\": \\"gpu\\", \\"addresses\\": [' +
                                    addresses.encode('ascii') + b']}')

                os.chmod(temp_file.name, stat.S_IRWXU | stat.S_IXGRP | stat.S_IRGRP |
                         stat.S_IROTH | stat.S_IXOTH)

                # the single worker takes all gpus discovered, and a single executor will get them
                # each task on that executor will get a single gpu
                conf = conf.setAll([
                    ('spark.worker.resource.gpu.discoveryScript', temp_filename),
                    ('spark.worker.resource.gpu.amount', str(gpus)),
                    ('spark.task.resource.gpu.amount', '1'),
                    ('spark.executor.resource.gpu.amount', str(gpus)),
                ])

            session = SparkSession \
                .builder \
                .config(conf=conf) \
                .getOrCreate()

            try:
                yield session
            finally:
                session.stop()


def fn():
    return 0


@contextlib.contextmanager
def spark_driver_service(num_proc, initial_np=None, fn=fn, args=(), kwargs={},
                         key=None, nics=None, verbose=2):
    initial_np = initial_np or num_proc
    key = key or secret.make_secret_key()
    driver = SparkDriverService(initial_np, num_proc, fn, args, kwargs, key, nics)
    client = SparkDriverClient(driver.addresses(), key, verbose)

    try:
        yield driver, client, key
    finally:
        driver.shutdown()


@contextlib.contextmanager
def spark_task_service(index, key=None, nics=None, match_intf=False,
                       minimum_command_lifetime_s=0, verbose=2):
    key = key or secret.make_secret_key()
    task = SparkTaskService(index, key, nics, minimum_command_lifetime_s, verbose)
    client = SparkTaskClient(index, task.addresses(), key, verbose, match_intf)

    try:
        yield task, client, key
    finally:
        task.shutdown()


def with_features(raw_df, feature_cols):
    vector_assembler = VectorAssembler().setInputCols(feature_cols).setOutputCol('features')
    pipeline = Pipeline().setStages([vector_assembler])

    df = pipeline.fit(raw_df).transform(raw_df)
    return df


def create_xor_data(spark):
    data = [[0, 0, 0.0, 0.1], [0, 1, 1.0, 0.2], [1, 0, 1.0, 0.3], [1, 1, 0.0, 0.4]]
    schema = StructType([StructField('x1', IntegerType()),
                         StructField('x2', IntegerType()),
                         StructField('y', FloatType()),
                         StructField('weight', FloatType())])
    raw_df = create_test_data_from_schema(spark, data, schema)
    df = with_features(raw_df, ['x1', 'x2'])
    return df


def create_noisy_xor_data(spark):
    schema = StructType([StructField('x1', FloatType()),
                         StructField('x2', FloatType()),
                         StructField('y', FloatType()),
                         StructField('weight', FloatType())])
    data = [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    n = 1024
    weights = np.random.uniform(0, 1, n)

    samples = []
    noise = np.random.normal(0, 0.1, [n, 2])
    for i, eps in enumerate(noise):
        original = data[i % len(data)]
        sample = original[0:2] + eps
        samples.append(sample.tolist() + [original[2]] + [float(weights[i])])

    raw_df = create_test_data_from_schema(spark, samples, schema)
    df = with_features(raw_df, ['x1', 'x2'])
    return df


def create_mnist_data(spark):
    features = DenseVector([1.0] * 64)
    label_vec = DenseVector([0.0, 0.0, 1.0] + [0.0] * 7)
    label = 2.0
    data = [[features, label_vec, label]] * 10
    schema = StructType([StructField('features', VectorUDT()),
                         StructField('label_vec', VectorUDT()),
                         StructField('label', FloatType())])
    df = create_test_data_from_schema(spark, data, schema)
    return df


def create_test_data_from_schema(spark, data, schema):
    return spark.createDataFrame(data, schema=schema)
