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

import contextlib
import os
import platform
import stat

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

from horovod.spark.common.store import LocalStore

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
def spark_session(app, cores=2, gpus=0, *args):
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    master = 'local-cluster[{},1,1024]'.format(cores) if gpus > 0 else 'local[{}]'.format(cores)
    conf = SparkConf().setAppName(app).setMaster(master)

    with temppath() as temp_filename:
        if gpus > 0:
            with open(temp_filename, 'wb') as temp_file:
                addresses = ', '.join('\\"{}\\"'.format(i) for i in range(gpus))
                temp_file.write(b'echo {\\"name\\": \\"gpu\\", \\"addresses\\": [' +
                                addresses.encode('ascii') + b']}')

            os.chmod(temp_file.name, stat.S_IRWXU | stat.S_IXGRP | stat.S_IRGRP |
                     stat.S_IROTH | stat.S_IXOTH)

            conf = conf.set("spark.test.home", os.environ.get('SPARK_HOME'))
            conf = conf.set("spark.worker.resource.gpu.discoveryScript", temp_filename)
            conf = conf.set("spark.worker.resource.gpu.amount", 1)
            conf = conf.set("spark.task.resource.gpu.amount", "1")
            conf = conf.set("spark.executor.resource.gpu.amount", "1")

        session = SparkSession \
            .builder \
            .config(conf=conf) \
            .getOrCreate()

        try:
            yield session
        finally:
            session.stop()


def create_xor_data(spark):
    data = [[0, 0, 0.0, 0.1], [0, 1, 1.0, 0.2], [1, 0, 1.0, 0.3], [1, 1, 0.0, 0.4]]
    schema = StructType([StructField('x1', IntegerType()),
                         StructField('x2', IntegerType()),
                         StructField('y', FloatType()),
                         StructField('weight', FloatType())])
    raw_df = create_test_data_from_schema(spark, data, schema)

    vector_assembler = VectorAssembler().setInputCols(['x1', 'x2']).setOutputCol('features')
    pipeline = Pipeline().setStages([vector_assembler])

    df = pipeline.fit(raw_df).transform(raw_df)
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
