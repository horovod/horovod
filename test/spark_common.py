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

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

from horovod.spark.common.store import LocalStore

from common import tempdir

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
def spark_session(app, cores=2, *args):
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    conf = SparkConf().setAppName(app).setMaster('local[{}]'.format(cores))
    session = SparkSession \
        .builder \
        .config(conf=conf) \
        .getOrCreate()

    try:
        yield session
    finally:
        session.stop()


def create_xor_data(spark):
    data = [[0, 0, 0.0], [0, 1, 1.0], [1, 0, 1.0], [1, 1, 0.0]]
    schema = StructType([StructField('x1', IntegerType()),
                             StructField('x2', IntegerType()),
                             StructField('y', FloatType())])
    raw_df = create_test_data_from_schema(spark, data, schema)

    vector_assembler = VectorAssembler().setInputCols(['x1', 'x2']).setOutputCol('features')
    pipeline = Pipeline().setStages([vector_assembler])

    df = pipeline.fit(raw_df).transform(raw_df)
    return df


def create_test_data_from_schema(spark, data, schema):
    return spark.createDataFrame(data, schema=schema)
