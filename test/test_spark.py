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

import os
import subprocess
import time
import torch
import traceback
import unittest
import warnings

import horovod.spark
import horovod.torch as hvd


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
        from pyspark import SparkConf
        from pyspark.sql import SparkSession
        conf = SparkConf().setAppName("test_happy_run").setMaster("local[2]")
        spark = SparkSession \
            .builder \
            .config(conf=conf) \
            .getOrCreate()

        def fn():
            hvd.init()
            res = hvd.allgather(torch.tensor([hvd.rank()])).tolist()
            return res, hvd.rank()

        try:
            res = horovod.spark.run(fn, env={'PATH': os.environ.get('PATH')})
            self.assertListEqual([([0, 1], 0), ([0, 1], 1)], res)
        finally:
            spark.stop()

    def test_timeout(self):
        from pyspark import SparkConf
        from pyspark.sql import SparkSession
        conf = SparkConf().setAppName("test_happy_run").setMaster("local[2]")
        spark = SparkSession \
            .builder \
            .config(conf=conf) \
            .getOrCreate()

        try:
            horovod.spark.run(None, num_proc=4, start_timeout=5,
                              env={'PATH': os.environ.get('PATH')})
            self.fail("Timeout expected")
        except Exception as e:
            print('Caught exception:')
            traceback.print_exc()
            self.assertIn("Timed out waiting for Spark tasks to start", str(e))
        finally:
            spark.stop()

    def test_mpirun_not_found(self):
        from pyspark import SparkConf
        from pyspark.sql import SparkSession
        conf = SparkConf().setAppName("test_happy_run").setMaster("local[2]")
        spark = SparkSession \
            .builder \
            .config(conf=conf) \
            .getOrCreate()

        start = time.time()
        try:
            horovod.spark.run(None, env={'PATH': '/nonexistent'})
            self.fail("Failure expected")
        except Exception as e:
            print('Caught exception:')
            traceback.print_exc()
            self.assertIn("mpirun exited with code", str(e))
            self.assertLessEqual(time.time() - start, 10, "Failure propagation took too long")
        finally:
            spark.stop()
