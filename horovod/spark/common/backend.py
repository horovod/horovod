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

import horovod.spark.common._namedtuple_fix

import os

import pyspark

import horovod.spark


def default_num_proc():
    spark_context = pyspark.SparkContext._active_spark_context
    return spark_context.defaultParallelism


class Backend(object):
    def run(self, fn, args=(), kwargs={}, env=None):
        raise NotImplementedError()

    def num_processes(self):
        raise NotImplementedError()


class SparkBackend(Backend):
    def __init__(self, num_proc=None, env=None):
        self._num_proc = num_proc or default_num_proc()
        self._env = env

    def run(self, fn, args=(), kwargs={}, env=None):
        full_env = self._env or os.environ.copy()
        if env:
            full_env.update(env)

        if 'CUDA_VISIBLE_DEVICES' in full_env:
            # In TensorFlow 2.0, we set this to prevent memory leaks from the client.
            # See https://github.com/tensorflow/tensorflow/issues/33168
            del full_env['CUDA_VISIBLE_DEVICES']

        return horovod.spark.run(fn, args=args, kwargs=kwargs,
                                 num_proc=self._num_proc, env=full_env)

    def num_processes(self):
        return self._num_proc
