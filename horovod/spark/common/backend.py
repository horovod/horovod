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

import horovod.spark.common._namedtuple_fix

import os

import pyspark

import horovod.spark


def default_num_proc():
    spark_context = pyspark.SparkContext._active_spark_context
    return spark_context.defaultParallelism


class Backend(object):
    """Interface for remote execution of the distributed training function.

    A custom backend can be used in cases where the training environment running Horovod is different
    from the Spark application running the HorovodEstimator.
    """

    def run(self, fn, args=(), kwargs={}, env=None):
        """Executes the training `fn` and returns results from each worker in a list (ordered by ascending rank).

        Args:
            fn: Function to run.
            args: Arguments to pass to `fn`.
            kwargs: Keyword arguments to pass to `fn`.
            env: Environment dictionary to use in Horovod run.  Defaults to `os.environ`.

        Returns:
            List of results returned by running `fn` on each rank.
        """
        raise NotImplementedError()

    def num_processes(self):
        """Returns the number of processes to use for training."""
        raise NotImplementedError()


class SparkBackend(Backend):
    """Uses `horovod.spark.run` to execute the distributed training `fn`."""

    def __init__(self, num_proc=None, env=None, **kwargs):
        """
        Args:
            num_proc: Number of Horovod processes.  Defaults to `spark.default.parallelism`.
            env: Environment dictionary to use in Horovod run.  Defaults to `os.environ`.
            **kwargs: Additional arguments passed to `horovod.spark.run` at training time.
        """
        self._num_proc = num_proc or default_num_proc()
        self._env = env
        self._kwargs = kwargs

    def run(self, fn, args=(), kwargs={}, env=None):
        full_env = self._env or os.environ.copy()
        if env:
            full_env.update(env)

        if 'CUDA_VISIBLE_DEVICES' in full_env:
            # In TensorFlow 2.0, we set this before calling `run` in order to prevent TensorFlow
            # from allocating memory on the GPU outside the training process.  Once we submit the
            # function for execution, we want to ensure that TensorFLow has visibility into GPUs on
            # the device so we can use them for training, which is why we need to unset this.
            # See https://github.com/tensorflow/tensorflow/issues/33168
            del full_env['CUDA_VISIBLE_DEVICES']

        return horovod.spark.run(fn, args=args, kwargs=kwargs,
                                 num_proc=self._num_proc, env=full_env,
                                 **self._kwargs)

    def num_processes(self):
        return self._num_proc
