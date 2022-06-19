# Copyright 2022 G-Research. All Rights Reserved.
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

import argparse
import logging
import signal
import sys

from pyspark.sql import SparkSession

from horovod.runner.common.service.compute_service import ComputeService
from horovod.runner.common.util import secret
from horovod.spark import run
from horovod.tensorflow.data.compute_service import TfDataServiceConfig, compute_worker_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("configfile", type=str,
                        help=f"The path to store the compute service config file.")

    parser.add_argument("--dispatchers", required=False, default=1, type=int,
                        help=f"The number of dispatcher to support.",
                        dest="dispatchers")

    parser.add_argument("--dispatcher-side", required=False, default='compute', type=str,
                        help=f"Where do the dispatcher run? On 'compute' side or 'training' side.",
                        dest="dispatcher_side")

    parsed_args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()
    spark_context = spark.sparkContext
    workers = spark_context.defaultParallelism

    if workers % parsed_args.dispatchers:
        raise ValueError(f'Number of processes ({workers}) must be '
                         f'a multiple of number of dispatchers ({parsed_args.dispatchers}).')
    workers_per_dispatcher = workers // parsed_args.dispatchers

    key = secret.make_secret_key()
    compute = ComputeService(parsed_args.dispatchers, workers_per_dispatcher, key=key)

    compute_config = TfDataServiceConfig(
        dispatchers=parsed_args.dispatchers,
        workers_per_dispatcher=workers_per_dispatcher,
        dispatcher_side=parsed_args.dispatcher_side,
        addresses=compute.addresses(),
        key=key
    )
    compute_config.write(parsed_args.configfile)

    def _exit_gracefully():
        logging.info('Spark driver receiving SIGTERM. Exiting gracefully')
        spark_context.stop()

    signal.signal(signal.SIGTERM, _exit_gracefully)

    ret = run(compute_worker_fn,
              args=(compute_config,),
              stdout=sys.stdout,
              stderr=sys.stderr,
              num_proc=workers,
              verbose=2)

    compute.shutdown()
    spark.stop()

    sys.exit(ret)
