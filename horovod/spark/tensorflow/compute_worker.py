import argparse
import logging
import os
import signal
import sys
from datetime import datetime

from horovod.runner.common.util import secret
from horovod.spark import run
from pyspark import SparkConf
from pyspark.sql import SparkSession

from nereus.tf_data_service import TfDataServiceConfig, compute_worker_fn
from nereus.tf_data_service.compute_service import ComputeService
from nereus.utils.logging_utils import initialise_default_logging, Tee

initialise_default_logging(os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--compute-service-config-file", required=True, type=str,
                        help=f"The path to store the compute service config file.",
                        dest="compute_service_config_file")

    parser.add_argument("--dispatchers", required=True, type=int,
                        help=f"The number of dispatcher to support.",
                        dest="dispatchers")

    parser.add_argument("--workers-per-dispatcher", required=True, type=int,
                        help=f"The number of workers per dispatcher.",
                        dest="workers_per_dispatcher")

    parser.add_argument("--dispatcher-side", required=True, type=str,
                        help=f"Where do the dispatcher run? On 'compute' side or 'training' side.",
                        dest="dispatcher_side")

    parsed_args = parser.parse_args()
    workers = parsed_args.dispatchers * parsed_args.workers_per_dispatcher

    key = secret.make_secret_key()
    compute = ComputeService(parsed_args.dispatchers, parsed_args.workers_per_dispatcher, key=key)

    compute_config = TfDataServiceConfig(
        dispatchers=parsed_args.dispatchers,
        workers_per_dispatcher=parsed_args.workers_per_dispatcher,
        dispatcher_side=parsed_args.dispatcher_side,
        addresses=compute.addresses(),
        key=key
    )
    compute_config.write(parsed_args.compute_service_config_file)

    conf = SparkConf()
    conf.setMaster(f'local[{workers}]').setAppName('compute')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark_context = spark.sparkContext

    def _exit_gracefully():
        logging.info('Spark driver receiving SIGTERM. Exiting gracefully')
        spark_context.stop()

    signal.signal(signal.SIGTERM, _exit_gracefully)

    ret = run(compute_worker_fn,
              args=(compute_config,),
              stdout=Tee(f'nereus-compute-logs/stdout/{datetime.now().strftime("%Y%m%d-%H%M%S")}.log', mode='w'),
              stderr=Tee(f'nereus-compute-logs/stderr/{datetime.now().strftime("%Y%m%d-%H%M%S")}.log', mode='w'),
              num_proc=workers,
              verbose=2)

    compute.shutdown()
    spark.stop()

    sys.exit(ret)
