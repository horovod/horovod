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
import os
import sys

from pyspark import SparkConf
from pyspark.sql import SparkSession

# we want to reuse train_fn from tensorflow2_mnist_data_service_train_fn_*_side_dispatcher
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'tensorflow2'))

from tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher import train_fn as train_fn_compute_side
from tensorflow2_mnist_data_service_train_fn_training_side_dispatcher import train_fn as train_fn_training_side

from horovod.spark import run
from horovod.tensorflow.data.compute_service import TfDataServiceConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# This exemplifies how to use the Tensorflow Compute Service with Horovod.
# The Tensorflow Dispatcher can reside with the training script, or the compute service.
# If you use only one of these options, you can ignore the respective code of the other option in this example.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--compute-service-config-file", required=True, type=str,
                        help=f"The path to the compute service config file.",
                        dest="compute_service_config_file")

    parser.add_argument("--training-tasks", required=False, type=int,
                        help=f"The number of training tasks when there is only one dispatcher. "
                             f"Otherwise there are as many training tasks as there are dispatchers.",
                        dest="training_tasks")

    parser.add_argument("--reuse-dataset", required=False, action="store_true", default=False,
                        help=f"Reusing the dataset allows the training tasks to reads from a single dataset "
                             f"in a first-come-first-serve manner.",
                        dest="reuse_dataset")

    parser.add_argument("--round-robin", required=False, action="store_true", default=False,
                        help=f"Reusing the dataset can be done round-robin instead first-come-first-serve.",
                        dest="round_robin")

    parsed_args = parser.parse_args()

    compute_config = TfDataServiceConfig.read(parsed_args.compute_service_config_file)

    if compute_config.dispatchers == 1:
        if parsed_args.training_tasks is None:
            print('Please provide the number of training tasks via --training-tasks since the data service '
                  'is configured to have only one dispatcher for all tasks.\n'
                  'Thus we cannot determine the number of training tasks from the dispatchers.', file=sys.stderr)
            sys.exit(1)
    else:
        if parsed_args.training_tasks is not None and parsed_args.training_tasks != compute_config.dispatchers:
            print(f'The provide number of training tasks ({parsed_args.training_tasks}) must match '
                  f'the number of dispatchers ({compute_config.dispatchers}) configured in the '
                  f'data service config file ({parsed_args.compute_service_config_file}).', file=sys.stderr)
            sys.exit(1)

    training_tasks = parsed_args.training_tasks or compute_config.dispatchers

    conf = SparkConf()
    conf.setMaster(f'local[{training_tasks}]').setAppName('train')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # pick the right train_fn depending on the dispatcher side
    if compute_config.dispatcher_side == 'training':
        train_fn = train_fn_training_side
    elif compute_config.dispatcher_side == 'compute':
        train_fn = train_fn_compute_side
    else:
        raise ValueError(f'Unsupported dispatcher side: {compute_config.dispatcher_side}')

    # run the distributed training
    run(train_fn,
        args=(compute_config,),
        kwargs={
            'reuse_dataset': parsed_args.reuse_dataset,
            'round_robin': parsed_args.round_robin
        },
        num_proc=training_tasks,
        stdout=sys.stdout,
        stderr=sys.stderr)

    compute = compute_config.compute_client(verbose=2)
    compute.shutdown()