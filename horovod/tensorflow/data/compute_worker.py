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

import tensorflow as tf

import horovod.tensorflow as hvd
from horovod.runner.common.service.compute_service import ComputeService
from horovod.runner.common.util import secret
from horovod.tensorflow.data.compute_service import TfDataServiceConfig, compute_worker_fn


def main(dispatchers: int, dispatcher_side: str, configfile: str, timeout: int):
    hvd.init()
    rank, size = hvd.rank(), hvd.size()

    if size % dispatchers:
        raise ValueError(f'Number of processes ({size}) must be a multiple of number of dispatchers ({dispatchers}).')
    workers_per_dispatcher = size // dispatchers

    # start the compute service on rank 0
    compute = None
    try:
        compute_config = None

        if rank == 0:
            key = secret.make_secret_key()
            compute = ComputeService(dispatchers, workers_per_dispatcher, key=key)

            compute_config = TfDataServiceConfig(
                dispatchers=dispatchers,
                workers_per_dispatcher=workers_per_dispatcher,
                dispatcher_side=dispatcher_side,
                addresses=compute.addresses(),
                key=key,
                timeout=timeout
            )
            compute_config.write(configfile)

        # broadcast this config to all ranks via CPU ops
        with tf.device(f'/cpu:0'):
            compute_config = hvd.broadcast_object(compute_config, name='TfDataServiceConfig')

        # start all compute workers
        compute_worker_fn(compute_config)
    finally:
        if compute is not None:
            compute.shutdown()


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

    parser.add_argument("--timeout", required=False, default=60, type=int,
                        help=f"Timeout to setup worker and connect everything.",
                        dest="timeout")

    parsed_args = parser.parse_args()
    main(parsed_args.dispatchers, parsed_args.dispatcher_side, parsed_args.configfile, parsed_args.timeout)
