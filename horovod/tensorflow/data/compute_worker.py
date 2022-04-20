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

from horovod.runner.common.service.compute_service import ComputeService
from horovod.runner.common.util import secret, env
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

    rank, size = env.get_env_rank_and_size()
    if size % parsed_args.dispatchers:
        raise ValueError(f'Number of processes ({size}) must be a multiple of number of dispatchers ({parsed_args.dispatchers}).')
    workers_per_dispatcher = size // parsed_args.dispatchers

    # start the compute service on rank 0
    compute = None
    if rank == 0:
        key = secret.make_secret_key()
        compute = ComputeService(parsed_args.dispatchers, workers_per_dispatcher, key=key)

        compute_config = TfDataServiceConfig(
            dispatchers=parsed_args.dispatchers,
            workers_per_dispatcher=workers_per_dispatcher,
            dispatcher_side=parsed_args.dispatcher_side,
            addresses=compute.addresses(),
            key=key,
        )
        compute_config.write(parsed_args.configfile)
    else:
        compute_config = TfDataServiceConfig.read(parsed_args.configfile, wait_for_file_creation=True)

    # start all compute workers
    compute_worker_fn(compute_config)

    if compute is not None:
        compute.shutdown()
