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
from typing import Optional

import horovod.tensorflow as hvd
from horovod.runner.common.service.compute_service import ComputeService
from horovod.runner.common.util import secret
from horovod.tensorflow.data.compute_service import TfDataServiceConfig, compute_worker_fn


def main(dispatchers: int,
         dispatchers_work_dir: Optional[str],
         dispatchers_nic: str,
         dispatcher_side: str,
         configfile: str,
         timeout: int):
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
            compute = ComputeService(dispatchers,
                                     workers_per_dispatcher,
                                     fault_tolerant=dispatchers_work_dir is not None,
                                     key=key)

            compute_config = TfDataServiceConfig(
                dispatchers=dispatchers,
                dispatchers_work_dir=dispatchers_work_dir,
                dispatchers_nic=dispatchers_nic,
                workers_per_dispatcher=workers_per_dispatcher,
                dispatcher_side=dispatcher_side,
                addresses=compute.addresses(),
                key=key,
                timeout=timeout
            )
            compute_config.write(configfile)
        else:
            compute_config = TfDataServiceConfig.read(configfile, wait_for_file_creation=True)

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

    parser.add_argument("--dispatchers-work-dir", required=False, default='None', type=str,
                        help=f"The path to dispatchers working directories. Setting this enables fault tolerance mode.",
                        dest="dispatchers_work_dir")

    parser.add_argument("--dispatchers-nic", required=True, type=str,
                        help=f"The network interface (NIC) to reach the dispatchers.",
                        dest="dispatchers_nic")

    parser.add_argument("--dispatcher-side", required=False, default='compute', type=str,
                        help=f"Where do the dispatcher run? On 'compute' side or 'training' side.",
                        dest="dispatcher_side")

    parser.add_argument("--timeout", required=False, default=60, type=int,
                        help=f"Timeout to setup worker and connect everything.",
                        dest="timeout")

    parsed_args = parser.parse_args()
    main(parsed_args.dispatchers,
         parsed_args.dispatchers_work_dir,
         parsed_args.dispatchers_nic,
         parsed_args.dispatcher_side,
         parsed_args.configfile,
         parsed_args.timeout)
