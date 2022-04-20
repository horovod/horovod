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
import sys

from horovod import run
from horovod.runner.common.service.compute_service import ComputeService
from horovod.runner.common.util import secret
from horovod.tensorflow.data.compute_service import TfDataServiceConfig, compute_worker_fn

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

    parser.add_argument("--output-filename", required=False, default="compute-worker-log", type=str,
                        help=f"For Gloo, writes stdout / stderr of all workers to a filename of the form "
                             f"<output_filename>/rank.<rank>/<stdout | stderr>. The <rank> will be padded with 0 "
                             f"characters to ensure lexicographical order. For MPI, delegates its behavior to mpirun")

    parsed_args = parser.parse_args()
    workers = parsed_args.dispatchers * parsed_args.workers_per_dispatcher

    key = secret.make_secret_key()
    compute = ComputeService(parsed_args.dispatchers, parsed_args.workers_per_dispatcher, key=key)

    compute_config = TfDataServiceConfig(
        dispatchers=parsed_args.dispatchers,
        workers_per_dispatcher=parsed_args.workers_per_dispatcher,
        dispatcher_side=parsed_args.dispatcher_side,
        addresses=compute.addresses(),
        key=key,
    )
    compute_config.write(parsed_args.compute_service_config_file)

    ret = run(compute_worker_fn,
              args=(compute_config,),
              output_filename=parsed_args.output_filename,
              np=workers,
              start_timeout=30,
              disable_cache=True,
              verbose=True)

    compute.shutdown()

    sys.exit(ret)
