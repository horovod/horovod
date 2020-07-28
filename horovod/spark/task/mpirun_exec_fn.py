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

import os
import sys

from horovod.spark.task import task_exec
from horovod.runner.common.util import codec


def main(driver_addresses, settings):
    # prepend HOROVOD_SPARK_PYTHONPATH to PYTHONPATH
    if 'HOROVOD_SPARK_PYTHONPATH' in os.environ:
        ppath = os.environ['HOROVOD_SPARK_PYTHONPATH']

        # add injected HOROVOD_SPARK_PYTHONPATH to sys.path
        for p in reversed(ppath.split(os.pathsep)):
            sys.path.insert(1, p)  # don't put it in front which is usually .

        if 'PYTHONPATH' in os.environ:
            ppath = os.pathsep.join([ppath, os.environ['PYTHONPATH']])
        os.environ['PYTHONPATH'] = ppath

    # change current working dir to where the Spark worker runs
    # because orted runs this script where mpirun was executed
    # this env var is injected by the Spark task service
    work_dir = os.environ.get('HOROVOD_SPARK_WORK_DIR')
    if work_dir:
        if settings.verbose >= 2:
            print("Changing cwd from {} to {}".format(os.getcwd(), work_dir))
        os.chdir(work_dir)

    task_exec(driver_addresses, settings, 'OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_LOCAL_RANK')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s <driver addresses> <settings>' % sys.argv[0])
        sys.exit(1)
    main(codec.loads_base64(sys.argv[1]), codec.loads_base64(sys.argv[2]))
