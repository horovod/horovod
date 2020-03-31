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

try:
    from shlex import quote
except ImportError:
    from pipes import quote

from horovod.spark.task import task_exec
from horovod.run.common.util import codec


def main(driver_addresses, settings):
    # change current working dir to where the Spark worker runs
    # because orted runs this script where mpirun was executed
    # this env var is injected by the Spark task service
    work_dir = os.environ.get('HOROVOD_SPARK_WORK_DIR')
    if work_dir:
        cwd = os.getcwd()

        # add current working dir to sys.path
        # this makes python code where mpirun is executed available after changing cwd
        if cwd not in sys.path:
            sys.path.insert(1, cwd)  # don't put it in front as that is usually .
            print("Inserted cwd at position 1 into sys.path: {}".format(sys.path))

        # adjust PYTHONPATH according to above sys.path change
        if os.environ.get('PYTHONPATH'):
            os.environ['PYTHONPATH'] = os.pathsep.join([cwd, os.environ['PYTHONPATH']])
            if settings.verbose >= 2:
                print("Prepended cwd to PYTHONPATH: {}".format(os.environ['PYTHONPATH']))

        if settings.verbose >= 2:
            print("Changing cwd from {} to {}".format(os.getcwd(), work_dir))
        os.chdir(work_dir)

    task_exec(driver_addresses, settings, 'OMPI_COMM_WORLD_RANK')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s <driver addresses> <settings>' % sys.argv[0])
        sys.exit(1)
    main(codec.loads_base64(sys.argv[1]), codec.loads_base64(sys.argv[2]))
