# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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
import psutil
import sys
import subprocess
import threading


def execute(command):
    (r, w) = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(w)
        os.setpgid(0, 0)
        middleman = psutil.Process()

        def kill_children_if_parent_dies():
            os.read(r, 1)
            for child in middleman.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass

        bg = threading.Thread(target=kill_children_if_parent_dies)
        bg.daemon = True
        bg.start()

        executor = subprocess.Popen(command, shell=True)
        exit_code = executor.wait()
        os._exit(exit_code)

    os.close(r)
    _, status = os.wait()
    exit_code = status >> 8
    return exit_code


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <command>' % sys.argv[0])
        sys.exit(1)
    sys.exit(execute(sys.argv[1]))
