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
import signal
import subprocess
import sys
import threading
import time


KILL_TIMEOUT = 5


def kill_children():
    p = psutil.Process()

    # Ask executor to terminate itself and children gracefully.
    for child in p.children():
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    # Escalate to SIGKILL to children (recursively).
    time.sleep(KILL_TIMEOUT)
    for child in p.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def execute(command):
    (r, w) = os.pipe()
    middleman_pid = os.fork()
    if middleman_pid == 0:
        os.close(w)
        os.setpgid(0, 0)

        sigterm_received = threading.Event()

        def set_sigterm_received(signum, frame):
            sigterm_received.set()

        signal.signal(signal.SIGINT, set_sigterm_received)
        signal.signal(signal.SIGTERM, set_sigterm_received)

        def kill_children_if_parent_dies():
            os.read(r, 1)
            kill_children()

        bg = threading.Thread(target=kill_children_if_parent_dies)
        bg.daemon = True
        bg.start()

        def kill_children_if_sigterm_received():
            sigterm_received.wait()
            kill_children()

        bg = threading.Thread(target=kill_children_if_sigterm_received)
        bg.daemon = True
        bg.start()

        executor = subprocess.Popen(command, shell=True)
        exit_code = executor.wait()
        os._exit(exit_code)

    os.close(r)
    try:
        _, status = os.waitpid(middleman_pid, 0)
    except:
        # interrupted, send middleman TERM signal which will terminate children
        os.kill(middleman_pid, signal.SIGTERM)
        while True:
            try:
                _, status = os.waitpid(middleman_pid, 0)
                break
            except:
                # interrupted, wait for middleman to finish
                pass

    exit_code = status >> 8
    return exit_code


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <command>' % sys.argv[0])
        sys.exit(1)
    sys.exit(execute(sys.argv[1]))
