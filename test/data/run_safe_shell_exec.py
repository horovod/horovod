"""
The purpose of this file is to execute a safe_shell_exec command in an isolated Python interpreter that doesn't
share resources with the main unit testing interpreter.

Python runs a global process called Semaphore Tracker that is used across processes spawned with the multiprocessing
API. If a process spawned by multiprocessing is hard killed while holding semaphores, then they will not be properly
cleaned up and will be effectively leaked.

Note that this is only a problem at test time, as in practice the process that is hard-killed will not be the child of
yet another process.
"""

import sys
import time

from horovod.run.common.util import safe_shell_exec


class FakeEvent(object):
    def wait(self):
        time.sleep(999)


if __name__ == '__main__':
    cmd = ' '.join([sys.executable] + sys.argv[1:])

    # Mock out the event to avoid leaking semaphores
    safe_shell_exec._create_event = lambda ctx: FakeEvent()

    safe_shell_exec.execute(cmd)
