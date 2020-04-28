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
