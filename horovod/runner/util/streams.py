import threading


class Pipe:
    """A pipe that can be written and read concurrently. Buffers the last written string only."""
    def __init__(self):
        self._buf = None
        self._offs = None
        self._wait_cond = threading.Condition()
        self._closed = False

    def write(self, buf):
        self._wait_cond.acquire()
        try:
            while self._buf is not None and not self._closed:
                self._wait_cond.wait()

            if self._closed:
                raise RuntimeError('Pipe is closed')

            self._buf = buf
            self._offs = 0
        finally:
            self._wait_cond.notify_all()
            self._wait_cond.release()

    def read(self, length=-1):
        self._wait_cond.acquire()
        try:
            while self._buf is None and not self._closed:
                self._wait_cond.wait()

            if self._buf is None:
                return None

            if 0 < length < len(self._buf) - self._offs:
                end = self._offs + length
                buf = self._buf[self._offs:end]
                self._offs = end
            else:
                buf = self._buf[self._offs:]
                self._buf = None

            return buf
        finally:
            self._wait_cond.notify_all()
            self._wait_cond.release()

    def flush(self):
        pass

    def close(self):
        self._wait_cond.acquire()
        try:
            self._closed = True
        finally:
            self._wait_cond.notify_all()
            self._wait_cond.release()
