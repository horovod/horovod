# Copyright 2021 Uber Technologies, Inc. All Rights Reserved.
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

import threading


class Pipe:
    """
    A pipe that can be written and read concurrently.
    Works with strings and bytes. Buffers the last written string/bytes only.
    """
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
