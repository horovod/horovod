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

import unittest

from horovod.runner.util.streams import Pipe
from horovod.runner.util.threads import in_thread


class UtilTests(unittest.TestCase):

    def test_pipe(self):
        pipe = Pipe()
        pipe.write('abcdefg')
        read = pipe.read()
        self.assertEqual('abcdefg', read)

    def test_pipe_close_before_read(self):
        pipe = Pipe()
        pipe.write('abcdefg')
        pipe.close()
        self.assertEqual('abcdefg', pipe.read())
        self.assertIsNone(pipe.read())

    def test_pipe_close_empty_before_read(self):
        pipe = Pipe()

        buf = []
        timeout = 1.0

        def read():
            buf.append(pipe.read())

        thread = in_thread(read)
        thread.join(timeout)
        self.assertEqual(True, thread.is_alive())

        pipe.close()
        thread.join(timeout)
        self.assertEqual(False, thread.is_alive())
        self.assertEqual([None], buf)

    def test_pipe_close_empty_before_write(self):
        pipe = Pipe()
        pipe.close()
        with self.assertRaises(RuntimeError):
            pipe.write('abcdefg')

    def test_pipe_close_before_write(self):
        pipe = Pipe()
        pipe.write('abcdefg')
        pipe.close()
        with self.assertRaises(RuntimeError):
            pipe.write('hijklmn')

    def test_pipe_close_blocked_write(self):
        timeout = 1.0
        pipe = Pipe()
        pipe.write('abcdefg')

        thread = in_thread(pipe.write, ('hijklmn',))
        thread.join(timeout)
        self.assertEqual(True, thread.is_alive())

        pipe.close()
        with self.assertRaises(RuntimeError):
            pipe.write('opqrstu')
        thread.join(timeout)
        self.assertEqual(False, thread.is_alive())

    def test_pipe_read_length(self):
        pipe = Pipe()
        pipe.write('abcdefg')
        for c in 'abcdefg':
            r = pipe.read(length=1)
            self.assertEqual(c, r)

        pipe.write('∀∁∂∃∄∅')
        self.assertEqual('∀∁∂', pipe.read(length=3))
        self.assertEqual('∃∄∅', pipe.read(length=3))

        pipe.write('∍∎∏∐∑⌚⌛⛄✅…')
        self.assertEqual('∍∎∏∐∑⌚', pipe.read(length=6))
        self.assertEqual('⌛⛄✅…', pipe.read(length=6))

        pipe.write('∆∇∈∉')
        self.assertEqual('∆∇∈∉', pipe.read(length=4))

        pipe.write('∊∋∌')
        self.assertEqual('∊∋∌', pipe.read(length=4))

        pipe.write('∀∁∂∃∄∅')
        self.assertEqual('∀∁∂', pipe.read(length=3))
        self.assertEqual('∃∄∅', pipe.read())

    def test_pipe_blocking(self):
        buf = []
        timeout = 1.0

        def read():
            buf.append(pipe.read())

        def write(content):
            pipe.write(content)

        pipe = Pipe()

        pipe.write('first')
        thread = in_thread(write, ('second',))
        thread.join(timeout)

        self.assertEqual(True, thread.is_alive())
        self.assertEqual('first', pipe.read())
        self.assertEqual('second', pipe.read())
        thread.join(timeout)
        self.assertEqual(False, thread.is_alive())

        thread = in_thread(read)
        thread.join(timeout)
        self.assertEqual(True, thread.is_alive())

        pipe.write('third')
        thread.join(timeout)
        self.assertEqual(False, thread.is_alive())
        self.assertEqual(['third'], buf)

    def test_pipe_flush(self):
        pipe = Pipe()
        pipe.flush()

    def test_pipe_with_bytes(self):
        pipe = Pipe()
        pipe.write('∀∁∂∃∄∅'.encode('utf8'))
        self.assertEqual(b'\xe2\x88', pipe.read(2))
        self.assertEqual(b'\x80\xe2\x88\x81', pipe.read(4))
        self.assertEqual(b'\xe2\x88\x82\xe2\x88\x83\xe2\x88', pipe.read(8))
        self.assertEqual(b'\x84\xe2\x88\x85', pipe.read())
