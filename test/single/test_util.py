from runner.util.streams import Pipe
from horovod.runner.util.threads import in_thread

import unittest


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
