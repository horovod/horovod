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
import re
import subprocess
import sys
import threading
import time

from horovod.run.util.threads import in_thread, on_event

GRACEFUL_TERMINATION_TIME_S = 5


def forward_stream(src_stream, dst_stream, prefix, index):
    def prepend_context(line, rank, prefix):
        localtime = time.asctime(time.localtime(time.time()))
        return '{time}[{rank}]<{prefix}>:{line}'.format(
            time=localtime,
            rank=str(rank),
            prefix=prefix,
            line=line
        )

    line_buffer = ''
    while True:
        text = os.read(src_stream.fileno(), 1000)
        if not isinstance(text, str):
            text = text.decode('utf-8')
        if not text or len(text) == 0:
            break

        for line in re.split('([\r\n])', text):
            line_buffer += line
            if line == '\r' or line == '\n':
                if index is not None:
                    line_buffer = prepend_context(line_buffer, index, prefix)

                dst_stream.write(line_buffer)
                dst_stream.flush()
                line_buffer = ''

    # flush the line buffer if it is not empty
    if len(line_buffer):
        if index is not None:
            line_buffer = prepend_context(line_buffer, index, prefix)
        dst_stream.write(line_buffer)
        dst_stream.flush()

    src_stream.close()


def execute(command, env=None, stdout=None, stderr=None, index=None, events=None):
    process = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Redirect command stdout & stderr to provided streams or sys.stdout/sys.stderr.
    # This is useful for Jupyter Notebook that uses custom sys.stdout/sys.stderr or
    # for redirecting to a file on disk.
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr

    stdout_fwd = in_thread(target=forward_stream, args=(process.stdout, stdout, 'stdout', index))
    stderr_fwd = in_thread(target=forward_stream, args=(process.stderr, stderr, 'stderr', index))

    # TODO: Currently this requires explicitly declaration of the events and signal handler to set
    #  the event (gloo_run.py:_launch_jobs()). Need to figure out a generalized way to hide this behind
    #  interfaces.
    stop = threading.Event()
    events = events or []
    event_handles = []
    for event in events:
        # with silent=True because the process may have already been killed elsewhere
        event_handles.append(on_event(event, process.kill, stop=stop, silent=True))

    try:
        exit_code = process.wait()
    finally:
        stop.set()

    stdout_fwd.join()
    stderr_fwd.join()
    for handle in event_handles:
        handle.join()

    return exit_code
