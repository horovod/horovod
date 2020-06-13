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

import queue
import threading


def execute_function_multithreaded(fn,
                                   args_list,
                                   block_until_all_done=True,
                                   max_concurrent_executions=1000):
    """
    Executes fn in multiple threads each with one set of the args in the
    args_list.
    :param fn: function to be executed
    :type fn:
    :param args_list:
    :type args_list: list(list)
    :param block_until_all_done: if is True, function will block until all the
    threads are done and will return the results of each thread's execution.
    :type block_until_all_done: bool
    :param max_concurrent_executions:
    :type max_concurrent_executions: int
    :return:
    If block_until_all_done is False, returns None. If block_until_all_done is
    True, function returns the dict of results.
        {
            index: execution result of fn with args_list[index]
        }
    :rtype: dict
    """
    result_queue = queue.Queue()
    worker_queue = queue.Queue()

    for i, arg in enumerate(args_list):
        arg.append(i)
        worker_queue.put(arg)

    def fn_execute():
        while True:
            try:
                arg = worker_queue.get(block=False)
            except queue.Empty:
                return
            exec_index = arg[-1]
            res = fn(*arg[:-1])
            result_queue.put((exec_index, res))

    threads = []
    number_of_threads = min(max_concurrent_executions, len(args_list))

    for _ in range(number_of_threads):
        thread = in_thread(target=fn_execute, daemon=not block_until_all_done)
        threads.append(thread)

    # Returns the results only if block_until_all_done is set.
    results = None
    if block_until_all_done:

        # Because join() cannot be interrupted by signal, a single join()
        # needs to be separated into join()s with timeout in a while loop.
        have_alive_child = True
        while have_alive_child:
            have_alive_child = False
            for t in threads:
                t.join(0.1)
                if t.is_alive():
                    have_alive_child = True

        results = {}
        while not result_queue.empty():
            item = result_queue.get()
            results[item[0]] = item[1]

        if len(results) != len(args_list):
            raise RuntimeError(
                'Some threads for func {func} did not complete '
                'successfully.'.format(func=fn.__name__))
    return results


def in_thread(target, args=(), name=None, daemon=True, silent=False):
    """
    Executes the given function in background.
    :param target: function
    :param args: function arguments
    :param name: name of the thread
    :param daemon: run as daemon thread, do not block until thread is doe
    :param silent: swallows exceptions raised by target silently
    :return background thread
    """
    if not isinstance(args, tuple):
        raise ValueError('args must be a tuple, not {}, for a single argument use (arg,)'
                         .format(type(args)))

    if silent:
        def fn(*args):
            try:
                target(*args)
            except:
                pass
    else:
        fn = target

    bg = threading.Thread(target=fn, args=args, name=name)
    bg.daemon = daemon
    bg.start()
    return bg


def on_event(event, func, args=(),
             stop=None, check_stop_interval_s=1.0,
             daemon=True, silent=False):
    """
    Executes the given function in a separate thread when event is set.
    That threat can be stopped by setting the optional stop event.
    The stop event is check regularly every check_interval_seconds.
    Exceptions will silently be swallowed when silent is True.

    :param event: event that triggers func
    :type event: threading.Event
    :param func: function to trigger
    :param args: function arguments
    :param stop: event to stop thread
    :type stop: threading.Event
    :param check_stop_interval_s: interval in seconds to check the stop event
    :type check_stop_interval_s: float
    :param daemon: event thread is a daemon thread if set to True, otherwise stop event must be given
    :param silent: swallows exceptions raised by target silently
    :return: thread
    """
    if event is None:
        raise ValueError('Event must not be None')

    if not isinstance(args, tuple):
        raise ValueError('args must be a tuple, not {}, for a single argument use (arg,)'
                         .format(type(args)))

    if stop is None:
        if not daemon:
            raise ValueError('Stop event must be given for non-daemon event thread')

        def fn():
            event.wait()
            func(*args)
    else:
        def fn():
            while not event.is_set() and not stop.is_set():
                event.wait(timeout=check_stop_interval_s)
            if not stop.is_set():
                func(*args)

    return in_thread(fn, daemon=daemon, silent=silent)
