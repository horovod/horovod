# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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

from __future__ import absolute_import

import functools

from six.moves import queue

from horovod.common.exceptions import HorovodInternalError, WorkersAvailableException
from horovod.run.elastic.worker import WorkerNotificationManager


notification_manager = WorkerNotificationManager()


class State(object):
    def __init__(self):
        self._host_messages = queue.Queue()
        self._known_hosts = set()
        self._reset_callbacks = []

    def register_reset_callbacks(self, callbacks):
        self._reset_callbacks.extend(callbacks)

    def on_reset(self):
        for callback in self._reset_callbacks:
            callback()

    def on_hosts_added(self, hosts):
        for host in hosts:
            self._host_messages.put(host)

    def commit(self):
        self.save()
        self._update_known_hosts()

    def save(self):
        raise NotImplementedError()

    def restore(self):
        raise NotImplementedError()

    def sync(self):
        raise NotImplementedError()

    def _update_known_hosts(self):
        if not self._host_messages.empty():
            host = self._host_messages.get()
            if host not in self._known_hosts:
                self._known_hosts.add(host)
                raise WorkersAvailableException()


class AbstractObjectState(State):
    def __init__(self, **kwargs):
        self._saved_state = kwargs
        self._set_attrs()
        super(AbstractObjectState, self).__init__()

    def save(self):
        new_state = {}
        for attr in self._saved_state.keys():
            new_state[attr] = getattr(self, attr)
        self._saved_state = new_state

    def restore(self):
        self._set_attrs()

    def sync(self):
        raise NotImplementedError()

    def _set_attrs(self):
        for attr, value in self._saved_state.items():
            setattr(self, attr, value)


def run_fn(func, hvd):
    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        notification_manager.init()
        notification_manager.register_listener(state)

        try:
            reset_required = False
            while True:
                if reset_required:
                    _reset(state, hvd)

                state.sync()
                try:
                    print('Call the Function {}'.format(hvd.rank()))
                    return func(state, *args, **kwargs)
                except HorovodInternalError:
                    print('HorovodInternalError {}'.format(hvd.rank()))
                    state.restore()
                except WorkersAvailableException:
                    print('WorkersAvailableException {}'.format(hvd.rank()))
                    pass
                reset_required = True
        finally:
            notification_manager.remove_listener(state)
    return wrapper


def _reset(state, hvd):
    rnk = hvd.rank()
    print('SHUTDOWN {}'.format(rnk))
    hvd.shutdown()
    print('RINIT {}'.format(rnk))
    hvd.init()
    print('RESET {}'.format(rnk))
    state.on_reset()
