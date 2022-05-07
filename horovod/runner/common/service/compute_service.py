# Copyright 2022 G-Research. All Rights Reserved.
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

from horovod.runner.common.util import timeout, network
from horovod.runner.common.util.network import AckResponse
from horovod.runner.common.util.timeout import TimeoutException
from horovod.runner.util.threads import in_thread

"""
Items sent over the Wire between Compute Client and Service.
"""


class RegisterDispatcherRequest(object):
    """Registers a dispatcher server address along with a dispatcher id."""

    def __init__(self, dispatcher_id, dispatcher_address):
        self.dispatcher_id = dispatcher_id
        """Id of the dispatcher server (0 indexed)"""

        self.dispatcher_address = dispatcher_address
        """Address of the dispatcher"""


class WaitForDispatcherRegistrationRequest(object):
    """Wait for the given dispatcher to register. Blocks until the dispatcher registers or the timeout occurs."""

    def __init__(self, dispatcher_id, timeout):
        self.dispatcher_id = dispatcher_id
        """Dispatcher id"""

        self.timeout = timeout
        """Wait timeout in seconds"""


class WaitForDispatcherRegistrationResponse(object):
    """Response that the dispatcher has registered, providing its address."""
    def __init__(self, dispatcher_address):
        self.dispatcher_address = dispatcher_address
        """Address of the requested dispatcher."""


class RegisterDispatcherWorkerRequest(object):
    """Registers a worker server for a dispatcher server."""

    def __init__(self, dispatcher_id, worker_id):
        self.dispatcher_id = dispatcher_id
        """Id of the dispatcher server (0 indexed)"""

        self.worker_id = worker_id
        """Id of the worker server (0 indexed)"""


class WaitForDispatcherWorkerRegistrationRequest(object):
    """Wait for all workers of the given dispatcher to register.
    Blocks until all workers registers or the timeout occurs."""

    def __init__(self, dispatcher_id, timeout):
        self.dispatcher_id = dispatcher_id
        """Dispatcher id"""

        self.timeout = timeout
        """Wait timeout in seconds"""


class ShutdownRequest(object):
    """Initiate the shutdown of the compute service as it is no longer needed."""


class WaitForShutdownRequest(object):
    """Wait for the compute service to shutdown. Blocks until the shutdown is initiated."""


"""
ComputeService is used to communicate between training driver, training tasks, compute driver and compute tasks.
It is ML framework agnostic, though currently only used for Tensorflow data service (tf.data.experimental.service).
It orchestrates synchronization between tf.DispatchServer, tf.WorkerServer, training and compute tasks.

ComputeClient is used to query and change the internal state of the ComputeService.
"""


class ComputeService(network.BasicService):
    NAME = "Compute service"

    def __init__(self, dispatchers, workers_per_dispatcher, key, nics=None):
        if dispatchers <= 0:
            raise ValueError(f'The number of dispatchers must be larger than 0: {dispatchers}')
        if workers_per_dispatcher <= 0:
            raise ValueError(f'The number of workers per dispatcher must be larger than 0: {workers_per_dispatcher}')

        self._max_dispatcher_id = dispatchers - 1
        self._dispatcher_addresses = [None] * dispatchers
        self._workers_per_dispatcher = workers_per_dispatcher
        self._dispatcher_worker_ids = [set()] * dispatchers
        self._shutdown = False
        self._wait_cond = threading.Condition()

        super().__init__(ComputeService.NAME, key, nics)

    def _handle(self, req, client_address):
        if isinstance(req, RegisterDispatcherRequest):
            self._wait_cond.acquire()
            try:
                if not 0 <= req.dispatcher_id <= self._max_dispatcher_id:
                    return IndexError(f'Dispatcher id must be within [0..{self._max_dispatcher_id}]: '
                                      f'{req.dispatcher_id}')

                if self._dispatcher_addresses[req.dispatcher_id] is not None and \
                   self._dispatcher_addresses[req.dispatcher_id] != req.dispatcher_address:
                    return ValueError(f'Dispatcher with id {req.dispatcher_id} has already been registered under '
                                      f'different address {self._dispatcher_addresses[req.dispatcher_id]}: '
                                      f'{req.dispatcher_address}')

                self._dispatcher_addresses[req.dispatcher_id] = req.dispatcher_address
                self._wait_cond.notify_all()
            finally:
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, WaitForDispatcherRegistrationRequest):
            self._wait_cond.acquire()
            try:
                if not 0 <= req.dispatcher_id <= self._max_dispatcher_id:
                    return IndexError(f'Dispatcher id must be within [0..{self._max_dispatcher_id}]: '
                                      f'{req.dispatcher_id}')

                tmout = timeout.Timeout(timeout=req.timeout,
                                        message='Timed out waiting for {activity}. Try to find out what takes '
                                                'the dispatcher so long to register or increase timeout.')

                while self._dispatcher_addresses[req.dispatcher_id] is None:
                    self._wait_cond.wait(tmout.remaining())
                    tmout.check_time_out_for(f'dispatcher {req.dispatcher_id} to register')
            except TimeoutException as e:
                return e
            finally:
                self._wait_cond.release()
            return WaitForDispatcherRegistrationResponse(self._dispatcher_addresses[req.dispatcher_id])

        if isinstance(req, RegisterDispatcherWorkerRequest):
            self._wait_cond.acquire()
            try:
                if not 0 <= req.dispatcher_id <= self._max_dispatcher_id:
                    return IndexError(f'Dispatcher id must be within [0..{self._max_dispatcher_id}]: '
                                      f'{req.dispatcher_id}')

                self._dispatcher_worker_ids[req.dispatcher_id].update({req.worker_id})
                self._wait_cond.notify_all()
            finally:
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, WaitForDispatcherWorkerRegistrationRequest):
            # if there is only a single dispatcher, wait for that one instead of the requested one
            dispatcher_id = req.dispatcher_id if self._max_dispatcher_id > 0 else 0

            self._wait_cond.acquire()
            try:
                if not 0 <= req.dispatcher_id <= self._max_dispatcher_id:
                    return IndexError(f'Dispatcher id must be within [0..{self._max_dispatcher_id}]: '
                                      f'{req.dispatcher_id}')

                tmout = timeout.Timeout(timeout=req.timeout,
                                        message='Timed out waiting for {activity}. Try to find out what takes '
                                                'the workers so long to register or increase timeout.')

                while len(self._dispatcher_worker_ids[dispatcher_id]) < self._workers_per_dispatcher:
                    self._wait_cond.wait(tmout.remaining())
                    tmout.check_time_out_for(f'workers for dispatcher {dispatcher_id} to register')
            except TimeoutException as e:
                return e
            finally:
                self._wait_cond.release()
            return network.AckResponse()

        if isinstance(req, ShutdownRequest):
            in_thread(self.shutdown)
            return network.AckResponse()

        if isinstance(req, WaitForShutdownRequest):
            self._wait_cond.acquire()
            try:
                while not self._shutdown:
                    self._wait_cond.wait()
            finally:
                self._wait_cond.release()
            return network.AckResponse()

        return super()._handle(req, client_address)

    def shutdown(self):
        self._wait_cond.acquire()
        try:
            # notify all requests that are waiting for shutdown
            self._shutdown = True
            self._wait_cond.notify_all()
        finally:
            self._wait_cond.release()

        # this will wait until all requests have been completed
        super(ComputeService, self).shutdown()


class ComputeClient(network.BasicClient):

    def __init__(self, compute_addresses, key, verbose=1):
        super().__init__(ComputeService.NAME, compute_addresses, key, verbose)

    def register_dispatcher(self, dispatcher_id, dispatcher_address):
        self._send(RegisterDispatcherRequest(dispatcher_id, dispatcher_address))

    def wait_for_dispatcher_registration(self, dispatcher_id, timeout) -> str:
        resp = self._send(WaitForDispatcherRegistrationRequest(dispatcher_id, timeout))
        return resp.dispatcher_address

    def register_worker_for_dispatcher(self, dispatcher_id, worker_id):
        self._send(RegisterDispatcherWorkerRequest(dispatcher_id, worker_id))

    def wait_for_dispatcher_worker_registration(self, dispatcher_id, timeout):
        self._send(WaitForDispatcherWorkerRegistrationRequest(dispatcher_id, timeout))

    def shutdown(self):
        self._send(ShutdownRequest())

    def wait_for_shutdown(self):
        self._send(WaitForShutdownRequest())

    def _send(self, req, stream=None):
        """Raise exceptions that we retrieve for any request."""
        resp = super(ComputeClient, self)._send(req, stream)
        if isinstance(resp, Exception):
            raise resp
        return resp
