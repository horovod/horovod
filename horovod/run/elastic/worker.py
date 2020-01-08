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

import os
import threading

from horovod.run.common.util import network, secret
from horovod.run.elastic.rendezvous import PUT_WORKER_ADDRESSES
from horovod.run.http.http_client import put_data_into_kvstore


HOROVOD_GLOO_RENDEZVOUS_ADDR = 'HOROVOD_GLOO_RENDEZVOUS_ADDR'
HOROVOD_GLOO_RENDEZVOUS_PORT = 'HOROVOD_GLOO_RENDEZVOUS_PORT'
HOROVOD_GLOO_IFACE = 'HOROVOD_GLOO_IFACE'
HOROVOD_HOSTNAME = 'HOROVOD_HOSTNAME'
HOROVOD_LOCAL_RANK = 'HOROVOD_LOCAL_RANK'


class HostsAddedRequest(object):
    """Notifies worker that new hosts have been made available."""
    def __init__(self, hosts):
        self.hosts = hosts


class HostsAddedResponse(object):
    pass


class WorkerNotificationManager(object):
    def __init__(self, rendezvous_addr=None, rendezvous_port=None,
                 nic=None, hostname=None, local_rank=None):
        self._lock = threading.Lock()
        self._service = None
        self._listeners = set()
        self._rendezvous_addr = rendezvous_addr or os.environ.get(HOROVOD_GLOO_RENDEZVOUS_ADDR)
        self._rendezvous_port = rendezvous_port if rendezvous_port is not None else \
            int(os.environ.get(HOROVOD_GLOO_RENDEZVOUS_PORT))
        self._nic = nic or os.environ.get(HOROVOD_GLOO_IFACE)
        self._hostname = hostname or os.environ.get(HOROVOD_HOSTNAME)
        self._local_rank = local_rank if local_rank is not None else \
            int(os.environ.get(HOROVOD_LOCAL_RANK))

    def init(self):
        with self._lock:
            if self._service:
                return

            secret_key = secret.make_secret_key()
            self._service = WorkerNotificationService(secret_key, self._nic, self)

            value = (self._service.addresses(), secret_key)
            put_data_into_kvstore(self._rendezvous_addr, self._rendezvous_port,
                                  PUT_WORKER_ADDRESSES, self._worker_id(), value)

    def register_listener(self, listener):
        self._listeners.add(listener)

    def remove_listener(self, listener):
        self._listeners.remove(listener)

    def handle_hosts_added(self, hosts):
        for listener in self._listeners:
            listener.on_hosts_added(hosts)

    def _worker_id(self):
        return '{}:{}'.format(self._hostname, self._local_rank)


class WorkerNotificationService(network.BasicService):
    NAME = 'worker notification service'

    def __init__(self, key, nic, manager):
        super(WorkerNotificationService, self).__init__(WorkerNotificationService.NAME,
                                                        key,
                                                        nic)
        self._manager = manager

    def _handle(self, req, client_address):
        if isinstance(req, HostsAddedRequest):
            self._manager.handle_hosts_added(req.hosts)
            return HostsAddedResponse()

        return super(WorkerNotificationService, self)._handle(req, client_address)


class WorkerNotificationClient(network.BasicClient):
    def __init__(self, addresses, key, verbose, match_intf=False):
        super(WorkerNotificationClient, self).__init__(WorkerNotificationService.NAME,
                                                       addresses,
                                                       key,
                                                       verbose,
                                                       match_intf=match_intf)

    def notify_hosts_added(self, hosts):
        self._send(HostsAddedRequest(hosts))
