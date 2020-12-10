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

import os
import threading

from enum import IntFlag

from horovod.runner.common.util import network, secret
from horovod.runner.elastic.rendezvous import PUT_WORKER_ADDRESSES
from horovod.runner.http.http_client import put_data_into_kvstore


HOROVOD_GLOO_RENDEZVOUS_ADDR = 'HOROVOD_GLOO_RENDEZVOUS_ADDR'
HOROVOD_GLOO_RENDEZVOUS_PORT = 'HOROVOD_GLOO_RENDEZVOUS_PORT'
HOROVOD_GLOO_IFACE = 'HOROVOD_GLOO_IFACE'
HOROVOD_HOSTNAME = 'HOROVOD_HOSTNAME'
HOROVOD_LOCAL_RANK = 'HOROVOD_LOCAL_RANK'

class HostUpdateResult(IntFlag):
    no_update = 0
    removed = 1
    added = 2
    mixed = removed | added


class HostsUpdatedRequest(object):
    """Notifies worker that the set of available hosts/slots has changed."""
    def __init__(self, timestamp, res=HostUpdateResult.no_update):
        self.timestamp = timestamp
        self.res = res


class WorkerNotificationManager(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._service = None
        self._listeners = set()

    def init(self, rendezvous_addr=None, rendezvous_port=None,
             nic=None, hostname=None, local_rank=None):
        with self._lock:
            if self._service:
                return

            rendezvous_addr = rendezvous_addr or os.environ.get(HOROVOD_GLOO_RENDEZVOUS_ADDR)
            if not rendezvous_addr:
                return

            rendezvous_port = rendezvous_port if rendezvous_port is not None else \
                int(os.environ.get(HOROVOD_GLOO_RENDEZVOUS_PORT))
            nic = nic or os.environ.get(HOROVOD_GLOO_IFACE)
            hostname = hostname or os.environ.get(HOROVOD_HOSTNAME)
            local_rank = local_rank if local_rank is not None else \
                int(os.environ.get(HOROVOD_LOCAL_RANK))

            secret_key = secret.make_secret_key()
            self._service = WorkerNotificationService(secret_key, nic, self)

            value = (self._service.addresses(), secret_key)
            put_data_into_kvstore(rendezvous_addr,
                                  rendezvous_port,
                                  PUT_WORKER_ADDRESSES,
                                  self._create_id(hostname, local_rank),
                                  value)

    def register_listener(self, listener):
        self._listeners.add(listener)

    def remove_listener(self, listener):
        self._listeners.remove(listener)

    def handle_hosts_updated(self, timestamp, update_res):
        for listener in self._listeners:
            listener.on_hosts_updated(timestamp, update_res)

    def _create_id(self, hostname, local_rank):
        return '{}:{}'.format(hostname, local_rank)


class WorkerNotificationService(network.BasicService):
    NAME = 'worker notification service'

    def __init__(self, key, nic, manager):
        super(WorkerNotificationService, self).__init__(WorkerNotificationService.NAME,
                                                        key,
                                                        nic)
        self._manager = manager

    def _handle(self, req, client_address):
        if isinstance(req, HostsUpdatedRequest):
            self._manager.handle_hosts_updated(req.timestamp, req.res)
            return network.AckResponse()

        return super(WorkerNotificationService, self)._handle(req, client_address)


class WorkerNotificationClient(network.BasicClient):
    def __init__(self, addresses, key, verbose, match_intf=False):
        super(WorkerNotificationClient, self).__init__(WorkerNotificationService.NAME,
                                                       addresses,
                                                       key,
                                                       verbose,
                                                       match_intf=match_intf)

    def notify_hosts_updated(self, timestamp, update_res):
        self._send(HostsUpdatedRequest(timestamp, update_res))
