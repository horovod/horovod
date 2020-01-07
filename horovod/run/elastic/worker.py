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

from horovod.run.common.util import network


class HostAddedRequest(object):
    """Notifies worker that a new host has been made available."""
    def __init__(self, hostname):
        self.hostname = hostname


class HostAddedResponse(object):
    pass


class WorkerNotificationService(network.BasicService):
    NAME = 'worker notification service'

    def __init__(self, key, nic):
        super(WorkerNotificationService, self).__init__(WorkerNotificationService.NAME,
                                                        key,
                                                        nic)

    def _handle(self, req, client_address):
        if isinstance(req, HostAddedRequest):
            return HostAddedResponse()

        return super(WorkerNotificationService, self)._handle(req, client_address)


class WorkerNotificationClient(network.BasicClient):
    def __init__(self, addresses, key, verbose, match_intf=False):
        super(WorkerNotificationClient, self).__init__(WorkerNotificationService.NAME,
                                                       addresses,
                                                       key,
                                                       verbose,
                                                       match_intf=match_intf)

    def notify_host_added(self, host):
        self._send(HostAddedRequest(host))
