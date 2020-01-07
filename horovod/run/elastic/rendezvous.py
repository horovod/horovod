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
# =============================================================================

from __future__ import absolute_import

from horovod.run.common.util import codec
from horovod.run.http.http_server import RendezvousHandler

# GET methods
GET_RANK_AND_SIZE = 'rank_and_size'

# PUT methods
PUT_WORKER_PORT = 'worker_port'


def create_rendezvous_handler(driver):
    class ElasticRendezvousHandler(RendezvousHandler):
        def _get_value(self, scope, key):
            if scope == GET_RANK_AND_SIZE:
                host, local_rank = key.split(':')
                return self._get_rank_and_size(host, int(local_rank))

            return super(RendezvousHandler, self)._get_value(scope, key)

        def _get_rank_and_size(self, host, local_rank):
            driver.record_ready(host, local_rank)
            slot_info = driver.get_slot_info(host, local_rank)
            return slot_info.to_response_string()

        def _put_value(self, scope, key, value):
            if scope == PUT_WORKER_PORT:
                host, local_rank = key.split(':')
                addresses, port = codec.loads_base64(value)
                self._put_worker_port(host, int(local_rank), addresses, port)

            super(RendezvousHandler, self)._put_value(scope, key, value)

        def _put_worker_port(self, host, local_rank, addresses, port):
            driver.register_worker_server(host, local_rank, addresses, port)

    return ElasticRendezvousHandler
