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

import logging

from horovod.runner.common.util import codec
from horovod.runner.http.http_server import RendezvousHandler

# GET methods
GET_RANK_AND_SIZE = 'rank_and_size'

# PUT methods
PUT_WORKER_ADDRESSES = 'worker_addresses'


def create_rendezvous_handler(driver):
    class ElasticRendezvousHandler(RendezvousHandler):
        def _get_value(self, scope, key):
            if scope == GET_RANK_AND_SIZE:
                host, local_rank = key.split(':')
                return self._get_rank_and_size(host, int(local_rank))

            return super(RendezvousHandler, self)._get_value(scope, key)

        def _get_rank_and_size(self, host, local_rank):
            logging.info('_get_rank_and_size: {} {}'.format(host, local_rank))
            driver.record_ready(host, local_rank)
            slot_info = driver.get_slot_info(host, local_rank)
            logging.info('rank and size: {} {}'.format(slot_info.rank, slot_info.size))
            return slot_info.to_response_string().encode('ascii')

        def _put_value(self, scope, key, value):
            if scope == PUT_WORKER_ADDRESSES:
                host, local_rank = key.split(':')
                addresses, secret_key = codec.loads_base64(value)
                self._put_worker_addresses(host, int(local_rank), addresses, secret_key)

            super(RendezvousHandler, self)._put_value(scope, key, value)

        def _put_worker_addresses(self, host, local_rank, addresses, secret_key):
            driver.register_worker_server(host, local_rank, addresses, secret_key)

    return ElasticRendezvousHandler
