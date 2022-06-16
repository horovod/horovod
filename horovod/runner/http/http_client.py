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
from urllib.error import HTTPError, URLError
from urllib.request import Request
from urllib.request import urlopen

from horovod.runner.common.util import codec


def read_data_from_kvstore(addr, port, scope, key):
    try:
        url = "http://{addr}:{port}/{scope}/{key}".format(
            addr=addr, port=str(port), scope=scope, key=key
        )
        req = Request(url)
        resp = urlopen(req)
        # TODO: remove base64 encoding because base64 is not efficient
        return codec.loads_base64(resp.read())
    except (HTTPError, URLError) as e:
        raise RuntimeError("Read data from KVStore server failed.", e)


def put_data_into_kvstore(addr, port, scope, key, value):
    try:
        url = "http://{addr}:{port}/{scope}/{key}".format(
            addr=addr, port=str(port), scope=scope, key=key
        )
        req = Request(url, data=codec.dumps_base64(value, to_ascii=False))
        req.get_method = lambda: "PUT"  # for urllib2 compatibility
        urlopen(req)
    except (HTTPError, URLError) as e:
        raise RuntimeError("Put data input KVStore server failed.", e)
