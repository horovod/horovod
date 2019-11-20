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

import hashlib
import os
import socket

NAMESPACE_PATH = '/proc/self/ns'


def _namespaces():
    hash = ''
    if os.path.exists(NAMESPACE_PATH):
        for file in os.listdir(NAMESPACE_PATH):
            if hash != '':
                hash += ' '
            hash += os.readlink(os.path.join(NAMESPACE_PATH, file))
    return hash


def _hash(string):
    return hashlib.md5(string.encode('ascii')).hexdigest()


def host_hash():
    """
    Computes a hash that represents this host, a unit of processing power that shares memory.

    The hash contains the part of the hostname, e.g. `host` for hostname `host.example.com`,
    plus a hash derived from the full hostname and further information about this machine.

    This considers environment variable CONTAINER_ID which is present when running Spark via YARN.
    A YARN container does not share memory with other containers on the same host,
    so it must be considered a `host` in the sense of the `host_hash`.
    """
    hostname = socket.gethostname()
    host = hostname.split('.')[0]
    ns = _namespaces()
    host_info = '{hostname}-{ns}'.format(hostname=hostname, ns=ns)

    # when running in YARN containers we need to consider a container a host
    # otherwise we might violate resource allocation if we run all tasks of a host in one container
    # see [issues 1497](https://github.com/horovod/horovod/issues/1497) for details
    container = os.environ.get("CONTAINER_ID")
    if container is not None:
        host_info = '{host_info}-{container}'.format(host_info=host_info, container=container)

    return '{host}-{hash}'.format(host=host, hash=_hash(host_info))
