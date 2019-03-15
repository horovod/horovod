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


def host_hash():
    hostname = socket.gethostname()
    ns = _namespaces()
    return '%s-%s' % (hostname, hashlib.md5(ns.encode('ascii')).hexdigest())
