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
import socket
from typing import Optional, List

import psutil


class NoValidAddressesFound(Exception):
    pass


def local_addresses(nics: Optional[List[str]] = None, port: Optional[int] = None):
    """
    Return all network addresses of the local host. Returns a dict with nics as keys and
    lists of addresses as values. If nics are given to this method, only those are contained
    in the result, if they exist. Addresses are either address and port tuples if port is given
    to this method, and plain address strings otherwise.

    Raises NoValidAddressesFound if nics are given but no such nic is found.
    """
    result = {}
    for intf, intf_addresses in psutil.net_if_addrs().items():
        if nics and intf not in nics:
            continue
        for addr in intf_addresses:
            if addr.family == socket.AF_INET:
                if intf not in result:
                    result[intf] = []
                if port:
                    result[intf].append((addr.address, port))
                else:
                    result[intf].append(addr.address)

    if not result and nics:
        raise NoValidAddressesFound(
            f'No available network interface found matching user provided interfaces {nics}, '
            f'existing nics: {list(psutil.net_if_addrs().keys())}'
        )

    return result
