# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import psutil
import random
import socket

from socket import AF_INET
from psutil import net_if_addrs

from horovod.run.util import threads

def _get_local_host_addresses():
    local_addresses = []
    for intf_info_list in psutil.net_if_addrs().values():
        for intf_info in intf_info_list:
            if intf_info.family == socket.AF_INET:
                local_addresses.append(intf_info.address)
    return local_addresses


def get_local_host_intfs():
    return set(psutil.net_if_addrs().keys())


def filter_local_addresses(all_host_names):
    local_addresses = _get_local_host_addresses()

    def resolve_host_name(host_name):
        try:
            return socket.gethostbyname(host_name)
        except socket.gaierror:
            return None

    args_list = [[host] for host in all_host_names]
    host_addresses = threads.execute_function_multithreaded(
        resolve_host_name, args_list)

    # host_addresses is a map
    remote_host_names = []
    for i in range(len(all_host_names)):
        host_address = host_addresses[i]
        host_name = all_host_names[i]

        if not host_address or host_address not in local_addresses:
            remote_host_names.append(host_name)

    return remote_host_names


# Given server factory, find a usable port
def find_port(server_factory):
    min_port = 1024
    max_port = 65536
    num_ports = max_port - min_port
    start_port = random.randrange(0, num_ports)
    for port_offset in range(num_ports):
        try:
            port = min_port + (start_port + port_offset) % num_ports
            addr = ('', port)
            server = server_factory(addr)
            return server, port
        except Exception as e:
            pass

    raise Exception('Unable to find a port to bind to.')


def _get_driver_ip(nics):
    """
    :param nics: object return by `_driver_fn`
    :return: driver ip. We make sure all workers can connect to this ip.
    """
    iface = list(nics)[0]
    driver_ip = None
    for addr in net_if_addrs()[iface]:
        if addr.family == AF_INET:
            driver_ip = addr.address

    if not driver_ip:
        raise RuntimeError(
            'Cannot find an IPv4 address of the common interface.')

    return driver_ip

