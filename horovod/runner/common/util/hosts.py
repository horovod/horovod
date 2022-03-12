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

import collections
import re

from dataclasses import dataclass


class HostInfo:
    def __init__(self, hostname, slots):
        self.hostname = hostname
        self.slots = slots

    @staticmethod
    def from_string(host_string):
        hostname, slots = host_string.strip().split(':')
        return HostInfo(hostname, int(slots))


@dataclass
class SlotInfo:
    hostname: str
    rank: int
    local_rank: int
    cross_rank: int
    size: int
    local_size: int
    cross_size: int

    def to_response_string(self):
        return ','.join(str(v) for v in [self.rank, self.size,
                                         self.local_rank, self.local_size,
                                         self.cross_rank, self.cross_size])


INVALID_SLOT_INFO = SlotInfo(hostname='',
                             rank=-1, local_rank=-1, cross_rank=-1,
                             size=-1, local_size=-1, cross_size=-1)


def parse_host_files(filename):
    """
    Transform the hostfile into a format of
    <IP address> or <host name>:<Number of GPUs>
    :param filename: Should be in <IP address> or <host name> slots=<number of GPUs>
    :return: Comma separated string of <IP address> or <host name>:<Number of GPUs>
    """
    hosts = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            hostname = line.split()[0]
            slots = line.split('=')[1]
            hosts.append('{name}:{slots}'.format(name=hostname, slots=slots))
    return ','.join(hosts)


def parse_hosts_and_slots(hosts):
    host_names = []
    host_to_slots = {}

    host_list = hosts.split(',')
    pattern = re.compile(r'^[\w.-]+:[0-9]+$')
    for host in host_list:
        if not pattern.match(host.strip()):
            raise ValueError('Invalid host input, please make sure it has '
                             'format as : worker-0:2,worker-1:2.')
        hostname, slots = host.strip().split(':')
        host_names.append(hostname)
        host_to_slots[hostname] = int(slots)
    return host_names, host_to_slots


def parse_hosts(hosts_string):
    """Parse a string of comma-separated hostname:slots mappings into a list of HostItem objects.

    :param hosts_string: list of addresses and number of processes on each host.
        For example:
            - 'worker-0:2,worker-1:2'
            - '10.11.11.11:4,10.11.11.12:4'
    :return: a list of HostInfo objects describing host to slot mappings
    :rtype: list[HostInfo]
    """
    return [HostInfo.from_string(host_string) for host_string in hosts_string.split(',')]


def get_host_assignments(hosts, min_num_proc, max_num_proc=None):
    """Assign hosts with process capacities (slots) to ranks in the Horovod process.

    This function will try to allocate as many as possible processes on the same host to leverage
    local network.

    :param hosts: list of HostInfo objects describing host and slot capacity
    :type hosts: list[HostInfo]
    :param min_num_proc: minimum number of processes to be allocated
    :type min_num_proc: int
    :param max_num_proc: (optional) maximum number of processes to be allocated
    :type max_num_proc: int
    :return: a list of the allocation of process on hosts in a `SlotInfo` object.
    :rtype: list[SlotInfo]
    """
    host_ranks = []
    cross_ranks = collections.defaultdict(dict)
    rank = 0
    for host_info in hosts:
        ranks = []
        for local_rank in range(host_info.slots):
            if rank == max_num_proc:
                break

            ranks.append(rank)
            rank += 1

            cross_ranks_at_local = cross_ranks[local_rank]
            cross_ranks_at_local[host_info.hostname] = len(cross_ranks_at_local)

        host_ranks.append((host_info, ranks))

    world_size = rank
    if world_size < min_num_proc:
        raise ValueError('Requested more processes ({}) than there are available slots ({})'
                         .format(min_num_proc, world_size))

    alloc_list = []
    for host_info, ranks in host_ranks:
        local_size = len(ranks)
        for local_rank, rank in enumerate(ranks):
            cross_ranks_at_local = cross_ranks[local_rank]
            cross_rank = cross_ranks_at_local[host_info.hostname]
            cross_size = len(cross_ranks_at_local)

            alloc_list.append(
                SlotInfo(
                    hostname=host_info.hostname,
                    rank=rank,
                    local_rank=local_rank,
                    cross_rank=cross_rank,
                    size=world_size,
                    local_size=local_size,
                    cross_size=cross_size))

    return alloc_list
