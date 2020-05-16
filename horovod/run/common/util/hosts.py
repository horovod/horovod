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


class HostInfo:
    def __init__(self, hostname, slots):
        self.hostname = hostname
        self.slots = slots

    @staticmethod
    def from_string(host_string):
        hostname, slots = host_string.strip().split(':')
        return HostInfo(hostname, int(slots))


class SlotInfo:
    def __init__(self, hostname, rank, local_rank, cross_rank, size=None, local_size=None, cross_size=None):
        self.hostname = hostname
        self.rank = rank
        self.size = size
        self.local_rank = local_rank
        self.local_size = local_size
        self.cross_rank = cross_rank
        self.cross_size = cross_size

    def to_response_string(self):
        return ','.join(str(v) for v in [self.rank, self.size,
                                         self.local_rank, self.local_size,
                                         self.cross_rank, self.cross_size])

    def __eq__(self, other):
        if isinstance(other, SlotInfo):
            return self.hostname == other.hostname and \
                   self.rank == other.rank and self.size == other.size and \
                   self.local_rank == other.local_rank and self.local_size == other.local_size and \
                   self.cross_rank == other.cross_rank and self.cross_size == other.cross_size
        return False


INVALID_SLOT_INFO = SlotInfo(hostname='',
                             rank=-1, local_rank=-1, cross_rank=-1,
                             size=-1, local_size=-1, cross_size=-1)


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


def get_host_assignments(hosts, min_np, max_np=None):
    """Assign hosts with process capacities (slots) to ranks in the Horovod process.

    This function will try to allocate as many as possible processes on the same host to leverage
    local network.

    :param hosts: list of HostInfo objects describing host and slot capacity
    :type hosts: list[HostInfo]
    :param np: total number of processes to be allocated
    :type np: int
    :return: a list of the allocation of process on hosts in a AllocInfo object.
            Members in the object include: hostname, rank, local_rank, cross_rank,
            total_size, local_size, cross_size
    :rtype: list[SlotInfo]
    """
    rank = 0
    alloc_list = []

    # key: local_rank; value: cross_size for this local_rank
    local_sizes = collections.defaultdict(int)
    # key: cross_rank; value: local_size for this cross_rank
    cross_sizes = collections.defaultdict(int)

    # allocate processes into slots
    for host_idx, host_info in enumerate(hosts):
        for local_rank in range(host_info.slots):
            if rank == max_np:
                break
            cross_rank = host_idx
            alloc_list.append(
                SlotInfo(
                    host_info.hostname,
                    rank,
                    local_rank,
                    cross_rank))
            cross_sizes[local_rank] += 1
            local_sizes[cross_rank] += 1
            rank += 1

    if rank < min_np:
        raise ValueError('Requested more processes ({}) than there are available slots ({})'
                         .format(min_np, rank))

    # Fill in the local_size and cross_size because we can only know these number after
    # allocation is done.
    for alloc_item in alloc_list:
        alloc_item.local_size = local_sizes[alloc_item.cross_rank]
        alloc_item.cross_size = cross_sizes[alloc_item.local_rank]
        alloc_item.size = rank
    return alloc_list
