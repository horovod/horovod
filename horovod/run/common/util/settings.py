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


class Settings(object):

    def __init__(self, verbose=0, ssh_port=None, extra_mpi_args=None, key=None, timeout=None,
                 num_hosts=None, num_proc=None, hosts=None, output_filename=None,
                 command=None):
        """
        :param verbose: level of verbosity
        :type verbose: int
        :param ssh_port: SSH port on all the hosts
        :type ssh_port: int
        :param extra_mpi_args: Extra MPI arguments to pass to mpirun
        :type extra_mpi_args: string
        :param key: used for encryption of parameters passed across the hosts
        :type key: str
        :param timeout: has to finish all the checks before this timeout runs
        out.
        :type timeout: horovod.run.common.util.timeout.Timeout
        :param num_hosts: number of horovod hosts
        :type num_hosts: int
        :param num_proc: number of horovod processes (-np)
        :type num_proc: int
        :param hosts: string of hostname with slots number
        :type hosts: string
        :param output_filename: optional filename to redirect stdout / stderr by process
        :try output_filename: string
        :param command: number of horovod processes (-np)
        :type num_proc: int
        """
        self.verbose = verbose
        self.ssh_port = ssh_port
        self.extra_mpi_args = extra_mpi_args
        self.key = key
        self.timeout = timeout
        self.num_hosts = num_hosts
        self.num_proc = num_proc
        self.hosts = hosts
        self.output_filename = output_filename
        self.command = command
