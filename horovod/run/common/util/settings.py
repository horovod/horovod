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

from __future__ import absolute_import


class BaseSettings(object):
    def __init__(self, num_proc=None, verbose=0, ssh_port=None, extra_mpi_args=None, tcp_flag=None,
                 binding_args=None, key=None, start_timeout=None, output_filename=None,
                 run_func_mode=None, nics=None, elastic=False):
        """
        :param num_proc: number of horovod processes (-np)
        :type num_proc: int
        :param verbose: level of verbosity
        :type verbose: int
        :param ssh_port: SSH port on all the hosts
        :type ssh_port: int
        :param extra_mpi_args: Extra MPI arguments to pass to mpirun
        :type extra_mpi_args: string
        :param tcp_flag: TCP only communication flag
        :type tcp_flag: boolean
        :param binding_args: Process binding arguments
        :type binding_args: string
        :param key: used for encryption of parameters passed across the hosts
        :type key: str
        :param start_timeout: has to finish all the checks before this timeout runs out.
        :type start_timeout: horovod.run.common.util.timeout.Timeout
        :param output_filename: optional filename to redirect stdout / stderr by process
        :type output_filename: string
        :param run_func_mode: whether it is run function mode
        :type run_func_mode: boolean
        :param nics: specify the NICs to be used for tcp network communication.
        :type nics: string
        :param elastic: enable elastic auto-scaling and fault tolerance mode
        :type elastic: boolean
        """
        self.num_proc = num_proc
        self.verbose = verbose
        self.ssh_port = ssh_port
        self.extra_mpi_args = extra_mpi_args
        self.tcp_flag = tcp_flag
        self.binding_args = binding_args
        self.key = key
        self.start_timeout = start_timeout
        self.output_filename = output_filename
        self.run_func_mode = run_func_mode
        self.nics = nics
        self.elastic = elastic
        
    # we do not serialize the key, as it is too risky that it could leak unintentionally
    def __getstate__(self):
        result = self.__dict__.copy()
        result['key'] = None
        return result


class Settings(BaseSettings):
    def __init__(self, num_hosts=None, hosts=None, **kwargs):
        """
        :param num_hosts: number of horovod hosts
        :type num_hosts: int
        :param hosts: string, comma-delimited, of hostname[s] with slots number[s]
        :type hosts: string
        """
        super(Settings, self).__init__(**kwargs)
        self.num_hosts = num_hosts
        self.hosts = hosts
