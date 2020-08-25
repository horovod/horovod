# Copyright IBM Corp. 2020. All Rights Reserved.
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

import io
import os

import yaml

from horovod.common.util import _cache
from horovod.runner.common.util import safe_shell_exec
from horovod.runner.util.remote import get_remote_command


class LSFUtils:
    """LSF Utilities"""
    _CSM_ALLOCATION_QUERY = "/opt/ibm/csm/bin/csm_allocation_query"
    _CSM_NODE_QUERY = "/opt/ibm/csm/bin/csm_node_attributes_query"
    _LSCPU_CMD = "LANG=en_US.utf8 lscpu"
    _THREAD_KEY= "Thread(s) per core"
    _csm_allocation_info = {}

    @staticmethod
    def using_lsf():
        """Returns True if LSF was used to start the current process."""
        return "LSB_JOBID" in os.environ

    @staticmethod
    def get_allocation_info():
        """Returns and sets the static CSM allocation info."""
        if not LSFUtils._csm_allocation_info:
            lsf_allocation_id = os.environ["CSM_ALLOCATION_ID"].strip()
            output = io.StringIO()
            exit_code = safe_shell_exec.execute("{cmd} -a {allocation}".format(
                cmd=LSFUtils._CSM_ALLOCATION_QUERY, allocation=lsf_allocation_id),
                stdout=output, stderr=output)
            if exit_code != 0:
                raise RuntimeError(
                    "{cmd} failed with exit code {exit_code}".format(
                        cmd=LSFUtils._CSM_ALLOCATION_QUERY, exit_code=exit_code))
            LSFUtils._csm_allocation_info = yaml.safe_load(output.getvalue())
            # Fetch the total number of cores and gpus for the first host
            output = io.StringIO()
            exit_code = safe_shell_exec.execute("{cmd} -n {node}".format(
                cmd=LSFUtils._CSM_NODE_QUERY,
                node=LSFUtils._csm_allocation_info["compute_nodes"][0]),
                stdout=output, stderr=output)
            if exit_code != 0:
                raise RuntimeError(
                    "{cmd} failed with exit code {exit_code}".format(
                        cmd=LSFUtils._CSM_NODE_QUERY, exit_code=exit_code))
            node_output = yaml.safe_load(output.getvalue())
            total_core_count = (int(node_output["Record_1"]["discovered_cores"]) -
                               int(node_output["Record_1"]["discovered_sockets"]) * LSFUtils._csm_allocation_info["isolated_cores"])
            LSFUtils._csm_allocation_info["compute_node_cores"]= total_core_count
            LSFUtils._csm_allocation_info["compute_node_gpus"] = int(node_output["Record_1"]["discovered_gpus"])
            # Sorting LSF hostnames
            LSFUtils._csm_allocation_info["compute_nodes"].sort()
        return LSFUtils._csm_allocation_info

    @staticmethod
    def get_compute_hosts():
        """Returns the list of LSF compute hosts."""
        return LSFUtils.get_allocation_info()["compute_nodes"]

    @staticmethod
    def get_num_cores():
        """Returns the number of cores per node."""
        return LSFUtils.get_allocation_info()["compute_node_cores"]

    @staticmethod
    def get_num_gpus():
        """Returns the number of gpus per node."""
        return LSFUtils.get_allocation_info()["compute_node_gpus"]

    @staticmethod
    @_cache
    def get_num_processes():
        """Returns the total number of processes."""
        return len(LSFUtils.get_compute_hosts()) * LSFUtils.get_num_gpus()

    @staticmethod
    @_cache
    def get_num_threads():
        """Returns the number of hardware threads."""
        lscpu_cmd = get_remote_command(LSFUtils._LSCPU_CMD, host=LSFUtils.get_compute_hosts()[0])
        output = io.StringIO()
        exit_code = safe_shell_exec.execute(lscpu_cmd, stdout=output, stderr=output)
        if exit_code != 0:
            raise RuntimeError("{cmd} failed with exit code {exit_code}".format(
                cmd=lscpu_cmd, exit_code=exit_code))
        return int(yaml.safe_load(output.getvalue())[LSFUtils._THREAD_KEY])
