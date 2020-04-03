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

from distutils.version import LooseVersion

import os
import pyspark

from horovod.run.common.util import codec, secret
from horovod.run.common.service import task_service


class ResourcesRequest(object):
    """Request Spark resources info for this task."""


class ResourcesResponse(object):
    def __init__(self, resources):
        self.resources = resources
        """Dictionary containing resource info."""


class SparkTaskService(task_service.BasicTaskService):
    NAME_FORMAT = 'task service #%d'

    @staticmethod
    def _get_command_env(key):
        # on a Spark cluster we need our train function to see the Spark worker environment
        # this includes PYTHONPATH, HADOOP_TOKEN_FILE_LOCATION and _HOROVOD_SECRET_KEY
        env = os.environ.copy()

        # we inject the secret key here
        env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(key)

        # we also need to provide the current working dir to mpirun_exec_fn.py
        env['HOROVOD_SPARK_WORK_DIR'] = os.getcwd()

        return env

    def __init__(self, index, key, nics, verbose=0):
        super(SparkTaskService, self).__init__(SparkTaskService.NAME_FORMAT % index,
                                               key, nics,
                                               SparkTaskService._get_command_env(key),
                                               verbose)

    def _handle(self, req, client_address):
        if isinstance(req, ResourcesRequest):
            return ResourcesResponse(self._get_resources())

        return super(SparkTaskService, self)._handle(req, client_address)

    def _get_resources(self):
        if LooseVersion(pyspark.__version__) >= LooseVersion('3.0.0'):
            from pyspark import TaskContext
            return TaskContext.get().resources()
        return dict()


class SparkTaskClient(task_service.BasicTaskClient):

    def __init__(self, index, task_addresses, key, verbose, match_intf=False):
        super(SparkTaskClient, self).__init__(SparkTaskService.NAME_FORMAT % index,
                                              task_addresses, key, verbose,
                                              match_intf=match_intf)

    def resources(self):
        resp = self._send(ResourcesRequest())
        return resp.resources
