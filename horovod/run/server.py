# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

import sys
import os

from horovod.spark.util import codec, secret
from horovod.spark.driver.driver_service import BasicService


def main(service_name):
    key = codec.loads_base64(os.environ[secret.HOROVOD_SECRET_KEY])
    try:
        service = BasicService(service_name, key)
        print('SERVER LAUNCH SUCCESSFUL on '
              '{service_name}.'.format(
            service_name=codec.dumps_base64(service_name)))
        print('PORT IS: {port} EOM'.format(
            port=codec.dumps_base64(service.get_port())))
    except Exception as e:
        print('SERVER LAUNCH FAILED.')
        print(e.message)
        exit(1)

    while True:
        pass


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: %s <driver addresses>' % sys.argv[0])
        sys.exit(1)
    main(codec.loads_base64(sys.argv[1]))
