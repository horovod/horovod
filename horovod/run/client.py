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
from horovod.spark.driver.driver_service import BasicClient
from horovod.spark.util.network import NoValidAddressesFound


def main(service_name, addresses):
    """
    :param service_name:
    :param addresses:     # addresses = [(ip, port)]
    :return:
    """

    key = codec.loads_base64(os.environ[secret.HOROVOD_SECRET_KEY])
    valid_interfaces = {}
    try:
        service = BasicClient(service_name, addresses, key)
        print("CLIENT LAUNCH SUCCESSFUL.")
        valid_interfaces = service.addresses()

    except NoValidAddressesFound as e:
        print("CLIENT LAUNCH SUCCESSFUL.")

    print("SUCCESSFUL INTERFACE ADDRESSES {addresses} EOM.".format(
        addresses=codec.dumps_base64(valid_interfaces)))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s <service_name> <list of addresses>' % sys.argv[0])
        sys.exit(1)
    main(codec.loads_base64(sys.argv[1]), codec.loads_base64(sys.argv[2]))
