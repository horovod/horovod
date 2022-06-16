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
# =============================================================================

import sys

from horovod.runner.common.util.env import get_env_rank_and_size
from horovod.runner.http.http_client import read_data_from_kvstore, put_data_into_kvstore


def main(addr, port):
    func = read_data_from_kvstore(addr, port, 'runfunc', 'func')
    try:
        ret_val = func()
    except BaseException as e:
        sys.stderr.write("User function raise error: {error}".format(error=str(e)))
        raise e

    rank, size = get_env_rank_and_size()
    put_data_into_kvstore(addr, port, 'runfunc_result', str(rank), ret_val)


if __name__ == '__main__':
    _, driver_addr, run_func_server_port_str = sys.argv
    run_func_server_port = int(run_func_server_port_str)
    main(driver_addr, run_func_server_port)
