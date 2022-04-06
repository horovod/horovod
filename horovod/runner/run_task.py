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
from urllib.error import URLError

from horovod.runner.common.util.env import get_env_rank_and_size
from horovod.runner.http.http_client import read_data_from_kvstore, put_data_into_kvstore


def _get_func(addrs, port, timeout=5):
    # we try all provided addresses to connect to the kvstore
    # the first addr that works will be returned, together with the run func
    # we give each IP 5 seconds timeout, if that is not enough, the driver is not really well reachable
    for addr in addrs:
        try:
            func = read_data_from_kvstore(addr, port, 'runfunc', 'func', timeout=timeout)
            return addr, func
        except RuntimeError as e:
            # when the RuntimeError is caused by an URLError, the addr is probably not reachable for us
            if len(e.args) >= 2 and isinstance(e.args[1], URLError):
                # provide a warning when multiple addrs are provided on how to improve this situation
                if len(addrs) > 1:
                    print(f'Driver is not reachable at {addr} within {timeout} seconds. '
                          f'Consider restricting the driver to some NICs, '
                          f'which reduces the number of IPs probed here.')
                continue
    raise ValueError(f'None of the provided IPs could be used to connect to driver''s KV store: {", ".join(addrs)}')


def main(addrs, port):
    addr, func = _get_func(addrs, port)
    try:
        ret_val = func()
    except BaseException as e:
        sys.stderr.write("User function raise error: {error}".format(error=str(e)))
        raise e

    rank, size = get_env_rank_and_size()
    put_data_into_kvstore(addr, port, 'runfunc_result', str(rank), ret_val)


if __name__ == '__main__':
    _, driver_addrs, run_func_server_port_str = sys.argv
    run_func_server_port = int(run_func_server_port_str)
    main(driver_addrs.split(','), run_func_server_port)
