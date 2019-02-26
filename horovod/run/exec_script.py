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

from horovod.spark.util import codec

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Usage: %s <pickled function obj path> '
              '<pickled arg obj path>' % sys.argv[0])
        sys.exit(1)

    fn, arg = codec.loads_base64(sys.argv[1])
    results = fn(*arg)

    output_format = 'RESULT: {result} EOM'
    print('FUNCTION SUCCESSFULLY EXECUTED.')
    print(output_format.format(result=codec.dumps_base64(results)))
