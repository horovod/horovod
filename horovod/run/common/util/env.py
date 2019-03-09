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

import re

LOG_LEVEL_STR = ['FETAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE']

# List of regular expressions to ignore environment variables by.
IGNORE_REGEXES = {re.compile(r'BASH_FUNC_.*\(\)'), re.compile('OLDPWD'),
                  # ignore strings with any none word chars (e.g. '<')
                  re.compile(r'[^\w/.:]'),
                  # TODO: Maybe we can write these to a file
                  #  during ma client installing?
                  # # Do not overwrite instance id on each instance
                  # re.compile('MICHELANGELO_INSTANCE_ID'),
                  # re.compile('PELOTON_INSTANCE_ID'),
                  # re.compile('PELOTON_TASK_ID'),
                  }


def load_ignore_envs(file_name):
    with open(file_name) as fp:
        while True:
            pattern = fp.readline().strip()
            if not pattern:
                return
            IGNORE_REGEXES.add(re.compile(pattern))


def is_exportable(v):
    return not any(pattern.search(v) for pattern in IGNORE_REGEXES)
