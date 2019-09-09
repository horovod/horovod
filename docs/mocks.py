# Copyright 2019 Uber Technologies, Inc.
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
from unittest.mock import MagicMock


MOCK_MODULES = [
    'cloudpickle',
    'ctypes',
    'psutil',
    'pyspark',

    'tensorflow',
    'tensorflow.python',
    'tensorflow.python.framework',
    'tensorflow.python.platform',
    'tensorflow.python.eager',
    'tensorflow.python.keras',

    'keras',
    'keras.backend',

    'torch',

    'mxnet',
    'mxnet.base',

    'horovod.common.util',
    'horovod.torch.mpi_lib_v2',
]


MOCK_TREE = {
    'tensorflow': {
        '__version__': '1.14.0',
        'train': {
            'Optimizer': MagicMock,
            'SessionRunHook': MagicMock,
        },
        'estimator': {
            'SessionRunHook': MagicMock,
        },
        'keras': {
            'callbacks': {
                'Callback': MagicMock,
            },
        },
    },
    'keras': {
        'callbacks': {
            'Callback': MagicMock,
        },
    },
    'torch': {
        '__version__': '1.0.0',
    },
    'horovod': {
        'common': {
            'util': {
                'get_ext_suffix': lambda: 'xyz',
            },
        },
    },
}


def gen_mock_package(path):
    if type(path) == str:
        path = path.split('.')

    class TreeMock(MagicMock):
        @classmethod
        def __getattr__(cls, name):
            full_path = path + [name]
            tree_ptr = MOCK_TREE
            for path_part in full_path:
                if path_part in tree_ptr:
                    if type(tree_ptr[path_part]) != dict:
                        return tree_ptr[path_part]
                    else:
                        tree_ptr = tree_ptr[path_part]
                else:
                    return MagicMock()
            return gen_mock_package(full_path)

    return TreeMock()


def instrument():
    sys.modules.update((mod_name, gen_mock_package(mod_name))
                       for mod_name in MOCK_MODULES)
