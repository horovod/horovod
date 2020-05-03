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


class Empty(object):
    pass


class HasOutputCols(object):
    pass


class Params(object):
    @staticmethod
    def _dummy():
        return MagicMock()


MOCK_MODULES = [
    'cloudpickle',
    'ctypes',
    'h5py',
    'psutil',

    'pyarrow',
    'pyarrow.parquet',

    'numpy',
    'numpy.core.multiarray',
    'numpy.dtype',

    'pyspark',
    'pyspark.ml',
    'pyspark.ml.linalg',
    'pyspark.ml.param',
    'pyspark.ml.param.shared',
    'pyspark.ml.util',
    'pyspark.sql',
    'pyspark.sql.functions',
    'pyspark.sql.types',

    'tensorflow',
    'tensorflow.python',
    'tensorflow.python.framework',
    'tensorflow.python.platform',
    'tensorflow.python.eager',
    'tensorflow.python.keras',

    'keras',
    'keras.backend',

    'torch',
    'torch.autograd.function',
    'torch.nn.functional',
    'torch.nn.modules.batchnorm',
    'torch.utils',
    'torch.utils.data',
    'torch.utils.tensorboard',

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
        'nn': {
            'modules': {
                'batchnorm': {
                    '_BatchNorm': MagicMock,
                }
            },
        },
    },
    'pyspark': {
        'ml': {
            'Estimator': Empty,
            'Model': Empty,
            'param': {
                'shared': {
                    'HasOutputCols': HasOutputCols,
                    'Param': MagicMock,
                    'Params': Params,
                    'TypeConverters': MagicMock(),
                },
            },
            'util': {
                'MLReadable': Empty,
                'MLWritable': Empty,
            }
        },
    },
    'horovod': {
        'common': {
            'util': {
                'get_ext_suffix': lambda: 'xyz',
            },
        },
        'spark': {
            'keras': {
                'estimator': {
                    'KerasEstimatorParamsReadable': MagicMock,
                    'KerasEstimatorParamsWritable': MagicMock,
                },
            },
            'torch': {
                'estimator': {
                    'TorchEstimatorParamsReadable': MagicMock,
                    'TorchEstimatorParamsWritable': MagicMock,
                },
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
