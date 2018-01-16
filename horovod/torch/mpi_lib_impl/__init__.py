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

from torch.utils.ffi import _wrap_function
from horovod.common import get_ext_suffix as _get_ext_suffix
from ._mpi_lib_impl import ffi as _ffi
import os as _os

# Make sure to preserve this code to load library with RTLD_GLOBAL,
# otherwise it will get unloaded.
_lib = _ffi.dlopen(_os.path.join(_os.path.dirname(__file__),
                                '_mpi_lib_impl' + _get_ext_suffix()),
                   _ffi.RTLD_GLOBAL)

__all__ = []
def _import_symbols(locals):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        if callable(fn):
            locals[symbol] = _wrap_function(fn, _ffi)
        else:
            locals[symbol] = fn
        __all__.append(symbol)

_import_symbols(locals())
