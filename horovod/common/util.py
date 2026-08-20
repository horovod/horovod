# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
# Modifications copyright Microsoft
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

import json
import os
import shutil
import sys
import sysconfig
import warnings

from contextlib import contextmanager

from horovod.common.exceptions import get_version_mismatch_message, HorovodVersionMismatchError


# The launcher-only build ships no framework extensions (the tensorflow/torch/mxnet
# integrations were removed), so there is nothing for the *_built() probes to load.
EXTENSIONS = []


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def get_extension_full_path(pkg_path, *args):
    assert len(args) >= 1
    dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
    full_path = os.path.join(dir_path, args[-1] + get_ext_suffix())
    return full_path


def check_extension(ext_name, ext_env_var, pkg_path, *args):
    full_path = get_extension_full_path(pkg_path, *args)
    if not os.path.exists(full_path):
        raise ImportError(
            'Extension {} has not been built: {} not found\n'
            'If this is not expected, reinstall Horovod with {}=1 to debug the build error.'.format(
                ext_name, full_path, ext_env_var
            )
        )


def extension_available(ext_base_name, verbose=False):
    # The launcher-only build ships no framework extensions (tensorflow/torch/mxnet
    # integrations were removed), so no extension is ever "built".
    return False


def _cache(f):
    cache = dict()

    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))

        if key in cache:
            return cache[key]
        else:
            retval = f(*args, **kwargs)
            cache[key] = retval
            return retval

    return wrapper


@_cache
def gpu_available(ext_base_name, verbose=False):
    # No framework extension to probe for GPU support in a launcher-only build.
    return False


@_cache
def mpi_built(verbose=False):
    # MPI launch wraps an external `mpirun`/`mpiexec`; report whether one is on PATH.
    return bool(shutil.which('mpirun') or shutil.which('mpiexec'))


@_cache
def gloo_built(verbose=False):
    # Gloo launch is pure Python (RendezvousServer); it is always available in this
    # launcher-only build, independent of any compiled Gloo allreduce controller.
    return True

@_cache
def nccl_built(verbose=False):
    # No compiled allreduce backends ship in a launcher-only build.
    return False


@_cache
def ddl_built(verbose=False):
    return False


@_cache
def ccl_built(verbose=False):
    return False

@contextmanager
def env(**kwargs):
    # ignore args with None values
    for k in list(kwargs.keys()):
        if kwargs[k] is None:
            del kwargs[k]

    # backup environment
    backup = {}
    for k in kwargs.keys():
        backup[k] = os.environ.get(k)

    # set new values & yield
    for k, v in kwargs.items():
        os.environ[k] = v

    try:
        yield
    finally:
        # restore environment
        for k in kwargs.keys():
            if backup[k] is not None:
                os.environ[k] = backup[k]
            else:
                del os.environ[k]


def get_average_backwards_compatibility_fun(reduce_ops):
    """
    Handle backwards compatibility between the old average and the new op parameters.
    Old code using the average parameter (e.g. hvd.allreduce(tensor, average=False))
    gets unchanged behavior, but mixing old and new is disallowed (e.g. no
    hvd.allreduce(tensor, average=False, op=hvd.Adasum)).
    """
    def impl(op, average):
        if op is not None:
            if average is not None:
                raise ValueError('The op parameter supersedes average. Please provide only one of them.')
            return op
        elif average is not None:
            warnings.warn('Parameter `average` has been replaced with `op` and will be removed in v1.0',
                          DeprecationWarning)
            return reduce_ops.Average if average else reduce_ops.Sum
        else:
            return reduce_ops.Average
    return impl


def num_rank_is_power_2(num_rank):
    """
    Tests if the given number of ranks is of power of 2. This check is required
    for Adasum allreduce.
    TODO support non-power of 2 ranks.
    """
    return num_rank != 0 and ((num_rank & (num_rank -1)) == 0)

def split_list(l, n):
    """
    Splits list l into n approximately even sized chunks.
    """
    d, r = divmod(len(l), n)
    return [l[i * d + min(i, r):(i + 1) * d + min(i + 1, r)] for i in range(n)]


def check_installed_version(name, version, exception=None):
    file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),\
        os.pardir, "metadata.json"))
    # metadata.json was produced by the (removed) C++/CMake build; it no longer exists
    # in a launcher-only build, so treat a missing or unreadable file as "no info".
    if not os.path.exists(file_path):
        return
    try:
        with open(file_path) as f:
            installed_version = json.load(f).get(name)
    except (OSError, ValueError):
        return
    if installed_version != version:
        if exception is None:
            warnings.warn(get_version_mismatch_message(name, version, installed_version))
        else:
            raise HorovodVersionMismatchError(name, version, installed_version) from exception

def is_iterable(x):
    try:
        _ = iter(x)
    except TypeError:
        return False
    return True


@_cache
def is_version_greater_equal_than(ver, target):
    from packaging import version
    if any([not isinstance(_str, str) for _str in (ver, target)]):
        raise ValueError("This function only accepts string arguments. \n"
                         "Received:\n"
                         "\t- ver (type {type_ver}: {val_ver})"
                         "\t- target (type {type_target}: {val_target})".format(
                            type_ver=(type(ver)),
                            val_ver=ver,
                            type_target=(type(target)),
                            val_target=target,
                         ))

    if len(target.split(".")) != 3:
        raise ValueError("We only accepts target version values in the form "
                         "of: major.minor.patch. Received: {}".format(target))

    return version.parse(ver) >= version.parse(target)
