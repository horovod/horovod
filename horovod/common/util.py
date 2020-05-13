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

import multiprocessing
import os
import sys
import sysconfig
import warnings

from contextlib import contextmanager


_PY3 = sys.version_info[0] == 3
EXTENSIONS = ['tensorflow', 'torch', 'mxnet']


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
            'Extension %s has not been built.  If this is not expected, reinstall '
            'Horovod with %s=1 to debug the build error.' % (ext_name, ext_env_var))


def _check_extension_lambda(ext_base_name, fn, fn_desc, verbose):
    """
    Tries to load the extension in a new process.  If successful, puts fn(ext)
    to the queue or False otherwise.  Mutes all stdout/stderr.
    """
    def _target_fn(ext_base_name, fn, fn_desc, queue, verbose):
        import importlib
        import sys
        import traceback

        if verbose:
            print('Checking whether extension {ext_base_name} was {fn_desc}.'.format(
                ext_base_name=ext_base_name, fn_desc=fn_desc))
        else:
            # Suppress output
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        try:
            ext = importlib.import_module('.' + ext_base_name, 'horovod')
            result = fn(ext)
        except:
            traceback.print_exc()
            result = None

        if verbose:
            print('Extension {ext_base_name} {flag} {fn_desc}.'.format(
                ext_base_name=ext_base_name, flag=('was' if result else 'was NOT'),
                fn_desc=fn_desc))

        queue.put(result)

    # 'fork' is required because horovodrun is a frozen executable
    ctx = multiprocessing.get_context('fork') if _PY3 else multiprocessing
    queue = ctx.Queue()
    p = ctx.Process(target=_target_fn,
                    args=(ext_base_name, fn, fn_desc, queue, verbose))
    p.daemon = True
    p.start()
    p.join()
    return queue.get_nowait()


def extension_available(ext_base_name, verbose=False):
    available_fn = lambda ext: ext is not None
    return _check_extension_lambda(
        ext_base_name, available_fn, 'built', verbose) or False


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
    available_fn = lambda ext: ext._check_has_gpu()
    return _check_extension_lambda(
        ext_base_name, available_fn, 'running with GPU', verbose) or False


@_cache
def mpi_built(verbose=False):
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.mpi_built()
        result = _check_extension_lambda(
            ext_base_name, built_fn, 'built with MPI', verbose)
        if result is not None:
            return result
    return False


@_cache
def gloo_built(verbose=False):
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.gloo_built()
        result = _check_extension_lambda(
            ext_base_name, built_fn, 'built with Gloo', verbose)
        if result is not None:
            return result
    raise RuntimeError('Failed to determine if Gloo support has been built. '
                       'Run again with --verbose for more details.')


@_cache
def nccl_built(verbose=False):
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.nccl_built()
        result = _check_extension_lambda(
            ext_base_name, built_fn, 'built with NCCL', verbose)
        if result is not None:
            return result
    raise RuntimeError('Failed to determine if NCCL support has been built. '
                       'Run again with --verbose for more details.')


@_cache
def ddl_built(verbose=False):
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.ddl_built()
        result = _check_extension_lambda(
            ext_base_name, built_fn, 'built with DDL', verbose)
        if result is not None:
            return result
    raise RuntimeError('Failed to determine if DDL support has been built. '
                       'Run again with --verbose for more details.')


@_cache
def ccl_built(verbose=False):
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.ccl_built()
        result = _check_extension_lambda(
            ext_base_name, built_fn, 'built with CCL', verbose)
        if result is not None:
            return result
    raise RuntimeError('Failed to determine if CCL support has been built. '
                       'Run again with --verbose for more details.')


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
        if op != None:
            if average != None:
                raise ValueError('The op parameter supersedes average. Please provide only one of them.')
            return op
        elif average != None:
            warnings.warn('Parameter `average` has been replaced with `op` and will be removed in v0.21.0',
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
