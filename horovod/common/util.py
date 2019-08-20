# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
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

from contextlib import contextmanager
import importlib
from multiprocessing import Process, Queue
import os
import sysconfig

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


def _get_extension_module(ext_base_name):
    try:
        return importlib.import_module('.' + ext_base_name, 'horovod')
    except:
        return None


def _check_extension_lambda(ext_base_name, fn):
    """
    Tries to load the extension in a new process.  If successful, puts fn(ext)
    to the queue or False otherwise.  Mutes all stdout/stderr.
    """
    def _target_fn(ext_base_name, fn, queue):
        import importlib
        import sys

        # Suppress output
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        try:
            ext = _get_extension_module(ext_base_name)
            queue.put(fn(ext))
        except:
            queue.put(False)

    queue = Queue()
    p = Process(target=_target_fn, args=(ext_base_name, fn, queue))
    p.daemon = True
    p.start()
    p.join()
    return queue.get_nowait() or False


def extension_available(ext_base_name):
    available_fn = lambda ext: True
    return _check_extension_lambda(ext_base_name, available_fn)


def mpi_built():
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.mpi_built()
        return _check_extension_lambda(ext_base_name, built_fn)
    return False


def gloo_built():
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.gloo_built()
        return _check_extension_lambda(ext_base_name, built_fn)
    return False


def nccl_built():
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.nccl_built()
        return _check_extension_lambda(ext_base_name, built_fn)
    return False


def ddl_built():
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.ddl_built()
        return _check_extension_lambda(ext_base_name, built_fn)
    return False


def mlsl_built():
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.mlsl_built()
        return _check_extension_lambda(ext_base_name, built_fn)
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
