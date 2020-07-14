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

import datetime
import errno
import os
import threading
import cloudpickle


class Cache(object):
    """
    Cache the function calls
    """

    def __init__(self, cache_folder, cache_staleness_threshold_in_minutes,
                 parameters_hash):
        # Protocol version 0 is the original "human-readable" protocol and is
        # compatible with earlier python 2 and 3.
        self._pickle_protocol = 0
        self._cache_file = os.path.join(cache_folder, 'cache.bin')
        try:
            # If folder exists, does not do anything.
            os.makedirs(cache_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if not os.path.isfile(self._cache_file) or \
                self._cache_file_is_corrupt_and_deleted():
            self._dump({'parameters_hash': parameters_hash})

        content = self._load(self._cache_file)

        if content.get('parameters_hash', None) == parameters_hash:
            # If previous cache was for the same set of parameters, use it.
            self._content = content
        else:
            self._content = {'parameters_hash': parameters_hash}

        self._cache_staleness_threshold = \
            datetime.timedelta(minutes=cache_staleness_threshold_in_minutes)
        self._lock = threading.Lock()

    def get(self, key):
        self._lock.acquire()
        timestamp, val = self._content.get(key, (None, None))
        self._lock.release()

        if timestamp:
            if timestamp >= datetime.datetime.now() - self._cache_staleness_threshold:
                return val
        else:
            return None

    def put(self, key, val):
        self._lock.acquire()
        self._content[key] = (datetime.datetime.now(), val)
        try:
            self._dump(self._content)
        finally:
            self._lock.release()

    def _dump(self, content):
        with open(self._cache_file, 'wb') as cf:
            cloudpickle.dump(content, cf, protocol=self._pickle_protocol)

    def _load(self, cache_file):
        with open(cache_file, 'rb') as cf:
            try:
                content = cloudpickle.load(cf)
            except Exception as e:
                print(
                    'There is an error with reading cache file. You '
                    'can delete the corrupt file: {cache_file}.'.format(
                        cache_file=cache_file))
                raise
        return content

    def _cache_file_is_corrupt_and_deleted(self):
        try:
            _ = self._load(self._cache_file)
            return False
        except Exception as e:
            os.remove(self._cache_file)
            return True


def use_cache():
    """
    If used to decorate a function and if fn_cache is set, it will store the
    output of the function if the output is not None. If a function output
    is None, the execution result will not be cached.
    :return:
    """

    def wrap(func):
        def wrap_f(*args, **kwargs):
            fn_cache = kwargs.pop('fn_cache')
            if fn_cache is None:
                results = func(*args, **kwargs)
            else:
                cached_result = fn_cache.get(
                    (func.__name__, tuple(args[0]), frozenset(kwargs.items())))
                if cached_result is not None:
                    return cached_result
                results = func(*args, **kwargs)
                if results is not None:
                    fn_cache.put(
                        (func.__name__, tuple(args[0]),
                            frozenset(kwargs.items())),
                        results)
            return results

        return wrap_f

    return wrap
