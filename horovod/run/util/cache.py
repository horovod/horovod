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

import cloudpickle
import datetime
import errno
import os
import threading


class Cache(object):
    """
    Cache the function calls
    """

    def __init__(self, cache_folder, cache_staleness_threshold_in_minutes,
                 parameters_hash):

        self._cache_file = os.path.join(cache_folder, 'cache.bin')
        try:
            # If folder exists, does not do anything.
            os.makedirs(cache_folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if not os.path.isfile(self._cache_file):
            with open(self._cache_file, 'wb') as cf:
                cloudpickle.dump({'parameters_hash': parameters_hash}, cf)

        with open(self._cache_file, 'rb') as cf:
            try:
                content = cloudpickle.load(cf)
            except Exception as e:
                print(
                    'There is an error with reading cache file. You '
                    'can delete the corrupt file: {cache_file}.'.format(
                        cache_file=self._cache_file))
                raise

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
            with open(self._cache_file, 'wb') as cf:
                cloudpickle.dump(self._content, cf)
        finally:
            self._lock.release()
