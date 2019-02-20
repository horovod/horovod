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

import hashlib
import hmac
import os


SECRET_LENGTH = 32  # bytes
DIGEST_LENGTH = 32  # bytes
HOROVOD_SECRET_KEY = '_HOROVOD_SECRET_KEY'


def make_secret_key():
    return os.urandom(SECRET_LENGTH)


def compute_digest(key, message):
    return hmac.new(key, message, hashlib.sha256).digest()


def check_digest(key, message, digest):
    computed_digest = compute_digest(key, message)
    return hmac.compare_digest(computed_digest, digest)
