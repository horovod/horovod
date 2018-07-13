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

# If PyTorch is installed, it must be imported before TensorFlow, otherwise
# we may get an error: dlopen: cannot load any more object with static TLS
try:
    import torch
except:
    pass

try:
    import tensorflow
except:
    pass

# Keras 2.0.0 has a race condition during first initialization that attempts
# to make a directory.  If multiple processes attempt to make the directory
# at the same time, all but the first one will fail.  This has been fixed
# in new versions of Keras.
try:
    import keras
except:
    pass
