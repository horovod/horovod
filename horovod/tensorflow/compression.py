# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 Uber Technologies, Inc.
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
"""Gradient compression algorithms."""

from enum import Enum
from functools import partial

import tensorflow as tf


class NoneCompression(object):
    """Default no-op compression."""
    __instance = None

    def __init__(self):
        if NoneCompression.__instance is not None:
            raise Exception("NoneCompression is a singleton")
        else:
            NoneCompression.__instance = self

    def compress(self, tensor):
        """Returns the tensor unmodified."""
        return tensor

    def decompress(self, tensor):
        """Returns the tensor unmodified."""
        return tensor

    @staticmethod
    def instance():
        """Returns the singleton instance."""
        if NoneCompression.__instance is None:
            NoneCompression()
        return NoneCompression.__instance


class FP16Compression(object):
    """Compress all floating point gradients to 16-bit."""
    def __init__(self, dtype):
        """Compresses tensors of the given dtype, and decompresses back."""
        self._dtype = dtype

    def compress(self, tensor):
        """Downcasts the tensor to 16-bit."""
        if tensor.dtype != self._dtype:
            raise ValueError('expected tensor of type %s but given %s' %
                             (str(self._dtype), str(tensor.dtype)))
        tensor_compressed = tensor
        if self._dtype.is_floating:
            # Only allow compression from other floating point types
            tensor_compressed = tf.cast(tensor, dtype=tf.float16)
        return tensor_compressed

    def decompress(self, tensor):
        """Upcasts the tensor to the dtype of the last compressed tensor."""
        tensor_decompressed = tensor
        if self._dtype.is_floating:
            tensor_decompressed = tf.cast(tensor, dtype=self._dtype)
        return tensor_decompressed


class Compression(Enum):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = partial(lambda dtype: NoneCompression.instance())

    """Compress all floating point gradients to 16-bit."""
    fp16 = partial(lambda dtype: FP16Compression(dtype))

    def get_compressor(self, dtype):
        """Returns a new compressor instance for the given dtype."""
        return self.value(dtype)
