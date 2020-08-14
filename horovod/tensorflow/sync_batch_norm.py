# Copyright 2020 Google Research. All Rights Reserved.
# Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import tensorflow as tf
from horovod.tensorflow.mpi_ops import _allreduce
from horovod.tensorflow.mpi_ops import size, rank
from horovod.tensorflow.mpi_ops import Sum

class SyncBatchNormalization(tf.keras.layers.BatchNormalization):
  """ Synchronous batch normalization. Stats are synchronized across all workers during training. """

  def __init__(self, fused=False, **kwargs):
    if fused in (True, None):
      raise ValueError('SyncBatchNormalization does not support fused=True.')
    if not kwargs.get('name', None):
      kwargs['name'] = 'sync_batch_normalization'
    super(SyncBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""

    worker_mean, worker_variance = super(SyncBatchNormalization, self)._moments(
      inputs, reduction_axes, keep_dims=keep_dims)

    if size() > 1:
      # Compute variance using: Var[X] = E[X^2] - E[X]^2.
      worker_square_of_mean = tf.math.square(worker_mean)
      worker_mean_of_square = worker_variance + worker_square_of_mean

      # Average stats across all workers
      worker_stack = tf.stack([worker_mean, worker_mean_of_square])
      group_stack = _allreduce(worker_stack, op=Sum)
      group_stack /= size()
      group_mean, group_mean_of_square = tf.unstack(group_stack)

      group_variance = group_mean_of_square - tf.math.square(group_mean)

      return (group_mean, group_variance)
    else:
      return (worker_mean, worker_variance)

  def call(self, *args, **kwargs):
    outputs = super(SyncBatchNormalization, self).call(*args, **kwargs)
    try:
      # A temporary hack for TF1 compatibility with Keras batch norm.
      # Ops are added to tf.GraphKeys.UPDATE_OPS manually to mimic
      # behavior of TF1 batch norm layer for use in control dependencies.
      for u in self.updates:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u)
    except AttributeError:
      pass
    return outputs
