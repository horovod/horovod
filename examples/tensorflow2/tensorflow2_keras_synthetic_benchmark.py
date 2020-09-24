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
import argparse
import os
import numpy as np
from timeit import default_timer as timer

import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda
device = 'GPU' if args.cuda else 'CPU'

# Horovod: initialize Horovod.
hvd.init()

if hvd.rank() == 0:
    print('Model: %s' % args.model)
    print('Batch size: %d' % args.batch_size)
    print('Number of %ss: %d' % (device, hvd.size()))

# Horovod: pin GPU to be used to process local rank (one GPU per process)
if args.cuda:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set up standard model.
model = getattr(applications, args.model)(weights=None)
opt = tf.optimizers.SGD(0.01)

# Synthetic dataset
data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((data, target)).cache().repeat().batch(args.batch_size)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt, compression=compression)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
              optimizer=opt,
              experimental_run_tf_function=False)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
]

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
        print('Total img/sec on %d %s(s): %.1f +-%.1f' %
             (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = args.batch_size * args.num_batches_per_iter / time
        print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, device))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)

# Horovod: write logs on worker 0.
if hvd.rank() == 0:
    timing = TimingCallback()
    callbacks.append(timing)

# Train the model.
model.fit(
    dataset,
    batch_size=args.batch_size,
    steps_per_epoch=args.num_batches_per_iter,
    callbacks=callbacks,
    epochs=args.num_iters,
    verbose=0,
)
