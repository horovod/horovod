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
import timeit

import tensorflow as tf
import horovod.tensorflow as hvd
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

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num-batches-per-commit', type=int, default=1,
                    help='number of batches per commit of the elastic state object')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

args = parser.parse_args()
args.cuda = not args.no_cuda

# Horovod: initialize Horovod.
hvd.init()

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
lr = 0.01
model = getattr(applications, args.model)(weights=None)
opt = tf.optimizers.SGD(lr * hvd.size())

data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)


@tf.function
def train_one_batch():
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: use DistributedGradientTape
    with tf.GradientTape() as tape:
        probs = model(data, training=True)
        loss = tf.losses.sparse_categorical_crossentropy(target, probs)

    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape, compression=compression)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))


def benchmark_step(state):
    train_one_batch()
    if state is not None:
        state.batch += 1
        if state.batch == args.num_batches_per_commit:
            state.batch = 0
            state.commit()


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))


# Run one batch to initialize weights before synchronization
train_one_batch()


@hvd.elastic.run
def run_benchmark(state):
    with tf.device(device):
        # Warm-up
        if not state.warm:
            log('Running warmup...')
            timeit.timeit(lambda: benchmark_step(state), number=args.num_warmup_batches)
            state.warm = True
            state.commit()

        # Benchmark
        if state.iter == 0:
            log('Running benchmark...')
        for x in range(state.iter, args.num_iters):
            time = timeit.timeit(lambda: benchmark_step(state), number=args.num_batches_per_iter)
            img_sec = args.batch_size * args.num_batches_per_iter / time
            log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
            state.img_secs.append(img_sec)
            state.iter = x
            state.commit()


def on_state_reset():
    opt.lr.assign(lr * hvd.size())


state = hvd.elastic.TensorFlowKerasState(model, opt, img_secs=[], iter=0, batch=0, warm=False)
state.register_reset_callbacks([on_state_reset])
run_benchmark(state)

# Results
img_sec_mean = np.mean(state.img_secs)
img_sec_conf = 1.96 * np.std(state.img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
