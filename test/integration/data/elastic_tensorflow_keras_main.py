# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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
import json
import os
import psutil
import time

import tensorflow as tf

import horovod.tensorflow.keras as hvd

parser = argparse.ArgumentParser(description='TensorFlow Keras Elastic Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batches-per-epoch', type=int, default=10,
                    help='number of batches per epoch')
parser.add_argument('--batches-per-commit', type=int, default=1,
                    help='number of batches per commit of the elastic state object')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of epochs')
parser.add_argument('--epoch-wait', type=int, default=0,
                    help='number of seconds each epoch takes')
parser.add_argument('--logfile', default='/tmp/logfile.txt',
                    help='log file to record results (one line per epoch)')
parser.add_argument('--discovery-schedule', default='[]',
                    help='JSON string specifying schedule of host updates each epoch')
parser.add_argument('--discovery-wait', type=int, default=3,
                    help='number of seconds the worker waits for an expected host discovery')
parser.add_argument('--exit-schedule', default='{}',
                    help='JSON string mapping from (epoch, batch) to list of ranks to exit at that time')
parser.add_argument('--exit-mode', default='exception',
                    help='means used to cause a worker to exit [exception | kill]')

args = parser.parse_args()

config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config.gpu_options.allow_growth = False
config.gpu_options.visible_device_list = ''
tf.keras.backend.set_session(tf.Session(config=config))

hvd.init()

lr = 0.01
model = tf.keras.Sequential([tf.keras.layers.Dense(2, activation='softmax', input_shape=(2,))])
optimizer = tf.keras.optimizers.SGD(lr * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

batch_size = 32
data = tf.random.uniform([batch_size, 2])
target = tf.random.uniform([batch_size, 1], minval=0, maxval=2, dtype=tf.int64)
dataset = tf.data.Dataset.from_tensor_slices((data, target)).cache().repeat().batch(batch_size)

hostname = os.environ.get('HOROVOD_HOSTNAME')
start_rank = int(os.environ.get('HOROVOD_RANK', 0))

discovery_schedule = json.loads(args.discovery_schedule)
epoch_to_hosts = {epoch: hosts for epoch, hosts in discovery_schedule if epoch is not None}
default_hosts = discovery_schedule[-1][1] if discovery_schedule else []

exit_schedule = json.loads(args.exit_schedule) if args.exit_schedule else {}


def check_exit(epoch, batch):
    key = str((epoch, batch))
    if key in exit_schedule:
        ranks_to_exit = exit_schedule[key]
        if start_rank in ranks_to_exit:
            if args.exit_mode == 'exception':
                raise RuntimeError('check_rank and exit epoch={} batch={} start_rank={} rank={}'
                                   .format(epoch, batch, start_rank, hvd.rank()))
            else:
                psutil.Process(os.getpid()).kill()


def log_state(state):
    state_dict = {
        'epoch': state.epoch,
        'batch': state.batch,
        'commits': state.commits,
        'hostname': hostname,
        'start_rank': start_rank,
        'rank': hvd.rank(),
        'size': hvd.size(),
        'rendezvous': state.rendezvous}
    with open(args.logfile, 'a') as f:
        f.write(json.dumps(state_dict) + os.linesep)


def on_state_reset():
    tf.keras.backend.set_value(model.optimizer.lr, lr * hvd.size())


state = hvd.elastic.KerasState(model, batch=0, epoch=0, commits=0, rendezvous=0)
state.register_reset_callbacks([on_state_reset])


# Handles all the test logic that surrounds the training loop
class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, state):
        self.state = state

    def on_batch_begin(self, batch, logs=None):
        check_exit(self.state.epoch, self.state.batch)

    def on_epoch_begin(self, epoch, logs=None):
        print('epoch {} batch {}'.format(self.state.epoch, self.state.batch))

    def on_epoch_end(self, epoch, logs=None):
        if hvd.rank() == 0:
            log_state(self.state)

            current_hosts = epoch_to_hosts.get(self.state.epoch, default_hosts)
            next_hosts = epoch_to_hosts.get(self.state.epoch + 1, default_hosts)
            if args.discovery_wait > 0 and current_hosts != next_hosts:
                print('host changes: {} -> {}'.format(current_hosts, next_hosts))
                start = int(time.time())
                while state._host_messages.empty():
                    if int(time.time()) - start > args.discovery_wait:
                        raise TimeoutError('Timed out waiting for notifications from driver.')
                    time.sleep(0.1)

        if args.epoch_wait > 0:
            time.sleep(args.epoch_wait)


# Special callback for testing that allows us to record how many times we have committed
class TrackingCommitCallback(hvd.elastic.CommitStateCallback):
    def commit(self):
        self.state.commits += 1
        super().commit()


callbacks = [
    TestCallback(state),
    hvd.elastic.UpdateEpochStateCallback(state),
    hvd.elastic.UpdateBatchStateCallback(state),
    TrackingCommitCallback(state, args.batches_per_commit),
]


@hvd.elastic.run
def train(state):
    state.rendezvous += 1
    # Horovod: adjust number of steps based on number of GPUs.
    state.model.fit(dataset,
                    steps_per_epoch=args.batches_per_epoch,
                    callbacks=callbacks,
                    epochs=args.epochs - state.epoch,
                    verbose=1 if hvd.rank() == 0 else 0)


train(state)
