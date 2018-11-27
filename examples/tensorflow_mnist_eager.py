# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
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
#!/usr/bin/env python

import tensorflow as tf
import horovod.tensorflow as hvd

def main(_):
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    tf.enable_eager_execution(config=config)
    tfe = tf.contrib.eager

    mnist_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])

    # Horovod: adjust learning rate based on number of GPUs.
    opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
          (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
             tf.cast(mnist_labels,tf.int64))
    )
    dataset = dataset.shuffle(1000).batch(32)

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    checkpoint_dir = './checkpoints'
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(
        model=mnist_model, optimizer=opt, step_counter=step_counter)

    # Horovod: adjust number of steps based on number of GPUs.
    for (batch, (images, labels)) in enumerate(dataset.take(20000 // hvd.size())):
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            # Horovod: broadcast initial variable states
            # from rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights
            # or restored from a checkpoint.
            hvd.bcast(0, mnist_model.variables) if batch == 0 else None

            loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss_value, mnist_model.variables)
        opt.apply_gradients(
            zip(grads, mnist_model.variables), global_step=tf.train.get_or_create_global_step())
        if batch % 10 == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    checkpoint.save(checkpoint_dir) if hvd.rank() == 0 else None

if __name__ == "__main__":
    tf.app.run()
