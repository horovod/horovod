# Copyright 2022 G-Research. All Rights Reserved.
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

import os

import tensorflow as tf
from filelock import FileLock

import horovod.tensorflow.keras as hvd
from horovod.tensorflow.data.compute_service import TfDataServiceConfig, tf_data_service


# arguments reuse_dataset and round_robin only used when single dispatcher is present
def train_fn(compute_config: TfDataServiceConfig, reuse_dataset: bool = False, round_robin: bool = False):
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    with tf_data_service(compute_config, hvd.rank()) as dispatcher_address:
        # this lock guarantees only one training task downloads the dataset
        with FileLock(os.path.expanduser("~/.horovod_lock")):
            (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
             tf.cast(mnist_labels, tf.int64))
        )

        # Allow tf.data service to pre-process the pipeline
        dataset = dataset.repeat() \
            .shuffle(10000) \
            .batch(128) \
            .apply(tf.data.experimental.service.distribute(
                service=dispatcher_address,
                processing_mode="distributed_epoch",
                job_name='job' if reuse_dataset else None,
                consumer_index=hvd.rank() if round_robin else None,
                num_consumers=hvd.size() if round_robin else None)) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        mnist_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
            tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Horovod: adjust learning rate based on number of GPUs.
        scaled_lr = 0.001 * hvd.size()
        opt = tf.optimizers.Adam(scaled_lr)

        # Horovod: add Horovod DistributedOptimizer.
        opt = hvd.DistributedOptimizer(
            opt, backward_passes_per_step=1, average_aggregated_gradients=True)

        # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
        # uses hvd.DistributedOptimizer() to compute gradients.
        mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                            optimizer=opt,
                            metrics=['accuracy'],
                            experimental_run_tf_function=False)

        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
        ]

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

        # Horovod: write logs on worker 0.
        verbose = 1 if hvd.rank() == 0 else 0

        # Train the model.
        # Horovod: adjust number of steps based on number of GPUs.
        mnist_model.fit(dataset, steps_per_epoch=32 // hvd.size(), callbacks=callbacks, epochs=24, verbose=verbose)
