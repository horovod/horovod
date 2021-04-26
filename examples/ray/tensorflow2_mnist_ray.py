import argparse
import tensorflow as tf
import horovod.tensorflow.keras as hvd

import ray
from horovod.ray import RayExecutor

parser = argparse.ArgumentParser(
    description='Tensorflow Ray example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--address', type=str, help="Ray cluster address")


def train(num_epochs):
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()],
                                                   'GPU')

    (mnist_images, mnist_labels), _ = \
        tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
         tf.cast(mnist_labels, tf.int64)))
    dataset = dataset.repeat().shuffle(10000).batch(128)

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
    opt = hvd.DistributedOptimizer(opt)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    mnist_model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=opt,
        metrics=['accuracy'],
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

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(
            initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    # Horovod: write logs on worker 0.
    verbose = 1 if hvd.rank() == 0 else 0

    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    mnist_model.fit(
        dataset,
        steps_per_epoch=500 // hvd.size(),
        callbacks=callbacks,
        epochs=num_epochs,
        verbose=verbose)


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(address=args.address)
    settings = RayExecutor.create_settings(timeout_s=30)
    executor = RayExecutor(
        settings, num_hosts=2, num_slots=4, use_gpu=False, cpus_per_slot=8)
    executor.start()
    executor.run(train, kwargs=dict(num_epochs=1))
    executor.shutdown()
