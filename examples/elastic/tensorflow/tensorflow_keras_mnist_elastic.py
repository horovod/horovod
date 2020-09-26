import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import horovod.tensorflow.keras as hvd

parser = argparse.ArgumentParser(description='TensorFlow Keras MNIST Elastic',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batches-per-epoch', type=int, default=500,
                    help='number of batches per epoch scaled by world size')
parser.add_argument('--batches-per-commit', type=int, default=1,
                    help='number of batches per commit of the elastic state object')
parser.add_argument('--epochs', type=int, default=24,
                    help='number of epochs')
parser.add_argument('--learning-rate', type=float, default=1.0,
                    help='learning rate')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')

args = parser.parse_args()

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

lr = args.learning_rate
batch_size = args.batch_size
epochs = args.epochs
num_classes = 10

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(batch_size)

# NOTE: `input_shape` is required to ensure the model is built before training
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Horovod: adjust learning rate based on number of GPUs.
opt = keras.optimizers.Adadelta(lr * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])


def on_state_reset():
    tf.keras.backend.set_value(model.optimizer.lr, lr * hvd.size())


state = hvd.elastic.KerasState(model, batch=0, epoch=0)
state.register_reset_callbacks([on_state_reset])

# Horovod: elastic training callbacks to update and commit state.
callbacks = [
    # Handles keeping the current epoch in sync across reset events
    hvd.elastic.UpdateEpochStateCallback(state),

    # Handles keeping the current batch in sync and ensuring that
    # epochs that were partially completed resume from the last
    # committed batch
    hvd.elastic.UpdateBatchStateCallback(state),

    # Commit state at the end of every `batches_per_commit` batches
    # (default: 1) and at the end of each epoch.
    #
    # It is important that this callback comes last to ensure that all
    # other state is fully up to date before we commit, so we do not lose
    # any progress.
    hvd.elastic.CommitStateCallback(state, batches_per_commit=args.batches_per_commit),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))


@hvd.elastic.run
def train(state):
    # Horovod: adjust number of steps based on number of GPUs and number of epochs
    # based on the number of previously completed epochs.
    state.model.fit(dataset,
                    steps_per_epoch=args.batches_per_epoch // hvd.size(),
                    callbacks=callbacks,
                    epochs=epochs - state.epoch,
                    verbose=1 if hvd.rank() == 0 else 0)


train(state)
