from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import horovod.tensorflow.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

lr = 1.0
batch_size = 128
epochs = 24
num_classes = 10

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % hvd.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(batch_size)

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


state = hvd.elastic.KerasState(model, batch=100, epoch=0)
state.register_reset_callbacks([on_state_reset])

callbacks = [
    # Horovod: elastic training callbacks to update and commit state.
    hvd.elastic.CommitStateCallback(state),
    hvd.elastic.UpdateBatchStateCallback(state),
    hvd.elastic.UpdateEpochStateCallback(state),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))


@hvd.elastic.run
def train(state):
    # Horovod: adjust number of steps based on number of GPUs.
    state.model.fit(dataset,
                    steps_per_epoch=500 // hvd.size(),
                    callbacks=callbacks,
                    epochs=epochs - state.epoch,
                    verbose=1 if hvd.rank() == 0 else 0)


train(state)
