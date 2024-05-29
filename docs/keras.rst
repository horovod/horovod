Horovod with Keras
==================
Horovod supports Keras and regular TensorFlow in similar ways. To use Horovod with Keras, make the following modifications to your training script:

1. Run ``hvd.init()``.

.. raw:: html

    <p/>

2. Pin each GPU to a single process.

   With the typical setup of one GPU per process, set this to *local rank*. The first process on
   the server will be allocated the first GPU, the second process will be allocated the second GPU, and so forth.

   For **TensorFlow v1**:

   .. code-block:: python

       config = tf.ConfigProto()
       config.gpu_options.visible_device_list = str(hvd.local_rank())
       K.set_session(tf.Session(config=config))

   For **TensorFlow v2**:

   .. code-block:: python

       gpus = tf.config.experimental.list_physical_devices('GPU')
       for gpu in gpus:
           tf.config.experimental.set_memory_growth(gpu, True)
       if gpus:
           tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

.. raw:: html

    <p/>

3. Scale the learning rate by the number of workers.

   Effective batch size in synchronous distributed training is scaled by the number of workers.
   An increase in learning rate compensates for the increased batch size.

.. raw:: html

    <p/>

4. Wrap the optimizer in ``hvd.DistributedOptimizer``.

   The distributed optimizer delegates gradient computation to the original optimizer, averages gradients using *allreduce* or *allgather*, and then applies those averaged gradients.

   **Note:** For model parallel usecases there are local variables (layers) that their gradients need not to be synced (by allreduce or allgather). You can register those variables with the returned wrapper optimizer by calling its register_local_var() API. Alternatively, you can use the ``horovod.keras.PartialDistributedOptimizer`` API and and pass the local layers to this API in order to register their local variables.

.. raw:: html

    <p/>

5. Add ``hvd.callbacks.BroadcastGlobalVariablesCallback(0)`` to broadcast initial variable states from rank 0 to all other processes.

   This is necessary to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.

   **Note:** For model parallel use cases there are local variables (layers) that their weights need not to be broadcasted. You can pass those local variables to this callback by adding ``hvd.callbacks.BroadcastGlobalVariablesCallback(0, local_variables=[list of local variables])`` instead.

.. raw:: html

    <p/>

6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.

   Accomplish this by guarding model checkpointing code with ``hvd.rank() != 0``.

.. raw:: html

    <p/>

.. NOTE:: - Keras 2.0.9 has a `known issue <https://github.com/fchollet/keras/issues/8353>`_ that makes each worker allocate all GPUs on the server, instead of the GPU assigned by the *local rank*. If you have multiple GPUs per server, upgrade to Keras 2.1.2 or downgrade to Keras 2.0.8.

          - To use ``keras`` bundled with ``tensorflow`` you must use ``from tensorflow import keras`` instead of ``import keras`` and ``import horovod.tensorflow.keras as hvd`` instead of ``import horovod.keras as hvd`` in the import statements.

See full training `simple <https://github.com/horovod/horovod/blob/master/examples/keras/keras_mnist.py>`_ (shown below) and `advanced <https://github.com/horovod/horovod/blob/master/examples/keras/keras_mnist_advanced.py>`_ examples.


.. code-block:: python

    from __future__ import print_function
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    import math
    import tensorflow as tf
    import horovod.keras as hvd

    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    batch_size = 128
    num_classes = 10

    # Horovod: adjust number of epochs based on number of GPUs.
    epochs = int(math.ceil(12.0 / hvd.size()))

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # The data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Horovod: adjust learning rate based on number of GPUs.
    opt = keras.optimizers.Adadelta(1.0 * hvd.size())

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    model.fit(x_train, y_train,
              batch_size=batch_size,
              callbacks=callbacks,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

TensorFlow v2 Keras Example (from the `MNIST <https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_keras_mnist.py>`_ example):

.. code-block:: python

    import tensorflow as tf
    import horovod.tensorflow.keras as hvd

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # Build model and dataset
    dataset = ...
    model = ...
    opt = tf.optimizers.Adam(0.001 * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    mnist_model.compile(loss=tf.losses.SparseCategoricalCrossentropy(),
                        optimizer=opt,
                        metrics=['accuracy'],
                        experimental_run_tf_function=False)

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    model.fit(dataset,
              steps_per_epoch=500 // hvd.size(),
              callbacks=callbacks,
              epochs=24,
              verbose=1 if hvd.rank() == 0 else 0)
