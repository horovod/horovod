Horovod with TensorFlow
=======================
To use Horovod with TensorFlow, make the following modifications to your training script:

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

   The distributed optimizer delegates gradient computation to the original optimizer, averages gradients using **allreduce** or **allgather**, and then applies those averaged gradients.

   For **TensorFlow v2**, when using a ``tf.GradientTape``, wrap the tape in ``hvd.DistributedGradientTape`` instead of wrapping the optimizer.

.. raw:: html

    <p/>


5. Broadcast the initial variable states from rank 0 to all other processes.

   This is necessary to ensure consistent initialization of all workers when training is started with random weights or restored from a checkpoint.

   For **TensorFlow v1**, add ``hvd.BroadcastGlobalVariablesHook(0)`` when using a ``MonitoredTrainingSession``.
   When not using ``MonitoredTrainingSession``, execute the ``hvd.broadcast_global_variables`` op after global variables have been initialized.

   For **TensorFlow v2**, use ``hvd.broadcast_variables`` after models and optimizers have been initialized.

.. raw:: html

    <p/>


6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.

   For **TensorFlow v1**, accomplish this by passing ``checkpoint_dir=None`` to ``tf.train.MonitoredTrainingSession`` if ``hvd.rank() != 0``.

   For **TensorFlow v2**, construct a ``tf.train.Checkpoint`` and only call ``checkpoint.save()`` when ``hvd.rank() == 0``.

.. raw:: html

    <p/>


TensorFlow v1 Example (see the `examples <https://github.com/horovod/horovod/blob/master/examples/>`_ directory for full training examples):

.. code-block:: python

    import tensorflow as tf
    import horovod.tensorflow as hvd


    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Build model...
    loss = ...
    opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

    # Add Horovod Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)

    # Add hook to broadcast variables from rank 0 to all other processes during
    # initialization.
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    # Make training operation
    train_op = opt.minimize(loss)

    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           config=config,
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Perform synchronous training.
        mon_sess.run(train_op)

TensorFlow v2 Example (from the `MNIST <https://github.com/horovod/horovod/blob/master/examples/tensorflow2_mnist.py>`_ example):

.. code-block:: python

    import tensorflow as tf
    import horovod.tensorflow as hvd

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
    loss = tf.losses.SparseCategoricalCrossentropy()
    opt = tf.optimizers.Adam(0.001 * hvd.size())

    checkpoint_dir = './checkpoints'
    checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)

    @tf.function
    def training_step(images, labels, first_batch):
        with tf.GradientTape() as tape:
            probs = mnist_model(images, training=True)
            loss_value = loss(labels, probs)

        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        opt.apply_gradients(zip(grads, mnist_model.trainable_variables))

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        #
        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if first_batch:
            hvd.broadcast_variables(mnist_model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)

        return loss_value

    # Horovod: adjust number of steps based on number of GPUs.
    for batch, (images, labels) in enumerate(dataset.take(10000 // hvd.size())):
        loss_value = training_step(images, labels, batch == 0)

        if batch % 10 == 0 and hvd.local_rank() == 0:
            print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    if hvd.rank() == 0:
        checkpoint.save(checkpoint_dir)
