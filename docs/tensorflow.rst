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

   **Note:** For model parallel use cases there are local variables (layers) that their gradients need not to be synced (by allreduce or allgather). You can register those variables with the returned wrapper optimizer by calling its ``register_local_var()`` API or alternatively, you can use the ``horovod.keras.PartialDistributedOptimizer`` API and and pass the local layers to this API in order to register their local variables. Additionally, when using ``tf.GradientTape``, wrap the tape in ``hvd.PartialDistributedGradientTape`` instead of ``DistributedGradientTape`` and pass the local layers to it in order to register their local variables.

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

TensorFlow v2 Example (from the `MNIST <https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_mnist.py>`_ example):

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

Horovod with TensorFlow Data Service
------------------------------------

A `TensorFlow Data Service <https://www.tensorflow.org/api_docs/python/tf/data/experimental/service>`_
allows to move CPU intensive processing of your dataset from your training process to a cluster of
CPU-rich processes.

With Horovod, it is easy to spin up a TensorFlow Data Service on your Horovod cluster and to connect
your Horovod training job to it.

Run the following command to run a TensorFlow Data Service via Horovod:

.. code-block:: bash

    horovodrun -np 4 python -m horovod.tensorflow.data.compute_worker /tmp/compute.json

This starts a TensorFlow Data Service (here called compute job) with one dispatcher and four workers.

.. note:: The config file is written by the compute job and has to be located on a path that is accessible
    to all nodes that run the compute job, e.g. a distributed file system.

Your training job can then move CPU intensive dataset operations to this data service by
calling ``.send_to_data_service(…)`` on the TensorFlow dataset:

.. code-block:: python

    from horovod.tensorflow.data.compute_service import TfDataServiceConfig

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    compute_config = TfDataServiceConfig.read('/tmp/compute.json', wait_for_file_creation=True)

    dataset = dataset.repeat() \
        .shuffle(10000) \
        .batch(128) \
        .send_to_data_service(compute_config, rank, size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

All transformations before calling ``send_to_data_service`` will be executed by the data service,
while all transformations after it are executed locally by the training script.

You can find the `tensorflow2_mnist_data_service.py <https://github.com/horovod/horovod/blob/master/examples/tensorflow2/tensorflow2_mnist_data_service.py>`_
example in the examples directory.

First start the data service as shown above. While the data service is running, start the example training script:

.. code-block:: bash

    horovodrun -np 2 python tensorflow2_mnist_data_service.py /tmp/compute.json

The compute job normally runs on CPU nodes while the training job runs on GPU nodes. This allows to run CPU intensive
dataset transformation on CPU nodes while running GPU intensive training on GPU nodes. There can be multiple CPUs
dedicated to one GPU task.

Use the ``--hosts`` argument to run compute and train job on CPU (here ``cpu-node-1`` and ``cpu-node-2``)
and GPU nodes (here ``gpu-node-1`` and ``gpu-node-2``), respectively:

.. code-block:: bash

    horovodrun -np 4 --hosts cpu-node-1:2,cpu-node-2:2 python -m horovod.tensorflow.data.compute_worker /tmp/compute.json
    horovodrun -np 2 --hosts gpu-node-1:1,gpu-node-2:1 python tensorflow2_mnist_data_service.py /tmp/compute.json

.. note::

    Please make sure you understand how TensorFlow Data Service distributes dataset transformations:
    See the `distribute <https://www.tensorflow.org/api_docs/python/tf/data/experimental/service/distribute>`_ transformation.

Multiple Dispatchers
~~~~~~~~~~~~~~~~~~~~

The data service allows for multiple dispatchers, one per training task. Each dispatcher gets the same number of workers.
As workers are dedicated to a single dispatcher, workers get dedicated to a single training task.
The size of your compute job (``-np 4``) has to be a multiple of the number of dispatchers (``--dispatchers 2``):

.. code-block:: bash

    horovodrun -np 4 python -m horovod.tensorflow.data.compute_worker --dispatchers 2 /tmp/compute.json

This requires the number of dispatchers (``--dispatchers 2``) to match the size of your training job (``-np 2``):

.. code-block:: bash

    horovodrun -np 2 python tensorflow2_mnist_data_service.py /tmp/compute.json

Single Dispatchers
~~~~~~~~~~~~~~~~~~

With a single dispatcher, TensorFlow allows to reuse the dataset across all training tasks. This is done on a
first-come-first-serve basis, or round robin. The only supported processing mode is ``"distributed_epoch"``.

Training-side dispatchers
~~~~~~~~~~~~~~~~~~~~~~~~~

The dispatchers by default run inside the compute job. You can, however, also run them inside the training job.
Add ``--dispatcher-side training`` to tell the compute job that dispatchers are started by the training job.

.. code-block:: bash

    horovodrun -np 4 python -m horovod.tensorflow.data.compute_worker --dispatcher-side training /tmp/compute.json

The training script then starts the dispatchers via ``with tf_data_service(…)`` and distributes the dataset itself:

.. code-block:: python

    hvd.init()
    rank = hvd.rank()
    size = hvd.size()

    compute_config = TfDataServiceConfig.read('/tmp/compute.json', wait_for_file_creation=True)

    with tf_data_service(compute_config, rank) as dispatcher_address:

        dataset = dataset.repeat() \
            .shuffle(10000) \
            .batch(128) \
            .apply(tf.data.experimental.service.distribute(
                processing_mode="distributed_epoch",
                service=dispatcher_address,
                job_name='job' if reuse_dataset else None,
                consumer_index=rank if round_robin else None,
                num_consumers=size if round_robin else None)) \
            .prefetch(tf.data.experimental.AUTOTUNE)

To see the specific changes needed to make the training job run dispatchers,
simply diff the training-side example with the compute-side example:

.. code-block:: bash

    diff -w examples/tensorflow2/tensorflow2_mnist_data_service_train_fn_*

Compute job on Spark cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The compute job can be started on a Spark cluster using ``spark-submit``:

.. code-block:: bash

    worker_py=$(python -c "import horovod.spark.tensorflow.compute_worker as worker; print(worker.__file__)")
    spark-submit --master "local[4]" "$worker_py" /tmp/compute.json


While the compute job is running, start the training job:

    cd examples/spark/tensorflow2
    spark-submit --master "local[2]" --py-files tensorflow2_mnist_data_service_train_fn_compute_side_dispatcher.py,tensorflow2_mnist_data_service_train_fn_training_side_dispatcher.py tensorflow2_mnist_data_service.py /tmp/compute.json

As usual, the config file has to be located on a path that is accessible to all nodes that run the compute job.
