.. inclusion-marker-start-do-not-remove

Elastic Horovod
===============


Elastic training enables Horovod to scale up and down the number of workers dynamically at runtime, without
requiring a restart or resuming from checkpoints saved to durable storage. With elastic training, workers can come
and go from the Horovod job without interrupting the training process.


When to use elastic training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- You are running an `autoscaling <https://en.wikipedia.org/wiki/Autoscaling>`__ job that may acquire more resources for training over time.
- Your job is running on preemptable or spot instances that may come and go with little warning.
- Your nodes are unreliable and you want your job to continue training if some of the hosts fail.


Requirements
~~~~~~~~~~~~

- TensorFlow >= 1.15 or PyTorch >= 1.0
- Horovod >= 0.20.0 with Gloo support (install Horovod using ``HOROVOD_WITH_GLOO=1`` to ensure it is installed)
- A way to discover available hosts at runtime


Modifying the training script with State Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The biggest difference when moving from normal distributed training to elastic training is the need to track and synchronize
state among the workers as workers are added or removed from the job.

To enable elastic training, make the following changes to your training script:

1. Wrap your main training process (everything following initialization) in a function decorated with ``hvd.elastic.run``.

   The first argument to this decorated function should be an instance of ``hvd.elastic.State``.  Before executing the
   decorated function, this state object will be synchronized across workers.  This ensures that workers that were
   newly added, as well as workers that might have inconsistent state, all share the same state before training begins.

   Because the sync function uses collective ops, and upon worker add the active workers will not reset from before this
   function, *no Horovod collective ops (broadcast, allreduce, allgather, etc.) can be called before this function*.

2. Place all variables that need to be kept in sync between worker replicas (model parameters, optimizer state, epoch and batch numbers, etc.) into a ``hvd.elastic.State`` object.

   Standard state implementations are provided for TensorFlow, Keras, and PyTorch.  However, it may be necessary in some cases to override
   the base ``hvd.elastic.State`` object to handle broadcasting custom types.

3. Periodically call ``state.commit()`` to backup a copy of your state in memory.

   This is useful to prevent corrupted state in the event that a worker fails unexpectedly. For example, if training fails
   in the middle of a parameter update, some gradient updates may have applied while others were still being allreduced.  When this
   happens, a ``HorovodInternalError`` will be raised, and all parameters will be restored to the values at the time of the last commit.

   Because commits can be expensive (as the model size increases), there is a tradeoff between the per-batch processing time
   and how far the training process needs to rollback in the event of a failure.  For example, if you commit once every 10
   batches, you reduce the amount of copying by a factor of 10. But if a failure occurs, you may need to redo up to 10
   previously processed batches.

   Elastic Horovod can avoid these rollbacks by performing what we call a *graceful removal* of a worker. If the driver
   process discovers that a host has been made available or flagged for removal, it will push a notification to the workers.
   The next time ``state.commit()`` or the more lightweight ``state.check_host_updates()`` is called, a ``HostsUpdatedInterrupt``
   will be raised.  This event is handled similar to the ``HorovodInternalError``, except that parameter state will not be
   restored to the last commit.

   In general, if your hardware is generally reliable, and your orchestration system gives the driver ample warning
   when a host is scheduled to be removed from the job, then you can safely call ``state.commit()`` on a reduced frequency,
   and call ``state.check_host_updates()`` at the end of each batch instead.

4. Register callbacks with the ``hvd.elastic.State`` object to respond to changes in the worker membership in the job.

   For example, rescaling the learning rate with the new world size or repartitioning the dataset would commonly be done
   through these callbacks.

   Callbacks are called after Horovod has reinitialized, but before state is synchronized across the workers.

The reset process following a ``HorovodInternalError`` (failure) or ``HostsUpdatedInterrupt`` (add/remove request) is as follows:

1. Catch exception within the ``hvd.elastic.run`` decorator.
2. Restore last committed state if ``HorovodInternalError`` was raised.
3. Reinitialize Horovod context performing a new round of rendezvous.
4. Synchronize state among the workers by broadcasting from the new worker-0.
5. Resume training by executing the underlying training function.

During rendezvous, older workers will take priority in being assigned worker-0 status to ensure that the state that
is broadcast is up to date.


Elastic TensorFlow
~~~~~~~~~~~~~~~~~~

TensorFlow v1 Example:

.. code-block:: python
    :emphasize-lines: 17,18,23,29,32,33

    import tensorflow as tf
    import horovod.tensorflow as hvd

    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    dataset = ...
    model = ...

    lr = tf.Variable(base_lr * hvd.size())
    optimizer = tf.train.GradientDescentOptimizer(lr)
    optimizer = hvd.DistributedOptimizer(optimizer)

    @hvd.elastic.run
    def train(state, train_one_batch):
        for state.epoch in range(state.epoch, epochs):
            for state.batch in range(state.batch, batches_per_epoch):
                train_one_batch()
                if state.batch % batches_per_commit == 0:
                    state.commit()
            state.batch = 0

    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())

        def on_state_reset():
            lr.load(base_lr * hvd.size(), session)

        state = hvd.elastic.TensorFlowState(session=session, batch=0, epoch=0)
        state.register_reset_callbacks([on_state_reset])

        train_opt = optimizer.minimize(loss)
        train(state, lambda: session.run(train_opt))

TensorFlow v2 Example:

.. code-block:: python
    :emphasize-lines: 33,34,40,43,46,47

    import tensorflow as tf
    import horovod.tensorflow as hvd

    hvd.init()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    dataset = ...
    model = ...

    optimizer = tf.optimizers.Adam(lr * hvd.size())

    @tf.function
    def train_one_batch(data, target, allreduce=True):
        with tf.GradientTape() as tape:
            probs = model(data, training=True)
            loss = tf.losses.categorical_crossentropy(target, probs)

        if allreduce:
            tape = hvd.DistributedGradientTape(tape)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Initialize model and optimizer state so we can synchronize across workers
    data, target = get_random_batch()
    train_one_batch(data, target, allreduce=False)

    @hvd.elastic.run
    def train(state):
        for state.epoch in range(state.epoch, epochs):
            for state.batch in range(state.batch, batches_per_epoch):
                data, target = get_random_batch()
                train_one_batch(data, target)
                if state.batch % batches_per_commit == 0:
                    state.commit()
            state.batch = 0

    def on_state_reset():
        optimizer.lr.assign(lr * hvd.size())

    state = hvd.elastic.TensorFlowKerasState(model, optimizer, batch=0, epoch=0)
    state.register_reset_callbacks([on_state_reset])
    train(state)


Elastic Keras
~~~~~~~~~~~~~

.. code-block:: python
    :emphasize-lines: 21,24,25,28,29,30,36,37

    import tensorflow as tf
    import horovod.tensorflow.keras as hvd

    hvd.init()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.keras.backend.set_session(tf.Session(config=config))

    dataset = ...
    model = ...

    opt = keras.optimizers.Adadelta(lr * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    def on_state_reset():
        tf.keras.backend.set_value(model.optimizer.lr, lr * hvd.size())

    state = hvd.elastic.KerasState(model, batch=100, epoch=0)
    state.register_reset_callbacks([on_state_reset])

    callbacks = [
        hvd.elastic.CommitStateCallback(state),
        hvd.elastic.UpdateBatchStateCallback(state),
        hvd.elastic.UpdateEpochStateCallback(state),
    ]

    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

    @hvd.elastic.run
    def train(state):
        model.fit(dataset,
                  steps_per_epoch=500 // hvd.size(),
                  callbacks=callbacks,
                  epochs=epochs - state.epoch,
                  verbose=1 if hvd.rank() == 0 else 0)

    train(state)


Elastic PyTorch
~~~~~~~~~~~~~~~

.. code-block:: python
    :emphasize-lines: 14,15,28,31,36,37

    import torch
    import horovod.torch as hvd

    hvd.init()

    torch.cuda.set_device(hvd.local_rank())

    dataset = ...
    model = ...

    optimizer = optim.SGD(model.parameters(), lr * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

    @hvd.elastic.run
    def train(state):
        batch_offset = state.batch
        for state.epoch in range(state.epoch, epochs):
            for state.batch in range(state.batch, batches_per_epoch):
                data, target = get_random_batch()

                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                if state.batch % batches_per_commit == 0:
                    state.commit()
            state.batch = 0

    def on_state_reset():
        # adjust learning rate on reset
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * hvd.size()

    state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0)
    state.register_reset_callbacks([on_state_reset])
    train(state)


Running with horovodrun
~~~~~~~~~~~~~~~~~~~~~~~

Elastic training jobs are started using the ``horovodrun`` command line tool. The major difference when launching
elastic jobs is that hosts are not specified explicitly, but instead **discovered** at runtime.  The most general way
to allow Horovod to discover available hosts is to provide a ``--host-discovery-script`` when launching the job:

.. code-block:: bash

    $ horovodrun -np 8 --host-discovery-script discover_hosts.sh python train.py

The host discovery script must have user executable permissions, and return one host with its available slots per line
of the form: ``<hostname>:<slots>``.  For example:

.. code-block:: bash

    $ ./discover_hosts.sh
    host-1:4
    host-2:4
    host-3:4

If the host discovery scripts fails to execute (due to a permissions issue) or otherwise returns a non-zero exit code
the first time it is called, the training process will fail immediately. However, subsequent errors will result in
retries until the job times-out (due to failure to discover a sufficient number of slots).

Your discovery script may omit the ``:<slots>`` if you explicitly specify the number of slots per host as an argument:

.. code-block:: bash

    $ horovodrun -np 8 --host-discovery-script discover_hosts.sh --slots 4 python train.py

The elastic training job will not start until at least ``-np`` slots are available for running worker processes.

You can additionally specify the minimum and maximum number of processes to run with during the job:

.. code-block:: bash

    $ horovodrun -np 8 --min-np 4 --max-np 12 --host-discovery-script discover_hosts.sh python train.py

If the number of available slots falls below ``--min-np`` (due to host failure, preemption, etc.), then the job will
pause waiting for more hosts to become available or until ``HOROVOD_ELASTIC_TIMEOUT`` (default: 600 seconds) has
elapsed.  If unspecified, minimum np defaults to ``-np``.

The maximum np can be used to cap the number of processes (to prevent over-utilizing available resources) and to serve
as a reference point for learning rate scales and data partitions (in cases where these need to be held constant
regardless of the current number of workers).  If unspecified, maximum np also defaults to ``-np``.

Instances that fail will be added to a blacklist, as they may have faulty hardware.  Ranks that fail repeatedly
will result in job failure, as it may be the case that the training process cannot make progress.


Running on Ray
~~~~~~~~~~~~~~

Running an elastic training script with Ray is simple and provides additional benefits to existing Horovod Elastic functionality:

* You can execute training from interactive Python environments (i.e., a Jupyter notebook)
* You can automatically leverage Ray's autoscaler to add/remove spot instances on AWS/GCP/Azure/Kubernetes.


To use elastic training with Ray:

.. code-block:: python

    import horovod.torch as hvd

    # Put the Horovod concepts into a single function
    # This function will be serialized with Cloudpickle
    def training_fn():
        hvd.init()
        model = Model()
        torch.cuda.set_device(hvd.local_rank())

        @hvd.elastic.run
        def train(state):
            for state.epoch in range(state.epoch, epochs):
                ...
                state.commit()


        state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0)
        state.register_reset_callbacks([on_state_reset])
        train(state)
        return


    from horovod.ray import ElasticRayExecutor
    import ray

    ray.init()  # or ray.init(address="auto") if on a Ray cluster

    settings = ElasticRayExecutor.create_settings(verbose=True)
    executor = ElasticRayExecutor(settings, use_gpu=True, cpus_per_slot=2)
    executor.start()
    executor.run(training_fn)


Running on Spark
~~~~~~~~~~~~~~~~

Current constraints:

- `max_np` and `min_np` are `None` or equal to `num_np`, i.e. no auto-scaling, only fault tolerant


Practical Considerations: Consistent training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With workers frequently being added and removed from the training process, it creates the possibility for learning
rates, numbers of partitions, and other parameters that vary with the number of workers to hurt model convergence if
not properly handled.

Learning rate will need to be rescaled via callback when using gradient averaging.  Using Adasum, no adjustment will
need to be made assuming that local size remains the same.

If using random sampling to read data, then no repartitioning need occur. For the time being, this is the recommended
strategy to simplify elastic training configuration.

If using dataset partitioning, callbacks may be used to repartition dataset as necessary, skipping already processed
data. Care needs to be taken when partitioning the data to ensure that data is not processed more than once. As such,
the preferred approach is to keep the number of partitions constant (from ``hvd.max_size()``), but redistribute
partitions and use local gradient aggregation to keep total batch size constant.

.. inclusion-marker-end-do-not-remove
