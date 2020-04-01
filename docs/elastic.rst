.. inclusion-marker-start-do-not-remove

Elastic Horovod
===============


Elastic training enables Horovod to scale up and down the number of workers dynamically at runtime, without
requiring a restart or resuming from checkpoints saved to durable storage. With elastic training, workers can come
and go from the Horovod job without interrupting the training process.


When to use elastic training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- You are running an `autoscaling <https://en.wikipedia.org/wiki/Autoscaling>`__ job that may acquire more resources for training over tine.
- Your job is running on preemptable or spot instances that may come and go with little warning.
- Your nodes are unreliable and you want your job to continue training if some of the hosts fail.


Requirements
~~~~~~~~~~~~

- Gloo (install Horovod using ``HOROVOD_WITH_GLOO=1`` to ensure it is installed)
- A way to discover the available hosts


Modifying the training script with State Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The biggest difference when moving from normal distributed training to elastic training is the need to synchronize
state among the workers when workers are added or removed from the job.

All state that is to be synchronized among the workers should be encapsulated in an object deriving from
``hvd.elastic.State``. This object encapsulates everything that needs to be in sync between workers, including
model / optimizer state, current epoch, current batch index, etc.  It will be used for issuing callbacks to adjust
learning rate, redistribute data, etc.

In order for state to be recovered in the event of a failure, you must call ``state.commit()`` periodically.
Typically, commits are made once per batch, but can be less frequent if saving state is costly and chance of node
failure is low.

On commit, if any workers are pending being added, then they will be added after the commit. A rollback will occur on
active workers, and they will resync from the sync point.

A decorator ``hvd.elastic.run`` is used to drive the elastic training loop.  Decorating a function in this way
indicates that the function execution needs to be synchronized across workers.  When failure occurs or new worker is
added, other workers will rollback to this point to resync state before resuming training.

Because the sync function uses collective ops, and upon worker add the active workers will not reset from before this
function, *no Horovod collective ops (broadcast, allreduce, etc.) can be called before this function*.

If a worker fails during training, the other workers will catch a ``HorovodInternalError`` that is raised when
attempting to perform the next collective operation.  This will trigger a rollback to the beginning of the
synchronized function.

Rollback process is as follows:

1. Catch exception within the hvd.elastic.run decorator.
2. Restore last committed state (to clear any half-updated parameters).
3. Reinitialize Horovod context performing a new round of rendezvous.
4. Synchronize state among the workers by broadcasting from the new worker-0.
5. Resume training by executing the underlying training function.

When a new instance not on an internal blacklist is discovered, a message will be posted to a server running on the
other workers to indicate that new workers are awaiting being added.  On the next commit event, a rollback will occur
on the active workers, and the new workers will be launched by the driver.

Rollback process is similar to worker failure, but the last committed state will not be restored (as there is no
partial state that could have been lost).  However, broadcast is still necessary for new workers to obtain the most
updated state.

During rendezvous, older workers will take priority in being assigned worker-0 status to ensure that the state that
is broadcast is up to date.


Running with horovodrun
~~~~~~~~~~~~~~~~~~~~~~~

Elastic training jobs are started using the ``horovodrun`` command line tool. The major difference when launching
elastic jobs is that hosts are not specified explicitly, but instead **discovered** at runtime.  The most general way
to allow Horovod to discover available hosts is to provide a ``--host-discovery-script`` when launching the job:

.. code-block:: bash

    $ horovodrun -np 8 --host-disocvery-script discover_hosts.sh python train.py

The host discovery script must have user executable permissions, and return one host with its available slots per line
of the form: ``<hostname>:<slots>``.  For example:

.. code-block:: bash

    $ ./discover_hosts.sh

    host-1:4
    host-2:4
    host-3:4

Your disocvery script my omit the ``:<slots>`` if you explicitly specify the number of slots per host as an argument:

.. code-block:: bash

    $ horovodrun -np 8 --host-disocvery-script discover_hosts.sh --slots 4 python train.py

The elastic training job will not start until at least ``-np`` slots are available for running worker processes.

You can additionally specify the minimum and maximum number of processes to run with during the job:

.. code-block:: bash

    $ horovodrun -np 8 --min-np 4 --max-np 12 --host-disocvery-script discover_hosts.sh python train.py

If the number of available slots falls below ``--min-np`` (due to host failure, preemption, etc.), then the job will
pause waiting for more hosts to become available or until ``HOROVOD_ELASTIC_START_TIMEOUT`` (default: 600 seconds) has
elapsed.  If unspecified, minimum np defaults to ``-np``.

The maximum np can be used to cap the number of processes (to prevent over-utilizing available resources) and to serve
as a reference point for learning rate scales and data partitions (in cases where these need to be held constant
regardless of the current number of workers).  If unspecified, maximum np also defaults to ``-np``.

Instances that fail will be added to a blacklist, as they may have faulty hardware.  Ranks that fail repeatedly
will result in job failure, as it may be the case that the training process cannot make progress.


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
