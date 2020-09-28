.. inclusion-marker-start-do-not-remove

Horovod on Ray
==============

``horovod.ray`` allows users to leverage Horovod on `a Ray cluster <https://docs.ray.io/en/latest/cluster/index.html>`_.

Currently, the Ray + Horovod integration provides a :ref:`RayExecutor API <horovod_ray_api>`.

.. note:: The Ray + Horovod integration currently only supports a Gloo backend.

Installation
------------

Use the extra ``[ray]`` option to install Ray along with Horovod.

.. code-block:: bash

    $ HOROVOD_WITH_GLOO=1 ... pip install 'horovod[ray]'

See the Ray documentation for `advanced installation instructions <https://docs.ray.io/en/latest/installation.html>`_.


Horovod Ray Executor
--------------------

The Horovod Ray integration offers a ``RayExecutor`` abstraction (:ref:`docs <horovod_ray_api>`),
which is a wrapper over a group of `Ray actors (stateful processes) <https://docs.ray.io/en/latest/walkthrough.html#remote-classes-actors>`_.

.. code-block:: python

    from horovod.ray import RayExecutor

    # Start the Ray cluster or attach to an existing Ray cluster
    ray.init()

    # Start num_hosts * num_slots actors on the cluster
    executor = RayExecutor(
        setting, num_hosts=num_hosts, num_slots=num_slots, use_gpu=True)

    # Launch the Ray actors on each machine
    # This will launch `num_slots` actors on each machine
    executor.start()


All actors will be part of the Horovod ring, so ``RayExecutor`` invocations will be able to support arbitrary Horovod collective operations.

Note that there is an implicit assumption on the cluster being homogenous in shape (i.e., all machines have the same number of slots available). This is simply
an implementation detail and is not a fundamental limitation.

To actually execute a function, you can run the following:

.. code-block:: python

    # Using the stateless `run` method, a function can take in any args or kwargs
    def simple_fn():
        hvd.init()
        print("hvd rank", hvd.rank())
        return hvd.rank()

    # Execute the function on all workers at once
    result = executor.run(simple_fn)
    # Check that the rank of all workers is unique
    assert len(set(result)) == hosts * num_slots

    executor.shutdown()


Stateful Execution
~~~~~~~~~~~~~~~~~~

A unique feature of Ray is its support for `stateful Actors <https://docs.ray.io/en/latest/walkthrough.html#remote-classes-actors>`_. This means that you can start arbitrary Python classes on each worker, easily supporting operations and calls where data is cached in memory.

.. code-block:: python

    import torch
    from horovod.torch import hvd
    from horovod.ray import RayExecutor

    class MyModel:
        def __init__(self, learning_rate):
            self.model = NeuralNet()
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
            )
            self.optimizer = hvd.DistributedOptimizer(optimizer)

        def get_weights(self):
            return dict(self.model.parameters())

        def train(self):
            return self._train(self.model, self.optimizer)


    ray.init()
    executor = RayExecutor(...)
    executor.start(executable_cls=MyModel)

    # Run 5 training steps
    for i in range(5):
        # Stateful `execute` method takes the current worker executable as a parameter
        executor.execute(lambda worker: worker.train())

    # Obtain the trained weights from each model replica
    result = executor.execute(lambda worker: worker.get_weights())

    # `result` will be N copies of the model weights
    assert all(isinstance(res, dict) for res in result)

Elastic Ray Executor
--------------------

Ray also supports `elastic execution <elastic.rst>`_ via :ref:`the ElasticRayExecutor <horovod_ray_api>`. Similar to default Horovod, the difference between the non-elastic and elastic versions of Ray is that the hosts and number of workers is dynamically determined at runtime.

You must first set up `a Ray cluster`_. Ray clusters can support autoscaling for any cloud provider (AWS, GCP, Azure).

.. code-block:: bash

    # First, run `pip install boto3` and `aws configure`
    #
    # Create or update the cluster. When the command finishes, it will print
    # out the command that can be used to SSH into the cluster head node.
    $ ray up ray/python/ray/autoscaler/aws/example-full.yaml

After you have a Ray cluster setup, you will need to move parts of your existing elastic Horovod training script into a training function. Specifically,
the instantiation of your model and the invocation of the ``hvd.elastic.run`` call should be done inside this function.

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

You can then attach to the underlying Ray cluster and execute the training function:

.. code-block:: python

    import ray
    ray.init(address="auto")  # attach to the Ray cluster
    settings = ElasticRayExecutor.create_settings(verbose=True)
    executor = ElasticRayExecutor(
        settings, use_gpu=True, cpus_per_slot=2)
    executor.start()
    executor.run(training_fn)

Ray will automatically start remote actors which execute ``training_fn`` on nodes as they become available. Note that ``executor.run`` call will terminate whenever any one of the training functions terminates successfully, or if all workers fail.

AWS: Cluster Launcher
---------------------

You can also easily leverage the `Ray cluster launcher <https://docs.ray.io/en/latest/cluster/launcher.html>`_ to spin up cloud instances.

.. code-block:: yaml

    # Save as `ray_cluster.yaml`

    cluster_name: horovod-cluster
    provider: {type: aws, region: us-west-2}
    auth: {ssh_user: ubuntu}
    min_workers: 3
    max_workers: 3

    # Deep Learning AMI (Ubuntu) Version 21.0
    head_node: {InstanceType: p3.2xlarge, ImageId: ami-0b294f219d14e6a82}
    worker_nodes: {InstanceType: p3.2xlarge, ImageId: ami-0b294f219d14e6a82}
    setup_commands: # Set up each node.
        - HOROVOD_WITH_GLOO=1 HOROVOD_GPU_OPERATIONS=NCCL pip install horovod[ray]

You can start the specified Ray cluster and monitor its status with:

.. code-block:: bash

    $ ray up ray_cluster.yaml  # starts the head node
    $ ray monitor ray_cluster.yaml  # wait for worker nodes

Then, in your python script, make sure you add ``ray.init(address="auto")`` to connect
to the distributed Ray cluster.

.. code-block:: diff

    -ray.init()
    +ray.init(address="auto")

Then you can execute Ray scripts on the cluster:

.. code-block:: bash

    $ ray submit ray_cluster.yaml <your_script.py>

    # the above is is equivalent to
    $ ray attach ray_cluster.yaml  # ssh
    ubuntu@ip-172-31-24-53:~$ python <your_script.py>

.. inclusion-marker-end-do-not-remove
