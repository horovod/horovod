.. inclusion-marker-start-do-not-remove
Distributed Hyperparameter Search
=================================

Horovod's data parallelism training capabilities allows you to scale out and speed up the workload of training a deep learning model. However, distributed training often does not exhibit linear scaling in accuracy improvement - i.e., simply using 2x more workers does not necessarily mean the model will obtain the same accuracy in 2x less time.

To address this, you will often need to re-tune hyperparameters when training at scale, as many hyperparameters exhibit different behaviors at larger scales.

To address this issue, Horovod offers a Ray Tune integration to enable parallel hyperparameter tuning with distributed training.

Ray Tune is an industry standard tool for distributed hyperparameter tuning. Ray Tune includes the latest hyperparameter search algorithms, integrates with TensorBoard and other analysis libraries, and natively supports distributed training. The Ray Tune + Horovod integration leverages the underlying Ray framework to provide a scalable and comprehensive hyperparameter tuning setup.

**By the end of this guide, you will learn:**
* How you can set up Ray Tune and Horovod to tune your hyperparameters
* Typical hyperparameters to configure for distributed training

Horovod + Ray Tune
------------------

You can leverage Ray Tune with Horovod to combine distributed hyperparameter tuning with distributed training. Here is an example demonstrating basic usage:

.. code-block:: python

    import horovod.torch as hvd
    from ray import tune
    import time

    def training_function(config: Dict):
        hvd.init()
        for i in range(config["epochs"]):
            time.sleep(1)
            tune.report(test=1, rank=hvd.rank())

    trainable = DistributedTrainableCreator(
            training_function, num_slots=2, use_gpu=use_gpu)
    analysis = tune.run(
            trainable, num_samples=2, config={"epochs": tune.grid_search([1, 2, 3]))
    print(analysis.best_config)

Not only is the interface very simple, it is also packed with features.

Basic setup
-----------

Use the `DistributedTrainableCreator`_ function to adapt your Horovod training function to be compatible with Tune.

`DistributedTrainableCreator`_ exposes ``num_hosts``, ``num_slots``, ``use_gpu`` and ``num_cpus_per_slot`` - these parameters allow you to specify the resource allocation of a single "trial" (or "Trainable") which itself can be a distributed training job.


.. code-block:: python

    # Each training job will use 2 GPUs.
    trainable = DistributedTrainableCreator(
        training_function, num_slots=2, use_gpu=True)

The training function itself must do three things:

1. It must adhere to the Tune Function API signature.
1. Its body must include a ``horovod.init()`` call.
1. It must call ``tune.report`` during training, typically called iteratively at the end of every epoch.

Setting up a tuning cluster
---------------------------

You can easily leverage Ray Tune with Horovod on a laptop, single machine with multiple GPUs, or across multiple machines. To run on a single machine, you can execute your python script as-is (for example, horovod_simple.py, assuming Ray and Horovod are installed properly):

.. code-block:: bash

    python horovod_simple.py


To leverage a distributed hyperparameter tuning setup with Ray Tune + Horovod, you’ll need to install Ray and set up a Ray cluster. Ray clusters are started with the Ray Cluster Launcher or manually.

Below, we’ll use the Ray cluster launcher, but you can start Ray on any list of nodes, on any cluster manager or cloud provider.

To use the cluster launcher, you’ll first want to specify a configuration file. Below we have an example of using AWS EC2, but you can easily launch the cluster on any cloud provider:

.. code-block:: yaml

    # ray_cluster.yaml
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

You can then run ray up ray_cluster.yaml, and a cluster of 4 nodes will be automatically started for you with Ray.

.. code-block:: bash


    [6/6] Starting the Ray runtime
    Did not find any active Ray processes.
    Shared connection to 34.217.192.11 closed.
    Local node IP: 172.31.43.22
    2020-11-04 04:24:33,882 INFO services.py:1106 -- View the Ray dashboard at http://localhost:8265

    --------------------
    Ray runtime started.
    --------------------

    Next steps
      To connect to this Ray runtime from another node, run
        ray start --address='172.31.43.22:6379' --redis-password='5241590000000000'

      Alternatively, use the following Python code:
        import ray
        ray.init(address='auto', _redis_password='5241590000000000')

      If connection fails, check your firewall settings and network configuration.

      To terminate the Ray runtime, run
        ray stop
    Shared connection to 34.217.192.11 closed.
      New status: up-to-date

    Useful commands
      Monitor autoscaling with
        ray exec /Users/rliaw/dev/cfgs/check-autoscaler.yaml 'tail -n 100 -f /tmp/ray/session_latest/logs/monitor*'
      Connect to a terminal on the cluster head:
        ray attach /Users/rliaw/dev/cfgs/check-autoscaler.yaml
      Get a remote shell to the cluster manually:
        ssh -o IdentitiesOnly=yes -i /Users/rliaw/.ssh/ray-autoscaler_2_us-west-2.pem ubuntu@34.217.192.11

After the cluster is up, you can ssh into the head node and run your Tune script there.


.. _`DistributedTrainableCreator`: https://docs.ray.io/en/latest/tune/api_docs/integration.html#horovod-tune-integration-horovod

.. inclusion-marker-end-do-not-remove
