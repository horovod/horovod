.. raw:: html

    <p align="center"><img src="https://user-images.githubusercontent.com/16640218/34506318-84d0c06c-efe0-11e7-8831-0425772ed8f2.png" alt="Logo" width="200"/></p>
    <br/>

Horovod
=======

.. image:: https://badge.buildkite.com/6f976bc161c69d9960fc00de01b69deb6199b25680a09e5e26.svg?branch=master
   :target: https://buildkite.com/horovod/horovod
   :alt: Build Status

.. image:: https://readthedocs.org/projects/horovod/badge/?version=latest
   :target: https://horovod.readthedocs.io/en/latest/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :alt: License

.. image:: https://app.fossa.com/api/projects/git%2Bgithub.com%2Fhorovod%2Fhorovod.svg?type=shield
   :target: https://app.fossa.com/projects/git%2Bgithub.com%2Fhorovod%2Fhorovod?ref=badge_shield
   :alt: FOSSA Status

.. image:: https://bestpractices.coreinfrastructure.org/projects/2373/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/2373
   :alt: CII Best Practices

.. image:: https://pepy.tech/badge/horovod
   :target: https://pepy.tech/project/horovod
   :alt: Downloads

.. inclusion-marker-start-do-not-remove

|

Horovod is a distributed training framework for TensorFlow, Keras, PyTorch, and MXNet. The goal of Horovod is to make
distributed Deep Learning fast and easy to use.


.. raw:: html

   <p><img src="https://raw.githubusercontent.com/lfai/artwork/master/lfai/horizontal/color/lfai-color.png" alt="LF AI" width="200"/></p>


Horovod is hosted by the `LF AI Foundation <https://lfdl.io>`_ (LF AI). If you are a company that is deeply
committed to using open source technologies in artificial intelligence, machine and deep learning, and wanting to support
the communities of open source projects in these domains, consider joining the LF AI Foundation. For details
about who's involved and how Horovod plays a role, read the LF AI `announcement <https://lfdl.io/press/2018/12/13/lf-deep-learning-welcomes-horovod-distributed-training-framework-as-newest-project/>`_.

|

.. contents::

|

Why not traditional Distributed TensorFlow?
-------------------------------------------

The primary motivation for this project is to make it easy to take a single-GPU TensorFlow program and successfully train
it on many GPUs faster. This has two aspects:

1. How much modification does one have to make to a program to make it distributed, and how easy is it to run it?
2. How much faster would it run in distributed mode?

Internally at Uber we found the MPI model to be much more straightforward and require far less code changes than the
Distributed TensorFlow with parameter servers. See the `Usage <#usage>`__ section for more details.

In addition to being easy to use, Horovod is fast. Below is a chart representing the benchmark that was done on 128
servers with 4 Pascal GPUs each connected by RoCE-capable 25 Gbit/s network:

.. image:: https://user-images.githubusercontent.com/16640218/38965607-bf5c46ca-4332-11e8-895a-b9c137e86013.png
   :alt: 512-GPU Benchmark

Horovod achieves 90% scaling efficiency for both Inception V3 and ResNet-101, and 68% scaling efficiency for VGG-16.
See the `Benchmarks <docs/benchmarks.rst>`_ page to find out how to reproduce these numbers.

While installing MPI and NCCL itself may seem like an extra hassle, it only needs to be done once by the team dealing
with infrastructure, while everyone else in the company who builds the models can enjoy the simplicity of training them at
scale.


Install
-------

To install Horovod:

1. Install `Open MPI <https://www.open-mpi.org/>`_ or another MPI implementation. Learn how to install Open MPI `on this page <https://www.open-mpi.org/faq/?category=building#easy-build>`_.

   **Note**: Open MPI 3.1.3 has an issue that may cause hangs.  The recommended fix is to
   downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

.. raw:: html

    <p/>

2. If you've installed TensorFlow from `PyPI <https://pypi.org/project/tensorflow>`__, make sure that the ``g++-4.8.5`` or ``g++-4.9`` is installed.

   If you've installed PyTorch from `PyPI <https://pypi.org/project/torch>`__, make sure that the ``g++-4.9`` or above is installed.

   If you've installed either package from `Conda <https://conda.io>`_, make sure that the ``gxx_linux-64`` Conda package is installed.

.. raw:: html

    <p/>

3. Install the ``horovod`` pip package.

.. code-block:: bash

    $ pip install horovod

This basic installation is good for laptops and for getting to know Horovod.

If you're installing Horovod on a server with GPUs, read the `Horovod on GPU <docs/gpus.rst>`_ page.

If you want to use Docker, read the `Horovod in Docker <docs/docker.rst>`_ page.


Concepts
--------

Horovod core principles are based on `MPI <http://mpi-forum.org/>`_ concepts such as *size*, *rank*,
*local rank*, **allreduce**, **allgather** and, *broadcast*. See `this page <docs/concepts.rst>`_ for more details.

Supported frameworks
--------------------
See these pages for Horovod examples and best practices:

- `Horovod with TensorFlow <#usage>`__ (Usage section below)
- `Horovod with Keras <docs/keras.rst>`_
- `Horovod with PyTorch <docs/pytorch.rst>`_
- `Horovod with MXNet <docs/mxnet.rst>`_

Usage
-----

To use Horovod, make the following additions to your program. This example uses TensorFlow.

1. Run ``hvd.init()``.

2. Pin a server GPU to be used by this process using ``config.gpu_options.visible_device_list``.
   With the typical setup of one GPU per process, this can be set to *local rank*. In that case, the first process on
   the server will be allocated the first GPU, second process will be allocated the second GPU and so forth.

3. Scale the learning rate by number of workers. Effective batch size in synchronous distributed training is scaled by
   the number of workers. An increase in learning rate compensates for the increased batch size.

4. Wrap optimizer in ``hvd.DistributedOptimizer``.  The distributed optimizer delegates gradient computation
   to the original optimizer, averages gradients using **allreduce** or **allgather**, and then applies those averaged
   gradients.

5. Add ``hvd.BroadcastGlobalVariablesHook(0)`` to broadcast initial variable states from rank 0 to all other processes.
   This is necessary to ensure consistent initialization of all workers when training is started with random weights or
   restored from a checkpoint. Alternatively, if you're not using ``MonitoredTrainingSession``, you can simply execute
   the ``hvd.broadcast_global_variables`` op after global variables have been initialized.

6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.
   This can be accomplished by passing ``checkpoint_dir=None`` to ``tf.train.MonitoredTrainingSession`` if
   ``hvd.rank() != 0``.

Example (see the `examples <https://github.com/horovod/horovod/blob/master/examples/>`_ directory for full training examples):

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


Running Horovod
---------------

The example commands below show how to run distributed training. See the `Running Horovod <docs/running.rst>`_
page for more instructions, including RoCE/InfiniBand tweaks and tips for dealing with hangs.

1. To run on a machine with 4 GPUs:

.. code-block:: bash

     $ horovodrun -np 4 -H localhost:4 python train.py

2. To run on 4 machines with 4 GPUs each:

.. code-block:: bash

    $ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py

3. To run using Open MPI without the ``horovodrun`` wrapper, see the `Running Horovod with Open MPI <docs/mpirun.rst>`_ page.

4. To run in Docker, see the `Horovod in Docker <docs/docker.rst>`_ page.

5. To run in Kubernetes, see `Kubeflow <https://github.com/kubeflow/kubeflow/tree/master/kubeflow/mpi-job>`_, `MPI Operator <https://github.com/kubeflow/mpi-operator/>`_, `Helm Chart <https://github.com/kubernetes/charts/tree/master/stable/horovod/>`_, and `FfDL <https://github.com/IBM/FfDL/tree/master/etc/examples/horovod/>`_.

6. To run in Spark, see the `Spark <docs/spark.rst>`_ page.

7. To run in Singularity, see `Singularity <https://github.com/sylabs/examples/tree/master/machinelearning/horovod>`_.

Gloo
----
`Gloo <https://github.com/facebookincubator/gloo>`_ is an open source collective communications library developed by Facebook.

Gloo comes included with Horovod, and allows users to run Horovod without requiring MPI to be installed. Gloo support only requires
that you have `CMake <https://cmake.org/>`_ installed, and is only supported on Linux at this time.

For environments that have support both MPI and Gloo, you can choose to use Gloo at runtime by passing the ``--gloo`` argument to ``horovodrun``:

.. code-block:: bash

     $ horovodrun --gloo -np 2 python train.py

Gloo support is still early in its development, and more features are coming soon.

mpi4py
------
Horovod supports mixing and matching Horovod collectives with other MPI libraries, such as `mpi4py <https://mpi4py.scipy.org>`_,
provided that the MPI was built with multi-threading support.

You can check for MPI multi-threading support by querying the ``hvd.mpi_threads_supported()`` function.

.. code-block:: python

    import horovod.tensorflow as hvd

    # Initialize Horovod
    hvd.init()

    # Verify that MPI multi-threading is supported.
    assert hvd.mpi_threads_supported()

    from mpi4py import MPI
    assert hvd.size() == MPI.COMM_WORLD.Get_size()

You can also initialize Horovod with an `mpi4py` sub-communicator, in which case each sub-communicator
will run an independent Horovod training.

.. code-block:: python

    from mpi4py import MPI
    import horovod.tensorflow as hvd

    # Split COMM_WORLD into subcommunicators
    subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
                                   key=MPI.COMM_WORLD.rank)

    # Initialize Horovod
    hvd.init(comm=subcomm)

    print('COMM_WORLD rank: %d, Horovod rank: %d' % (MPI.COMM_WORLD.rank, hvd.rank()))


Inference
---------
Learn how to optimize your model for inference and remove Horovod operations from the graph `here <docs/inference.rst>`_.


Tensor Fusion
-------------
One of the unique things about Horovod is its ability to interleave communication and computation coupled with the ability
to batch small **allreduce** operations, which results in improved performance. We call this batching feature Tensor Fusion.

See `here <docs/tensor-fusion.rst>`__ for full details and tweaking instructions.


Analyzing Horovod Performance
-----------------------------
Horovod has the ability to record the timeline of its activity, called Horovod Timeline.

.. image:: https://user-images.githubusercontent.com/16640218/29735271-9e148da0-89ac-11e7-9ae0-11d7a099ac89.png
   :alt: Horovod Timeline

See `here <docs/timeline.rst>`__ for full details and usage instructions.


Automated Performance Tuning
----------------------------
Selecting the right values to efficiently make use of Tensor Fusion and other advanced Horovod features can involve
a good amount of trial and error. We provide a system to automate this performance optimization process called
**autotuning**, which you can enable with a single command line argument to ``horovodrun``.

See `here <docs/autotune.rst>`__ for full details and usage instructions.


Guides
------
1. Run distributed training in Microsoft Azure using `Batch AI and Horovod <https://github.com/Azure/BatchAI/tree/master/recipes/Horovod>`_. Send us links to any user guides you want to publish on this site

Troubleshooting
---------------
See the `Troubleshooting <docs/troubleshooting.rst>`_ page and please submit a `ticket <https://github.com/horovod/horovod/issues/new>`_
if you can't find an answer.


Citation
--------
Please cite Horovod in your publications if it helps your research:

::

    @article{sergeev2018horovod,
      Author = {Alexander Sergeev and Mike Del Balso},
      Journal = {arXiv preprint arXiv:1802.05799},
      Title = {Horovod: fast and easy distributed deep learning in {TensorFlow}},
      Year = {2018}
    }


Publications
------------
1. Sergeev, A., Del Balso, M. (2017) *Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow*.
Retrieved from `https://eng.uber.com/horovod/ <https://eng.uber.com/horovod/>`_

2. Sergeev, A. (2017) *Horovod - Distributed TensorFlow Made Easy*. Retrieved from
`https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy <https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy>`_

3. Sergeev, A., Del Balso, M. (2018) *Horovod: fast and easy distributed deep learning in TensorFlow*. Retrieved from
`arXiv:1802.05799 <https://arxiv.org/abs/1802.05799>`_


References
----------
The Horovod source code was based off the Baidu `tensorflow-allreduce <https://github.com/baidu-research/tensorflow-allreduce>`_
repository written by Andrew Gibiansky and Joel Hestness. Their original work is described in the article
`Bringing HPC Techniques to Deep Learning <http://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/>`_.

Mailing lists
-------------
Subscribe to `Horovod Announce <https://lists.lfai.foundation/g/horovod-announce>`_ and 
`Horovod Technical-Discuss <https://lists.lfai.foundation/g/horovod-technical-discuss>`_ to stay up to date.


.. inclusion-marker-end-do-not-remove
   Place contents above here if they should also appear in read-the-docs.
   Contents below are already part of the read-the-docs table of contents.
