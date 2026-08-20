Horovod documentation
=====================

Horovod is a launcher for distributed deep learning training jobs. This is a
**launcher-only** build: ``horovodrun`` (or the Python API ``horovod.runner.run``)
starts one process per slot for single-node or multi-node jobs, and the distributed
communication inside the launched processes is handled by your own framework code
(for example PyTorch ``torch.distributed``).

The TensorFlow / PyTorch / MXNet Horovod integrations and the C++ allreduce core
have been removed from this build, so there is nothing to compile and installation
is a pure-Python ``pip install``.

Get started
-----------

Install the launcher:

.. code-block:: bash

    $ pip install .

Launch a training script on the local machine (one process per GPU), using the
pure-Python Gloo launcher:

.. code-block:: bash

    $ horovodrun -np 4 --gloo python train.py

Launch across multiple hosts:

.. code-block:: bash

    $ horovodrun -np 8 -H host1:4,host2:4 --gloo python train.py

See ``examples/`` for a complete PyTorch ``torch.distributed`` example, including
the mapping from the ``HOROVOD_*`` environment variables set by the launcher to the
variables ``torch.distributed`` expects.

Guides
------

.. toctree::
   :maxdepth: 2

   running_include

   mpi_include

   lsf_include



Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
