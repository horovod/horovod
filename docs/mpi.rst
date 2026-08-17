.. inclusion-marker-start-do-not-remove

Horovod with MPI
================

MPI can be used as an alternative to Gloo for coordinating work between the
processes that ``horovodrun`` launches. This build of Horovod is a launcher-only
package: ``mpirun``/``horovodrun`` start the processes, and the distributed
communication inside them is handled by your own framework code (for example
PyTorch ``torch.distributed``).

First install `Open MPI <https://www.open-mpi.org/>`_ or another MPI
implementation. Learn how to install Open MPI `on this page
<https://www.open-mpi.org/faq/?category=building#easy-build>`_.

**Note**: Open MPI 3.1.3 has an issue that may cause hangs. The recommended fix is
to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

mpirun
------

``horovodrun`` introduces a convenient, Open MPI-based wrapper for launching your
training script. In some cases it is desirable to have fine-grained control over
the options passed to Open MPI; this page describes launching directly with
``mpirun``.

1. Launch on a machine with 4 GPUs:

   .. code-block:: bash

       horovodrun -np 4 python train.py

   Equivalent Open MPI command:

   .. code-block:: bash

       mpirun -np 4 \
           -bind-to none -map-by slot \
           -x LD_LIBRARY_PATH -x PATH \
           -mca pml ob1 -mca btl ^openib \
           python train.py

2. Launch on 4 machines with 4 GPUs each:

   .. code-block:: bash

      horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py

   Equivalent Open MPI command:

   .. code-block:: bash

       mpirun -np 16 \
           -H server1:4,server2:4,server3:4,server4:4 \
           -bind-to none -map-by slot \
           -x LD_LIBRARY_PATH -x PATH \
           -mca pml ob1 -mca btl ^openib \
           python train.py

Starting with Open MPI 3, it is important to add the ``-bind-to none`` and
``-map-by slot`` arguments. ``-bind-to none`` tells Open MPI not to bind a training
process to a single CPU core (which would hurt performance). ``-map-by slot``
allows a mixture of different NUMA configurations, because the default behavior is
to bind to the socket.

The ``-mca pml ob1`` and ``-mca btl ^openib`` flags force the use of TCP for MPI
communication. This avoids many multiprocessing issues that Open MPI has with RDMA,
which typically result in segmentation faults.

With the ``-x`` option you can set (``-x NCCL_DEBUG=INFO``) or copy
(``-x LD_LIBRARY_PATH``) an environment variable to all the workers.

Custom SSH ports
~~~~~~~~~~~~~~~~

Specify custom SSH ports with ``-mca plm_rsh_args "-p <port>"`` as follows:

.. code-block:: bash

    mpirun -np 16 \
        -H server1:4,server2:4,server3:4,server4:4 \
        -bind-to none -map-by slot \
        -mca plm_rsh_args "-p 12345" \
        -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python train.py

Hangs due to non-routed network interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Having network interfaces that are not routed can cause Open MPI to hang. An
example of such an interface is ``docker0``.

If you see non-routed interfaces (like ``docker0``) in the output of ``ifconfig``,
you should tell Open MPI not to use them via the
``-mca btl_tcp_if_exclude <interface>[,<interface>]`` parameter (and, if you use
NCCL for communication, ``NCCL_SOCKET_IFNAME=^<interface>[,<interface>]``).

Example ``mpirun`` command with the ``lo`` and ``docker0`` interfaces excluded:

.. code-block:: bash

    mpirun -np 16 \
        -H server1:4,server2:4,server3:4,server4:4 \
        -bind-to none -map-by slot \
        -x LD_LIBRARY_PATH -x PATH \
        -x NCCL_SOCKET_IFNAME=^lo,docker0 \
        -mca pml ob1 -mca btl ^openib \
        -mca btl_tcp_if_exclude lo,docker0 \
        python train.py

.. inclusion-marker-end-do-not-remove
