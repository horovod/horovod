:orphan:

Run Horovod with Open MPI
=========================
``horovodrun`` introduces a convenient, Open MPI-based wrapper for running Horovod scripts.

In some cases it is desirable to have fine-grained control over options passed to Open MPI.  This page describes
running Horovod training directly using Open MPI.

1. Run on a machine with 4 GPUs:

   .. code-block:: bash

       horovodrun -np 4 python train.py

   Equivalent Open MPI command:

   .. code-block:: bash

       mpirun -np 4 \
           -bind-to none -map-by slot \
           -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
           -mca pml ob1 -mca btl ^openib \
           python train.py

2. Run on 4 machines with 4 GPUs each:

   .. code-block:: bash

      horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py

   Equivalent Open MPI command:

   .. code-block:: bash

       mpirun -np 16 \
           -H server1:4,server2:4,server3:4,server4:4 \
           -bind-to none -map-by slot \
           -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
           -mca pml ob1 -mca btl ^openib \
           python train.py

Starting with the Open MPI 3, it's important to add the ``-bind-to none`` and ``-map-by slot`` arguments.
``-bind-to none`` specifies Open MPI to not bind a training process to a single CPU core (which would hurt performance).
``-map-by slot`` allows you to have a mixture of different NUMA configurations because the default behavior is to bind
to the socket.

The ``-mca pml ob1`` and ``-mca btl ^openib`` flags force the use of TCP for MPI communication.  This avoids many
multiprocessing issues that Open MPI has with RDMA which typically results in segmentation faults.  Using TCP for MPI
does not have noticeable performance impact since most of the heavy communication is done by NCCL, which will use RDMA
via RoCE or InfiniBand if they're available (see `Horovod on GPU <gpus.md>`_).  Notable exceptions from this rule are
models that heavily use ``hvd.broadcast()`` and ``hvd.allgather()`` operations.  To make those operations use RDMA,
read the `Open MPI with RDMA <#open-mpi-with-rdma>`_ section below.

With the ``-x`` option you can specify (``-x NCCL_DEBUG=INFO``) or copy (``-x LD_LIBRARY_PATH``) an environment variable to
all the workers.

Custom SSH ports
----------------

Specify custom SSH ports with ``-mca plm_rsh_args "-p <port>"`` as follows:

.. code-block:: bash

    mpirun -np 16 \
        -H server1:4,server2:4,server3:4,server4:4 \
        -bind-to none -map-by slot \
        -mca plm_rsh_args "-p 12345"
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
        -mca pml ob1 -mca btl ^openib \
        python train.py

This is frequently useful in the case of `running Horovod in Docker environment <docker.md>`_.

Open MPI with RDMA
------------------

As noted above, using TCP for MPI communication does not have any significant effects on performance in the majority of
cases. Models that make heavy use of ``hvd.broadcast()`` and ``hvd.allgather()`` operations are exceptions to that rule.

Default Open MPI ``openib`` BTL that provides RDMA functionality does not work well with MPI multithreading.  In order
to use RDMA with ``openib``, multithreading must be disabled via the ``-x HOROVOD_MPI_THREADS_DISABLE=1`` option.  See the
example below:

.. code-block:: bash

    mpirun -np 16 \
        -H server1:4,server2:4,server3:4,server4:4 \
        -bind-to none -map-by slot \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x PATH \
        -mca pml ob1 \
        python train.py

Other MPI RDMA implementations may or may not benefit from disabling multithreading, so please consult vendor
documentation.

Horovod Parameter Knobs
-----------------------

Many of the configurable parameters available as command line arguments to ``horovodrun`` can be used with ``mpirun``
through the use of environment variables.

Tensor Fusion:

.. code-block:: bash

    $ mpirun -x HOROVOD_FUSION_THRESHOLD=33554432 -x HOROVOD_CYCLE_TIME=3.5 ... python train.py

Timeline:

.. code-block:: bash

    $ mpirun -x HOROVOD_TIMELINE=/path/to/timeline.json -x HOROVOD_TIMELINE_MARK_CYCLES=1 ... python train.py

Autotuning:

.. code-block:: bash

    $ mpirun -x HOROVOD_AUTOTUNE=1 -x HOROVOD_AUTOTUNE_LOG=/tmp/autotune_log.csv ... python train.py

Note that when using ``horovodrun``, any command line arguments will override values set in the environment.

Hangs due to non-routed network interfaces
------------------------------------------

Having network interfaces that are not routed can cause Open MPI to hang. An example of such interface is ``docker0``.

If you see non-routed interfaces (like ``docker0``) in the output of ``ifconfig``, you should tell Open MPI and NCCL to not
use them via the ``-mca btl_tcp_if_exclude <interface>[,<interface>]`` and ``NCCL_SOCKET_IFNAME=^<interface>[,<interface>]``
parameters.

.. code-block:: bash

    ifconfig

Produces output like this::

    docker0   Link encap:Ethernet  HWaddr 02:42:2d:17:ea:66
              inet addr:172.17.0.1  Bcast:0.0.0.0  Mask:255.255.0.0
              UP BROADCAST MULTICAST  MTU:1500  Metric:1
              RX packets:0 errors:0 dropped:0 overruns:0 frame:0
              TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
              collisions:0 txqueuelen:0
              RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
    eth0      Link encap:Ethernet  HWaddr 24:8a:07:b3:7d:8b
              inet addr:10.0.0.1  Bcast:10.0.0.255  Mask:255.255.255.0
              UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
              RX packets:900002410 errors:0 dropped:405 overruns:0 frame:0
              TX packets:1521598641 errors:0 dropped:0 overruns:0 carrier:0
              collisions:0 txqueuelen:1000
              RX bytes:376184431726 (350.3 GiB)  TX bytes:954933846124 (889.3 GiB)
    eth1      Link encap:Ethernet  HWaddr 24:8a:07:b3:7d:8a
              inet addr:192.168.0.1  Bcast:192.168.0.255  Mask:255.255.255.0
              UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
              RX packets:2410141 errors:0 dropped:0 overruns:0 frame:0
              TX packets:2312177 errors:0 dropped:0 overruns:0 carrier:0
              collisions:0 txqueuelen:1000
              RX bytes:698398061 (666.0 MiB)  TX bytes:458504418 (437.2 MiB)
    lo        Link encap:Local Loopback
              inet addr:127.0.0.1  Mask:255.0.0.0
              inet6 addr: ::1/128 Scope:Host
              UP LOOPBACK RUNNING  MTU:65536  Metric:1
              RX packets:497075633 errors:0 dropped:0 overruns:0 frame:0
              TX packets:497075633 errors:0 dropped:0 overruns:0 carrier:0
              collisions:0 txqueuelen:1
              RX bytes:72680421398 (67.6 GiB)  TX bytes:72680421398 (67.6 GiB)

Example ``mpirun`` command with ``lo`` and ``docker0`` interfaces excluded:

.. code-block:: bash

    mpirun -np 16 \
        -H server1:4,server2:4,server3:4,server4:4 \
        -bind-to none -map-by slot \
        -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
        -x NCCL_SOCKET_IFNAME=^lo,docker0 \
        -mca pml ob1 -mca btl ^openib \
        -mca btl_tcp_if_exclude lo,docker0 \
        python train.py
