.. inclusion-marker-start-do-not-remove

Horovod with Intel(R) oneCCL
============================
To use Horovod with the Intel(R) oneAPI Collective Communications Library (oneCCL), follow the steps below.

1. Install `oneCCL <https://github.com/intel/oneccl>`_.

To install oneCCL, follow `these steps <https://github.com/intel/oneccl/blob/master/README.md>`_.

Source ``setvars.sh`` to start using oneCCL.

.. code-block:: bash

    source <install_dir>/env/setvars.sh

2. Set ``HOROVOD_CPU_OPERATIONS`` variable
    
.. code-block:: bash

    export HOROVOD_CPU_OPERATIONS=CCL

3. Install Horovod from source code

.. code-block:: bash

    python setup.py build
    python setup.py install

or via pip 

.. code-block:: bash
    
    pip install horovod

Advanced settings
*****************

Affinity
--------

You can specify the affinity for Horovod background thread with the ``HOROVOD_THREAD_AFFINITY`` environment variable.
See the instructions below.

Set Horovod background thread affinity according to the rule - if there is N Horovod processes per node, this variable should contain all the values for every local process using comma as a separator:

.. code-block:: bash
    
    export HOROVOD_THREAD_AFFINITY=c0,c1,...,c(N-1)

where c0,...,c(N-1) are core IDs to pin background threads from local processes.


Set the number of oneCCL workers:

.. code-block:: bash
    
    export CCL_WORKER_COUNT=X

where X is the number of oneCCL worker threads (workers) per process you'd like to dedicate to drive communication.


Set oneCCL workers affinity automatically:

.. code-block:: bash

    export CCL_WORKER_AFFINITY=auto

This is default mode. The exact core IDs will depend from process launcher used.

Set oneCCL workers affinity explicitly:

.. code-block:: bash

    export CCL_WORKER_AFFINITY=c0,c1,..,c(X-1)

where c0,c1,..,c(X-1) are core IDs dedicated to local oneCCL workers, i.e. X = ``CCL_WORKER_COUNT`` * Number of processes per node.

Please refer to `Execution of Communication Operations <https://oneapi-src.github.io/oneCCL/operation_execution.html>`_ for more information.


For example, we have 2 nodes and each node has 2 sockets: socket0 CPUs: 0-17,36-53 and socket1 CPUs: 18-35,54-71. We dedicate the last two cores of each socket for 2 oneCCL workers and pin Horovod background thread to one of the hyper-thread cores of oneCCL workers's cores. All these cores are excluded from Intel MPI pinning using ``I_MPI_PIN_PROCESSOR_EXCLUDE_LIST`` to dedicate them to oneCCL and Horovod tasks only, thus avoiding the conflict with framework's computational threads.

.. code-block:: bash
    
    export CCL_WORKER_COUNT=2
    export CCL_WORKER_AFFINITY="16,17,34,35"
    export HOROVOD_THREAD_AFFINITY="53,71"
    export I_MPI_PIN_DOMAIN=socket
    export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="16,17,34,35,52,53,70,71"

    mpirun -n 4 -ppn 2 -hostfile hosts python ./run_example.py


Caching
-------

Set cache hint for oneCCL operations:

.. code-block:: bash
    
    export HOROVOD_CCL_CACHE=0|1

Available for ``allreduce`` only yet. Disabled by default.

Please refer to `Caching of Communication Operations <https://oneapi-src.github.io/oneCCL/operation_caching.html>`_ for more information.

.. inclusion-marker-end-do-not-remove
