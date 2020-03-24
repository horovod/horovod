.. inclusion-marker-start-do-not-remove


Horovod in LSF
==============

This page includes examples for running Horovod in a LSF cluster.
``horovodrun`` will automatically detect the host names and GPUs of your LSF job.
If the LSF cluster supports ``jsrun``, ``horovodrun`` will use it as launcher
otherwise it will default to ``mpirun``.

Inside a LSF batch file or in an interactive session, you just need to use:

.. code-block:: bash

    horovodrun python train.py

Here, Horovod will start a process per GPU on all the hosts of the LSF job.

You can also limit the run to a subset of the job resources. For example, using only 6 GPUs:

.. code-block:: bash

    horovodrun -np 6 python train.py

You can still pass extra arguments to ``horovodrun``. For example, to trigger CUDA-Aware MPI:

.. code-block:: bash

    horovodrun --mpi-args="-gpu" python train.py

.. inclusion-marker-end-do-not-remove
