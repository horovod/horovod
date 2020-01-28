.. inclusion-marker-start-do-not-remove


Run Horovod
===========

This page includes examples for Open MPI that use ``horovodrun``. Check your
MPI documentation for arguments to the ``mpirun``
command on your system.

Typically one GPU will be allocated per process, so if a server has 4 GPUs,
you will run 4 processes. In ``horovodrun``,
the number of processes is specified with the ``-np`` flag.

To run on a machine with 4 GPUs:

.. code-block:: bash

    $ horovodrun -np 4 -H localhost:4 python train.py

To run on 4 machines with 4 GPUs each:

.. code-block:: bash

    $ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py

You can also specify host nodes in a host file. For example:

.. code-block:: bash

    $ cat myhostfile

    aa slots=2
    bb slots=2
    cc slots=2

This example lists the host names (aa, bb, and cc) and how many "slots" there
are for each.
Slots indicate how many processes can potentially execute on a node.
This format is the same as in
`mpirun command <https://www.open-mpi.org/doc/v4.0/man1/mpirun.1.php#toc6>`__.

To run on hosts specified in a hostfile:

.. code-block:: bash

    $ horovodrun -np 6 -hostfile myhostfile python train.py


Requirements
~~~~~~~~~~~~

Usage of ``horovodrun`` requires one of the following:

* Open MPI >= 2.X
* Spectrum MPI
* MPICH
* OpenRTE
* Gloo

If you do not have MPI installed, you can run ``horovodrun`` using Gloo.  Gloo dependencies come with Horovod
automatically, and only require CMake to be available on your system at the time you install Horovod.

If you wish to use a different version of MPI, you may still be able to run Horovod using `mpirun <mpirun.rst>`
directly.


Failures due to SSH issues
~~~~~~~~~~~~~~~~~~~~~~~~~~
The host where ``horovodrun`` is executed must be able to SSH to all other
hosts without any prompts.

If ``horovodrun`` fails with a permission error, verify that you can ssh to
every other server without entering a password or
answering questions like this:


``The authenticity of host '<hostname> (<ip address>)' can't be established.
RSA key fingerprint is xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx.
Are you sure you want to continue connecting (yes/no)?``


To learn more about setting up passwordless authentication, see `this page <http://www.linuxproblem.org/art_9.html>`__.

To avoid ``The authenticity of host '<hostname> (<ip address>)' can't be
established`` prompts, add all the hosts to
the ``~/.ssh/known_hosts`` file using ``ssh-keyscan``:

.. code-block:: bash

    $ ssh-keyscan -t rsa,dsa server1 server2 > ~/.ssh/known_hosts


Advanced: Run Horovod with Open MPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In some advanced cases you might want fine-grained control over options passed to Open MPI.
To learn how to run Horovod training directly using Open MPI,
read `Run Horovod with Open MPI <mpirun.rst>`_.

.. inclusion-marker-end-do-not-remove
