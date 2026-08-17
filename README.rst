Horovod (launcher-only build)
=============================

This is a **launcher-only** fork of `Horovod <https://github.com/horovod/horovod>`_.
It keeps only the ``horovodrun`` command-line launcher and the ``horovod.runner``
Python package, which launch single-node and multi-node multi-process (multi-GPU)
training jobs.

The TensorFlow / PyTorch / MXNet framework integrations and the C++ allreduce core
have been **removed**. ``horovodrun`` only starts one process per slot and sets
``HOROVOD_*`` environment variables; the distributed communication inside the
launched processes is handled by your own framework code — for example PyTorch
``torch.distributed``. Because there is no C++ to compile, installation is a
pure-Python ``pip install``.

.. contents::
   :local:

Install
-------

.. code-block:: bash

    $ pip install .

No CMake, C++ compiler, or MPI build is required for the launcher itself.

Usage
-----

Launch a training script with one process per GPU on the local machine, using the
pure-Python Gloo launcher:

.. code-block:: bash

    $ horovodrun -np 4 --gloo python train.py

Launch across multiple hosts (requires passwordless SSH between hosts):

.. code-block:: bash

    $ horovodrun -np 8 -H host1:4,host2:4 --gloo python train.py

Launch through an external ``mpirun``:

.. code-block:: bash

    $ horovodrun -np 4 --mpi python train.py

Check which launch controllers are available:

.. code-block:: bash

    $ horovodrun --check-build

Launch backends
---------------

* **Gloo** (``--gloo``) — a pure-Python launcher with no external dependencies; it
  is always available and does not require MPI or CMake.
* **MPI** (``--mpi``) — wraps an external ``mpirun``/``mpiexec``. It is also chosen
  by default when a ``mpirun`` is found on ``PATH``. See ``docs/mpi.rst`` for
  launching directly with ``mpirun``.
* **LSF / jsrun** — on IBM LSF clusters ``horovodrun`` detects the job and uses
  ``jsrun``. See ``docs/lsf.rst``.

Using PyTorch ``torch.distributed``
-----------------------------------

The launcher sets ``HOROVOD_*`` environment variables (``HOROVOD_RANK``,
``HOROVOD_SIZE``, ``HOROVOD_LOCAL_RANK``, ``HOROVOD_GLOO_RENDEZVOUS_ADDR``, ...),
whereas ``torch.distributed.init_process_group(init_method="env://")`` expects
``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR`` and ``MASTER_PORT``. Map
them in your training script.

See ``examples/pytorch_torch_distributed.py`` and ``examples/README.md`` for a
complete, runnable example of launching PyTorch DDP training with ``horovodrun``.

Documentation
-------------

* ``docs/running.rst`` — launching with ``horovodrun`` (Gloo), hosts/hostfiles, SSH.
* ``docs/mpi.rst`` — launching directly with ``mpirun``.
* ``docs/lsf.rst`` — launching on IBM LSF with ``jsrun``.

Tests
-----

Run the launcher unit tests:

.. code-block:: bash

    $ pip install -e .[test]
    $ pytest test/single

Provenance
----------

This project is derived from `Horovod <https://github.com/horovod/horovod>`_
(Apache 2.0), slimmed down to the process launcher. See ``LICENSE``.
