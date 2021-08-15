.. inclusion-marker-start-do-not-remove

Process Sets: Concurrently Running Collective Operations
========================================================

Most Horovod operations in TensorFlow, PyTorch, or MXNet feature a ``process_set`` argument: By setting up different
process sets you may have multiple subsets of the world of Horovod processes run distinct collective operations in
parallel. Besides Horovod's fundamental operations like ``hvd.allgather``, ``hvd.allreduce``, ``hvd.alltoall``,
``hvd.broadcast``, or ``hvd.grouped_allreduce``, also many high-level utility objects such as
``hvd.DistributedOptimizer`` come with support for process sets.

As an example consider building a Horovod model to be trained by four worker processes with two concurrent allreduce
operations on the "even" or "odd" subset.  In the following we will see three ways to configure Horovod to use an even
and an odd process set, offering you as much flexibility as you need. The code examples are presented for TensorFlow,
but the interface for the other supported frameworks is equivalent.

1) Static process sets
----------------------

.. code-block:: python

    # on all ranks
    even_set = hvd.ProcessSet([0,2])
    odd_set = hvd.ProcessSet([1,3])
    hvd.init(process_sets=[even_set, odd_set])

    for p in [hvd.global_process_set, even_set, odd_set]:
      print(p)
    # ProcessSet(process_set_id=0, ranks=[0, 1, 2, 3], mpi_comm=None)
    # ProcessSet(process_set_id=1, ranks=[0, 2], mpi_comm=None)
    # ProcessSet(process_set_id=2, ranks=[1, 3], mpi_comm=None)

    # on ranks 0 and 2
    result = hvd.allreduce(tensor_for_even_ranks, process_set=even_set)

    # on ranks 1 and 3
    result = hvd.allreduce(tensor_for_odd_ranks, process_set=odd_set)

Having initialized Horovod like this, the configuration of process sets cannot be changed without restarting the
program.  If you only use the default global process set (``hvd.global_process_set``), there is no impact on
performance.

2) Static process sets from MPI communicators
---------------------------------------------

.. code-block:: python

    # on all ranks
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
                                   key=MPI.COMM_WORLD.rank)

    split_process_set = hvd.ProcessSet(subcomm)

    hvd.init(comm, process_sets=[split_process_set])

    for p in [hvd.global_process_set, split_process_set]:
        print(p)
    # ProcessSet(process_set_id=0, ranks=[0, 1, 2, 3], mpi_comm=<mpi4py.MPI.Intracomm object at 0x7fb817323dd0>)
    # ProcessSet(process_set_id=1, ranks=[0, 2], mpi_comm=<mpi4py.MPI.Intracomm object at 0x7fb87e2ddfb0>)
    ## (split_process_set differs by rank)

    # on ranks 0 and 2
    result = hvd.allreduce(tensor_for_even_ranks, process_set=split_process_set)

    # on ranks 1 and 3
    result = hvd.allreduce(tensor_for_odd_ranks, process_set=split_process_set)

If you are already using multiple MPI communicators in your distributed program, you can plug them right in.

3) Dynamic process sets
-----------------------

.. code-block:: python

    # on all ranks
    hvd.init(process_sets="dynamic")  # alternatively set HOROVOD_DYNAMIC_PROCESS_SETS=1
    even_set = hvd.add_process_set([0,2])
    odd_set = hvd.add_process_set([1,3])

    for p in [hvd.global_process_set, even_set, odd_set]:
      print(p)
    # ProcessSet(process_set_id=0, ranks=[0, 1, 2, 3], mpi_comm=None)
    # ProcessSet(process_set_id=1, ranks=[0, 2], mpi_comm=None)
    # ProcessSet(process_set_id=2, ranks=[1, 3], mpi_comm=None)

    # on ranks 0 and 2
    result = hvd.allreduce(tensor_for_even_ranks, process_set=even_set)

    # on ranks 1 and 3
    result = hvd.allreduce(tensor_for_odd_ranks, process_set=odd_set)

The most flexible setup is achieved with "dynamic" process sets.  Process sets can be registered and deregistered
dynamically at any time after initializing Horovod via ``hvd.add_process_set()`` and ``hvd.remove_process_set()``.
Calls to these functions must be made identically and in the same order by all processes.

Note that dynamic process sets come with some slight extra synchronization overhead.

.. inclusion-marker-end-do-not-remove
