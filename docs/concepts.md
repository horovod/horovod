## Concepts

Horovod core principles are based on the [MPI](http://mpi-forum.org/) concepts *size*, *rank*,
*local rank*, *allreduce*, *allgather*, and *broadcast*. These are best explained by example. Say we launched
a training script on 4 servers, each having 4 GPUs. If we launched one copy of the script per GPU:

* *Size* would be the number of processes, in this case, 16.

* *Rank* would be the unique process ID from 0 to 15 (*size* - 1).

* *Local rank* would be the unique process ID within the server from 0 to 3.

* *Allreduce* is an operation that aggregates data among multiple processes and distributes
    results back to them.  *Allreduce* is used to average dense tensors.  Here's an illustration from the
    [MPI Tutorial](http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/):

    ![Allreduce Illustration](http://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/mpi_allreduce_1.png)

* *Allgather* is an operation that gathers data from all processes on every process.  *Allgather* is used to collect
    values of sparse tensors.  Here's an illustration from the [MPI Tutorial](http://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/):

    ![Allgather Illustration](http://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/allgather.png)

* *Broadcast* is an operation that broadcasts data from one process, identified by root rank, onto every other process.
    Here's an illustration from the [MPI Tutorial](http://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/):

    ![Broadcast Illustration](http://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/broadcast_pattern.png)
