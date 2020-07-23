## Horovod with Intel(R) oneCCL

To use Horovod with the Intel(R) oneAPI Collective Communications Library (oneCCL), follow the steps below.

1. Install [oneCCL](https://github.com/intel/oneccl).

To install oneCCL, follow [these steps](https://github.com/intel/oneccl/blob/master/README.md).

Source setvars.sh to start using oneCCL.

```bash
$ source <install_dir>/env/setvars.sh
```

2. Install the [Intel(R) MPI Library](https://software.intel.com/en-us/mpi-library).

To install the Intel MPI Library, follow [these steps](https://software.intel.com/en-us/mpi-library/documentation/get-started).

Source the mpivars.sh script to establish the proper environment settings.

```bash
$ source <installdir_MPI>/intel64/bin/mpivars.sh release_mt
```

3. Set HOROVOD_CPU_OPERATIONS variable

```bash
$ export HOROVOD_CPU_OPERATIONS=CCL
```

4. Install Horovod from source code

```bash
$ python setup.py build
$ python setup.py install
```
or via pip 

```bash
$ pip install horovod
```

**Advanced:** You can specify the affinity for BackgroundThread with the HOROVOD_THREAD_AFFINITY environment variable.
See the instructions below.

Set Horovod background thread affinity according to the rule. If there is N Horovod ranks per node, this variable should
contain all the values for every rank using comma as a separator:
```bash
$ export HOROVOD_THREAD_AFFINITY=c0,c1,...,c(N-1)
```
where c0,...,c(N-1) are core IDs to attach background thread to.

Set the number of oneCCL workers:
```bash
$ export CCL_WORKER_COUNT=X
```
where X is the number of threads you'd like to dedicate for driving communication. This means that for every rank there are X oneCCL
workers available.

Set oneCCL workers affinity:
```bash
$ export CCL_WORKER_AFFINITY=c0,c1,..,c(X-1)
```
where c0,c1,..,c(X-1) are core IDs dedicated to oneCCL workers (uses X 'last' cores by default). This variable sets affinity for all
oneCCL workers (CCL_WORKER_COUNT * Number of ranks per node) that are available for all the ranks running on one node.

For instance, we have 2 nodes and each node has 2 sockets: socket0 CPUs:0-17,36-53 and socket1 CPUs:18-35,54-71. We decide to pin CCL
workers to the last two cores of each socket while pinning Horovod background thread to one of the hyper-thread cores of CCL workers's
cores. All these cores are excluded from Intel MPI pinning using I_MPI_PIN_PROCESSOR_EXCLUDE_LIST to dedicate them to CCL and Horovod
tasks only, thus avoiding the conflict with framework's computational threads. 
```bash
$ export I_MPI_PIN_PROCESSOR_EXCLUDE_LIST="16,17,34,35,52,53,70,71"
$ export I_MPI_PIN_DOMAIN=socket
$ export HOROVOD_THREAD_AFFINITY="53,71"
$ export CCL_WORKER_COUNT=2
$ export CCL_WORKER_AFFINITY="16,17,34,35"
$ mpirun -n 4 -ppn 2 -hostfile hosts python ./run_example.py
```
