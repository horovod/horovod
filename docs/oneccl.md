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

**Advanced:** You can specify the affinity for BackgroundThread with the HOROVOD_CCL_BGT_AFFINITY environment variable.
See the instructions below.

Set Horovod background thread affinity:
```bash
$ export HOROVOD_CCL_BGT_AFFINITY=c0
```
where c0 is a core ID to attach background thread to.

Set the number of oneCCL workers:
```bash
$ export CCL_WORKER_COUNT=X
```
where X is the number of threads you’d like to dedicate for driving communication. This means that for every rank there are X oneCCL
workers available.

Set oneCCL workers affinity:
```bash
$ export CCL_WORKER_AFFINITY=c1,c2,..,cX
```
where c1,c2,..,cX are core IDs dedicated to oneCCL workers (uses X ‘last’ cores by default). This variable sets affinity for all
oneCCL workers (CCL_WORKER_COUNT * Number of ranks per node) that are available for all the ranks running on one node.
