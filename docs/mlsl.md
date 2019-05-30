## Horovod with Intel(R) MLSL

To use Horovod with the Intel(R) Machine Learning Scaling Library (Intel(R) MLSL), follow the steps below.

1. Install [Intel MLSL](https://github.com/intel/MLSL).

To install Intel MLSL, follow [these steps](https://github.com/intel/MLSL/blob/master/README.md).

Source mlslvars.sh to start using Intel MLSL. Two modes are available: `process` (default)
and `thread`. Use the `thread` mode if you are going to set more than zero MLSL servers via MLSL_NUM_SERVERS
environment variable.

```bash
$ source <install_dir>/intel64/bin/mlslvars.sh [mode]
```

2. Install the [Intel(R) MPI Library](https://software.intel.com/en-us/mpi-library).

To install the Intel MPI Library, follow [these steps](https://software.intel.com/en-us/mpi-library/documentation/get-started).

Source the mpivars.sh script to establish the proper environment settings.

```bash
$ source <installdir_MPI>/intel64/bin/mpivars.sh release_mt
```

3. Install Horovod from source code.

```bash
$ python setup.py build
$ python setup.py install
```

**Advanced:** You can specify the affinity for BackgroundThread with the HOROVOD_MLSL_BGT_AFFINITY environment variable.
See the instructions below.

Set Horovod background thread affinity:
```bash
$ export HOROVOD_MLSL_BGT_AFFINITY=c0
```
where c0 is a core ID to attach background thread to.

Set the number of MLSL servers:
```bash
$ export MLSL_NUM_SERVERS=X
```
where X is the number of cores you’d like to dedicate for driving communication. This means that for every rank there are X MLSL
servers available.

Set MLSL servers affinity:
```bash
$ export MLSL_SERVER_AFFINITY=c1,c2,..,cX
```
where c1,c2,..,cX are core IDs dedicated to MLSL servers (uses X ‘last’ cores by default). This variable sets affinity for all
MLSL servers (MLSL_NUM_SERVERS * Number of ranks per node) that are available for all the ranks running on one node.
