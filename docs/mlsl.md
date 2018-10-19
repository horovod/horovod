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
$ source <installdir_MPI>/intel64/bin/mpivars.sh
```

3. Install the `horovod` pip package.

```bash
$ python setup.py build
$ python setup.py install
```

### Advanced: You can specify the affinity for BackgroundThread with the HVD_MLSL_BGT_AFFINITY environment variable.