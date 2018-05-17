## Troubleshooting

### Import TensorFlow failed during installation

1. Is TensorFlow installed?

If you see the error message below, it means that TensorFlow is not installed.  Please install TensorFlow before installing
Horovod.

```
error: import tensorflow failed, is it installed?

Traceback (most recent call last):
  File "/tmp/pip-OfE_YX-build/setup.py", line 29, in fully_define_extension
    import tensorflow as tf
ImportError: No module named tensorflow
```

2. Are the CUDA libraries available?

If you see the error message below, it means that TensorFlow cannot be loaded.  If you're installing Horovod into a container
on a machine without GPUs, you may use CUDA stub drivers to work around the issue.

```
error: import tensorflow failed, is it installed?

Traceback (most recent call last):
  File "/tmp/pip-41aCq9-build/setup.py", line 29, in fully_define_extension
    import tensorflow as tf
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/__init__.py", line 24, in <module>
    from tensorflow.python import *
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/__init__.py", line 49, in <module>
    from tensorflow.python import pywrap_tensorflow
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 52, in <module>
    raise ImportError(msg)
ImportError: Traceback (most recent call last):
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/pywrap_tensorflow.py", line 41, in <module>
    from tensorflow.python.pywrap_tensorflow_internal import *
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 28, in <module>
    _pywrap_tensorflow_internal = swig_import_helper()
  File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/pywrap_tensorflow_internal.py", line 24, in swig_import_helper
    _mod = imp.load_module('_pywrap_tensorflow_internal', fp, pathname, description)
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
```

To use CUDA stub drivers:

```bash
# temporary add stub drivers to ld.so.cache
$ ldconfig /usr/local/cuda/lib64/stubs

# install Horovod, add other HOROVOD_* environment variables as necessary
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod

# revert to standard libraries
$ ldconfig
```

### MPI is not found during installation

1. Is MPI in PATH?

If you see the error message below, it means `mpicxx` was not found in PATH. Typically `mpicxx` is located in the same
directory as `mpirun`. Please add a directory containing `mpicxx` to PATH before installing Horovod.

```
error: mpicxx -show failed, is mpicxx in $PATH?

Traceback (most recent call last):
  File "/tmp/pip-dQ6A7a-build/setup.py", line 70, in get_mpi_flags
    ['mpicxx', '-show'], universal_newlines=True).strip()
  File "/usr/lib/python2.7/subprocess.py", line 566, in check_output
    process = Popen(stdout=PIPE, *popenargs, **kwargs)
  File "/usr/lib/python2.7/subprocess.py", line 710, in __init__
    errread, errwrite)
  File "/usr/lib/python2.7/subprocess.py", line 1335, in _execute_child
    raise child_exception
OSError: [Errno 2] No such file or directory
```

To use custom MPI directory:

```bash
$ export PATH=$PATH:/path/to/mpi/bin
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
```

2. Are MPI libraries added to `$LD_LIBRARY_PATH` or `ld.so.conf`?

If you see the error message below, it means `mpicxx` was not able to load some of the MPI libraries. If you recently
installed MPI, make sure that the path to MPI libraries is present the `$LD_LIBRARY_PATH` environment variable, or in the
`/etc/ld.so.conf` file.

```
mpicxx: error while loading shared libraries: libopen-pal.so.40: cannot open shared object file: No such file or directory
error: mpicxx -show failed (see error below), is MPI in $PATH?
Note: If your version of MPI has a custom command to show compilation flags, please specify it with the HOROVOD_MPICXX_SHOW environment variable.

Traceback (most recent call last):
File "/tmp/pip-build-wrtVwH/horovod/setup.py", line 107, in get_mpi_flags
shlex.split(show_command), universal_newlines=True).strip()
File "/usr/lib/python2.7/subprocess.py", line 574, in check_output
raise CalledProcessError(retcode, cmd, output=output)
CalledProcessError: Command '['mpicxx', '-show']' returned non-zero exit status 127
```

If you have installed MPI in a user directory, you can add the MPI library directory to `$LD_LIBRARY_PATH`:

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mpi/lib
```

If you have installed MPI in a non-standard system location (i.e. not `/usr` or `/usr/local`), you should add it to the
`/etc/ld.so.conf` file:

```bash
$ echo /path/to/mpi/lib | sudo tee -a /etc/ld.so.conf
```

Additionally, if you have installed MPI in a system location, you should run `sudo ldconfig` after installation to
register libraries in the cache:

```bash
$ sudo ldconfig
```

### Error during installation: invalid conversion from ‘const void*’ to ‘void*’ [-fpermissive]

If you see the error message below, it means that your MPI is likely outdated. We recommend installing
[Open MPI >=3.0.0](https://www.open-mpi.org/faq/?category=building#easy-build).

**Note**: Prior to installing a new version of Open MPI, don't forget to remove your existing MPI installation.

```
horovod/tensorflow/mpi_ops.cc: In function ‘void horovod::tensorflow::{anonymous}::PerformOperation(horovod::tensorflow::{anonymous}::TensorTable&, horovod::tensorflow::MPIResponse)’:
horovod/tensorflow/mpi_ops.cc:802:79: # error: invalid conversion from ‘const void*’ to ‘void*’ [-fpermissive]
                                  recvcounts, displcmnts, dtype, MPI_COMM_WORLD);
                                                                               ^
In file included from horovod/tensorflow/mpi_ops.cc:38:0:
/usr/anaconda2/include/mpi.h:633:5: error:   initializing argument 1 of ‘int MPI_Allgatherv(void*, int, MPI_Datatype, void*, int*, int*, MPI_Datatype, MPI_Comm)’ [-fpermissive]
 int MPI_Allgatherv(void* , int, MPI_Datatype, void*, int *, int *, MPI_Datatype, MPI_Comm);
     ^
horovod/tensorflow/mpi_ops.cc:1102:45: error: invalid conversion from ‘const void*’ to ‘void*’ [-fpermissive]
                               MPI_COMM_WORLD))
                                             ^
```

### Error during installation: fatal error: pyconfig.h: No such file or directory

If you see the error message below, it means that you need to install Python headers.

```
build/horovod/torch/mpi_lib/_mpi_lib.c:22:24: fatal error: pyconfig.h: No such file or directory
 #  include <pyconfig.h>
                        ^
compilation terminated.
```

You can do this by installing a `python-dev` or `python3-dev` package.  For example, on a Debian or Ubuntu system:

```bash
$ sudo apt-get install python-dev
```

### NCCL 2 is not found during installation

If you see the error message below, it means NCCL 2 was not found in the standard libraries location. If you have a directory
where you installed NCCL 2 which has both `include` and `lib` directories containing `nccl.h` and `libnccl.so` 
respectively, you can pass it via `HOROVOD_NCCL_HOME` environment variable. Otherwise you can specify them separately
via `HOROVOD_NCCL_INCLUDE` and `HOROVOD_NCCL_LIB` environment variables.

```
build/temp.linux-x86_64-2.7/test_compile/test_nccl.cc:1:18: fatal error: nccl.h: No such file or directory
 #include <nccl.h>
                  ^
compilation terminated.
error: NCCL 2.0 library or its later version was not found (see error above).
Please specify correct NCCL location via HOROVOD_NCCL_HOME environment variable or combination of HOROVOD_NCCL_INCLUDE and HOROVOD_NCCL_LIB environment variables.

HOROVOD_NCCL_HOME - path where NCCL include and lib directories can be found
HOROVOD_NCCL_INCLUDE - path to NCCL include directory
HOROVOD_NCCL_LIB - path to NCCL lib directory
```

For example:

```bash
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
```

Or:

```bash
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_INCLUDE=/path/to/nccl/include HOROVOD_NCCL_LIB=/path/to/nccl/lib pip install --no-cache-dir horovod
```

### Pip install: no such option: --no-cache-dir

If you see the error message below, it means that your version of pip is out of date. You can remove the `--no-cache-dir` flag
since your version of pip does not do caching. The `--no-cache-dir` flag is added to all examples to ensure that when you
change Horovod compilation flags, it will be rebuilt from source and not just reinstalled from the pip cache, which is
modern pip's [default behavior](https://pip.pypa.io/en/stable/reference/pip_install/#caching).

```
$ pip install --no-cache-dir horovod

Usage:
  pip install [options] <requirement specifier> ...
  pip install [options] -r <requirements file> ...
  pip install [options] [-e] <vcs project url> ...
  pip install [options] [-e] <local project path> ...
  pip install [options] <archive url/path> ...

no such option: --no-cache-dir
```

For example:

```bash
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
```

### ncclCommInitRank failed: unhandled cuda error

If you see the error message below during the training, it means that NCCL is not able to initialize correctly. You can
set the `NCCL_DEBUG` environment variable to `INFO` to have NCCL print debugging information which may reveal the reason.

```
UnknownError (see above for traceback): ncclCommInitRank failed: unhandled cuda error
         [[Node: training/TFOptimizer/DistributedAdadeltaOptimizer_Allreduce/HorovodAllreduce_training_TFOptimizer_gradients_dense_2_BiasAdd_grad_tuple_control_dependency_1_0 = HorovodAllreduce[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/gpu:0"](training/TFOptimizer/gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1)]]
         [[Node: training/TFOptimizer/DistributedAdadeltaOptimizer/update/_94 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_583_training/TFOptimizer/DistributedAdadeltaOptimizer/update", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
```

For example:

```bash
$ export NCCL_DEBUG=INFO
$ mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    python train.py
```

### ncclAllReduce failed: invalid data type

If you see the error message below during the training, it means that Horovod was linked to the wrong version of NCCL
library.

```
UnknownError (see above for traceback): ncclAllReduce failed: invalid data type
         [[Node: DistributedMomentumOptimizer_Allreduce/HorovodAllreduce_gradients_AddN_2_0 = HorovodAllreduce[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:GPU:0"](gradients/AddN_2)]]
         [[Node: train_op/_653 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_1601_train_op", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:
0"]()]]
```

If you're using Anaconda or Miniconda, you most likely have the `nccl` package installed. The solution is to remove
the package and reinstall Horovod:

```bash
$ conda remove nccl
$ pip uninstall -y horovod
$ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install --no-cache-dir horovod
```

### transport/p2p.cu:431 WARN failed to open CUDA IPC handle : 30 unknown error

If you see the error message below during the training with `-x NCCL_DEBUG=INFO`, it likely means that multiple servers
share the same `hostname`.

```
node1:22671:22795 [1] transport/p2p.cu:431 WARN failed to open CUDA IPC handle : 30 unknown error
```

MPI and NCCL rely on hostnames to distinguish between servers, so you should make sure that every server has a unique
hostname.

### Running out of memory

If you notice that your program is running out of GPU memory and multiple processes
are being placed on the same GPU, it's likely that your program (or its dependencies)
create a `tf.Session` that does not use the `config` that pins specific GPU.

If possible, track down the part of program that uses these additional `tf.Session`s and pass
the same configuration.

Alternatively, you can place following snippet in the beginning of your program to ask TensorFlow
to minimize the amount of memory it will pre-allocate on each GPU:

```python
small_cfg = tf.ConfigProto()
small_cfg.gpu_options.allow_growth = True
with tf.Session(config=small_cfg):
    pass
```

As a last resort, you can **replace** setting `config.gpu_options.visible_device_list`
with different code:

```python
# Pin GPU to be used
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
```

**Note**: Setting `CUDA_VISIBLE_DEVICES` is incompatible with `config.gpu_options.visible_device_list`.

Setting `CUDA_VISIBLE_DEVICES` has additional disadvantage for GPU version - CUDA will not be able to use IPC, which
will likely cause NCCL and MPI to fail.  In order to disable IPC in NCCL and MPI and allow it to fallback to shared
memory, use:
* `export NCCL_P2P_DISABLE=1` for NCCL.
* `--mca btl_smcuda_use_cuda_ipc 0` flag for OpenMPI and similar flags for other vendors.
