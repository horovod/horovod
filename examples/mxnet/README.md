# Environment
1) Ubuntu 16.04
2) NVCC (we tried with both CUDA 9.0 and 9.2)
3) CUDA Driver (we tried with both 384.111 and 396.37)
4) GCC 5.4.0
5) Install all dependencies following steps for [standard MXNet install](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU&version=master#)

# Additional dependencies required compared to vanilla MXNet
1) MPI (we tested using OpenMPI 3.1.1 compiled with CUDA-aware)
2) NCCL (we tested with both NCCL 2.1 and 2.2)

# Building MXNet
1) git clone --recursive https://github.com/ctcyang/incubator-mxnet-1.git -b horovod_feature mxnet
2) cd mxnet & cp make/config.mk .
3) Only config.mk + Makefile build chain works. We have not added CMakeLists support yet. Make following changes to config.mk. Note: these are *in addition* to the standard `USE_CUDA = 1` and `USE_CUDA_PATH = [cuda directory]` additions to config.mk when building MXNet for GPU:
  ```
  USE_NCCL = 1
  USE_NCCL_PATH = [directory in which libnccl.so resides]
  USE_MPI_PATH = [root directory in which MPI folders /lib and /include reside]
  ```
4) make -j16
5) pip install -e python

# Building Horovod
1) git clone https://github.com/ctcyang/horovod.git -b mxnet_feature horovod
2) cd horovod
3) sudo PATH=/usr/local/bin:$PATH LD_LIBRARY_PATH=/lib/nccl/cuda-9.0/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/lib/:/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib64/openmpi/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/lib/:/lib/nccl/cuda-9.0/lib HOROVOD_NCCL_HOME=/lib/nccl/cuda-9.0/ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 INCLUDES=['mxnet git directory'/python/mxnet/include] LIBRARY_DIRS=['mxnet git directory'/lib] /home/ubuntu/anaconda3/bin/pip install --upgrade -v --no-cache-dir ['horovod git directory']

An example script is: sudo PATH=/usr/local/bin:$PATH LD_LIBRARY_PATH=/lib/nccl/cuda-9.0/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/lib/:/home/ubuntu/src/cntk/bindings/python/cntk/libs:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib64/openmpi/lib/:/usr/local/cuda/lib64:/usr/local/lib:/usr/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/mpi/lib:/lib/:/lib/nccl/cuda-9.0/lib HOROVOD_NCCL_HOME=/lib/nccl/cuda-9.0/ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 INCLUDES=/home/ubuntu/mxnet/python/mxnet/include LIBRARY_DIRS=/home/ubuntu/mxnet/lib /home/ubuntu/anaconda3/bin/pip install --upgrade -v --no-cache-dir /home/ubuntu/horovod/

# Running
You can run the synthetic benchmark by doing (tested using OpenMPI 3.1.1 on AWS p3.16xlarge instances):

```mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python mxnet_imagenet_resnet50.py --benchmark 1 --batch-size=128 --network resnet-v1 --num-layers=50 --num-epochs 1 --kv-store None --dtype float32 --gpus 0```

Note 1: KVStore must be set to None.
Note 2: the use of MXNET_USE_OPERATOR_TUNING=0 flag to disable OpenMP tuning. If this flag is not included, then starting up 8 MXNet processes will take upwards of 2 minutes. We find disabling this tuning does not affect performance.

To run on Imagenet data:

```mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python mxnet_imagenet_resnet50.py --batch-size=128 --network resnet-v1 --num-layers=50 --num-epochs 1 --kv-store None --dtype float32 --gpus 0 --data-nthreads 4```

# Testing
The following Horovod unit tests do not pass:
  * tests that check Horovod+MXNet throws the correct error if the user passes in NDArrays that differ in size
  * test for average variant of Horovod Allreduce (i.e. that performs a divide by number of workers). We haven’t spent time debugging the reason

To run tests, we did:

```mpirun -np 8 --hostfile ~/host_file --bind-to none --map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 -x LD_LIBRARY_PATH -x PATH -x MXNET_USE_OPERATOR_TUNING=0 -mca pml ob1 -mca btl ^openib python test_mxnet.py```

# Troubleshooting
Two common errors that happen are:
1. `OSError: libmxnet.so: cannot open shared object file: No such file or directory`

Solution: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/master/lib`

2. `OSError: /home/ubuntu/horovod/horovod/mxnet/mpi_lib.cpython-36m-x86_64-linux-gnu.so: cannot open shared object file: No such file or directory`

Solution: launch Python from another directory
