#!/bin/bash
cd test && pip3 uninstall -y horovod && cd ..
rm -r ./build
python3 setup.py clean
HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_CUDA_HOME=/usr/local/cuda-10.0 HOROVOD_CUDA_INCLUDE=/usr/local/cuda-10.0/include python3 setup.py install
