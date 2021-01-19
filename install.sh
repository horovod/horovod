#!/bin/bash
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_GLOO=1

python setup.py install
