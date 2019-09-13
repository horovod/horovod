#!/bin/bash
cd test && pip3 uninstall -y horovod && cd ..
rm -r ./build
python3 setup.py clean
HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITH_TENSORFLOW=1 python3 setup.py install
