# horovod源码阅读笔记

这次阅读采用的horovod的commit id为a9dea74abc1f0b8e81cd2b6dd9fe81e2c4244e39

## 先导知识

horovod是一个python和C++混编的工程，最上层借助的工具是python的setuptools模块。之前对于这个模块的使用并不是很熟悉，因此想借此机会来学习一下setuptools这个工具的使用。

## 编译系统

编译参数如下，只基于mpi集合库编译pytorch的horovod。

```shell
#!/bin/bash
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_MPI=1
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_GLOO=1
python3 setup.py install
```

参与编译的源文件分为两部分，一部分为common模块下公共代码，另外一部分torch模块下的适配代码。

common部分代码

```
/mnt/horovod/horovod/common/common.cc
/mnt/horovod/horovod/common/controller.cc
/mnt/horovod/horovod/common/fusion_buffer_manager.cc
/mnt/horovod/horovod/common/group_table.cc
/mnt/horovod/horovod/common/half.cc
/mnt/horovod/horovod/common/logging.cc
/mnt/horovod/horovod/common/message.cc
/mnt/horovod/horovod/common/operations.cc
/mnt/horovod/horovod/common/parameter_manager.cc
/mnt/horovod/horovod/common/response_cache.cc
/mnt/horovod/horovod/common/stall_inspector.cc
/mnt/horovod/horovod/common/thread_pool.cc
/mnt/horovod/horovod/common/timeline.cc
/mnt/horovod/horovod/common/tensor_queue.cc
/mnt/horovod/horovod/common/ops/collective_operations.cc
/mnt/horovod/horovod/common/ops/operation_manager.cc
/mnt/horovod/horovod/common/optim/bayesian_optimization.cc
/mnt/horovod/horovod/common/optim/gaussian_process.cc
/mnt/horovod/horovod/common/utils/env_parser.cc
/mnt/horovod/horovod/common/mpi/mpi_context.cc
/mnt/horovod/horovod/common/mpi/mpi_controller.cc
/mnt/horovod/horovod/common/ops/mpi_operations.cc
/mnt/horovod/horovod/common/ops/adasum/adasum_mpi.cc
/mnt/horovod/horovod/common/ops/adasum_mpi_operations.cc
```

torch部分代码

```
/mnt/horovod/horovod/torch/handle_manager.cc
/mnt/horovod/horovod/torch/ready_event.cc
/mnt/horovod/horovod/torch/cuda_util.cc
/mnt/horovod/horovod/torch/mpi_ops_v2.cc
/mnt/horovod/horovod/torch/adapter_v2.cc
```

