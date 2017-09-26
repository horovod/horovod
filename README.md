# Horovod

[![Build Status](https://travis-ci.org/uber/horovod.svg?branch=master)](https://travis-ci.org/uber/horovod) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Horovod is a distributed training framework for TensorFlow. The goal of Horovod is to make distributed Deep Learning
fast and easy to use.

## Why not traditional Distributed TensorFlow?

The primary motivation for this project is to make it easy to take a single-GPU TensorFlow program and successfully train
it on many GPUs faster. This has two aspects:

1. How much modifications does one have to make to a program to make it distributed, and how easy is it to run it.
2. How much faster would it run in distributed mode?

Internally at Uber we found that it's much easier for people to understand an MPI model that requires minimal changes to
source code than to understand how to set up regular Distributed TensorFlow.

To give some perspective on that, [this commit](https://github.com/alsrgv/benchmarks/commit/86bf2f9269dbefb4e57a8b66ed260c8fab84d6c7) 
into our fork of TF Benchmarks shows how much code can be removed if one doesn't need to worry about towers and manually
averaging gradients across them, `tf.Server()`, `tf.ClusterSpec()`, `tf.train.SyncReplicasOptimizer()`, 
`tf.train.replicas_device_setter()` and so on. If none of these things makes sense to you - don't worry, you don't have to 
learn them if you use Horovod.

In addition to being easy to use, Horovod is fast. We have done two benchmarks which demonstrate that Horovod scales very well.

The first benchmark was done on 4 servers with 4 Pascal GPUs each connected by RoCE-capable 25 Gbit/s network:

| Setup                                     |     Inception V3    |      ResNet-101     |        VGG-16       |
|-------------------------------------------|:-------------------:|:-------------------:|:-------------------:|
| Baseline single-GPU (batch size=64)       |               134.4 |               119.4 |               130.9 |
|                 On 16 GPUs                |                   x |                   x |                   x |
| Distributed TensorFlow                    |     1,345.8 (10.0x) |        959.6 (8.0x) |         74.7 (0.6x) |
| Distributed TensorFlow (variables on CPU) |     1,576.4 (11.7x) |      1,168.8 (9.8x) |         79.5 (0.6x) |
| TCP Horovod (allreduce on CPU)            | **2,073.3 (15.4x)** |     1,338.3 (11.2x) |        616.8 (4.7x) |
| RDMA Horovod (allreduce on CPU)           | **2,073.1 (15.4x)** |     1,446.3 (12.1x) |        618.0 (4.7x) |
| TCP Horovod (allreduce on GPU with NCCL)  |     1,990.7 (14.8x) |     1,685.1 (14.1x) |     1,308.7 (10.0x) |
| RDMA Horovod (allreduce on GPU with NCCL) |     2,022.6 (15.0x) | **1,746.2 (14.6x)** | **1,787.4 (13.7x)** |

The second benchmark was done on 16 servers with 4 Pascal GPUs each connected by plain 40 Gbit/s network:

| Setup                                     |     Inception V3    |      ResNet-101     |        VGG-16       |
|-------------------------------------------|:-------------------:|:-------------------:|:-------------------:|
| Baseline single-GPU (batch size=64)       |               148.8 |               136.0 |               149.6 |
|                 On 64 GPUs                |                   x |                   x |                   x |
| Distributed TensorFlow                    |     4,225.3 (28.4x) |     2,996.0 (22.0x) |         97.0 (0.6x) |
| Distributed TensorFlow (variables on CPU) |     5,297.4 (35.6x) |     4,269.2 (31.4x) |        100.8 (0.7x) |
| TCP Horovod (allreduce on CPU)            |     6,549.6 (44.0x) |     3,761.6 (27.7x) |      1,462.6 (9.8x) |
| TCP Horovod (allreduce on GPU with NCCL)  | **7,932.1 (53.3x)** | **7,741.6 (56.9x)** | **6,084.2 (40.7x)** |

While installing MPI and NCCL itself may seem like an extra hassle, it only needs to be done once by the team dealing
with infrastructure, while everyone else in the company who builds the models can enjoy the simplicity of training them at
scale.

## Install

To install Horovod:

1. Install [Open MPI](https://www.open-mpi.org/) or another MPI implementation.

Steps to install Open MPI are listed [here](https://www.open-mpi.org/faq/?category=building#easy-build).

2. Install the `horovod` pip package.

```bash
$ pip install horovod
```

This basic installation is good for laptops and for getting to know Horovod.
If you're installing Horovod on a server with GPUs, read the [Horovod on GPU](docs/gpus.md) page.

## Concepts

Horovod core principles are based on [MPI](http://mpi-forum.org/) concepts such as *size*, *rank*,
*local rank*, *allreduce*, *allgather* and *broadcast*. See [here](docs/concepts.md) for more details.

## Usage

To use Horovod, make the following additions to your program:

1. Run `hvd.init()`.

2. Pin a server GPU to be used by this process using `config.gpu_options.visible_device_list`.
    With the typical setup of one GPU per process, this can be set to *local rank*. In that case, the first process on 
    the server will be allocated the first GPU, second process will be allocated the second GPU and so forth.

3. Wrap optimizer in `hvd.DistributedOptimizer`.  The distributed optimizer delegates gradient computation
    to the original optimizer, averages gradients using *allreduce* or *allgather*, and then applies those averaged
    gradients.

4. Add `hvd.BroadcastGlobalVariablesHook(0)` to broadcast initial variable states from rank 0 to all other
    processes. Alternatively, if you're not using `MonitoredTrainingSession`, you can simply execute the
    `hvd.broadcast_global_variables` op after global variables have been initialized.

Example (see the [examples](examples/) directory for full training examples):

```python
import tensorflow as tf
import horovod.tensorflow as hvd


# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

# Build model...
loss = ...
opt = tf.train.AdagradOptimizer(0.01)

# Add Horovod Distributed Optimizer
opt = hvd.DistributedOptimizer(opt)

# Add hook to broadcast variables from rank 0 to all other processes during
# initialization.
hooks = [hvd.BroadcastGlobalVariablesHook(0)]

# Make training operation
train_op = opt.minimize(loss)

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir="/tmp/train_logs",
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```

To run on a machine with 4 GPUs:

```bash
$ mpirun -np 4 python train.py
```

To run on 4 machines with 4 GPUs each using Open MPI:

```bash
$ mpirun -np 16 -x LD_LIBRARY_PATH -H server1:4,server2:4,server3:4,server4:4 python train.py
```

If you're using Open MPI and you have RoCE or InfiniBand, we found this custom RDMA queue configuration to help
performance a lot:

```bash
$ mpirun -np 16 -x LD_LIBRARY_PATH -mca btl_openib_receive_queues P,128,32:P,2048,32:P,12288,32:P,131072,32 -H server1:4,server2:4,server3:4,server4:4 python train.py
```

Check your MPI documentation for arguments to the `mpirun` command on your system.

## Inference

Learn how to optimize your model for inference and remove Horovod operations from the graph [here](docs/inference.md).

## Tensor Fusion

One of the unique things about Horovod is its ability to interleave communication and computation coupled with the ability
to batch small *allreduce* operations, which results in improved performance. We call this batching feature Tensor Fusion.

See [here](docs/tensor-fusion.md) for full details and tweaking instructions.

## Analyzing Horovod Performance

Horovod has the ability to record the timeline of its activity, called Horovod Timeline.

![Horovod Timeline](https://user-images.githubusercontent.com/16640218/29735271-9e148da0-89ac-11e7-9ae0-11d7a099ac89.png)

See [here](docs/timeline.md) for full details and usage instructions.

## Troubleshooting

See the [Troubleshooting](docs/troubleshooting.md) page and please submit the [ticket](https://github.com/uber/horovod/issues/new)
if you can't find an answer.

### References

1. Gibiansky, A. (2017). *Bringing HPC Techniques to Deep Learning*. Retrieved from
[http://research.baidu.com/bringing-hpc-techniques-deep-learning/](http://research.baidu.com/bringing-hpc-techniques-deep-learning/)
