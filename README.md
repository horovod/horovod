# Horovod

[![Build Status](https://travis-ci.org/uber/horovod.svg?branch=master)](https://travis-ci.org/uber/horovod) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<p align="center"><img src="https://user-images.githubusercontent.com/16640218/31681470-2a7968ee-b32c-11e7-88d8-3f51c1457c1a.png" alt="Logo" width="200"/></p>

Horovod is a distributed training framework for TensorFlow. The goal of Horovod is to make distributed Deep Learning
fast and easy to use.

## Why not traditional Distributed TensorFlow?

The primary motivation for this project is to make it easy to take a single-GPU TensorFlow program and successfully train
it on many GPUs faster. This has two aspects:

1. How much modifications does one have to make to a program to make it distributed, and how easy is it to run it.
2. How much faster would it run in distributed mode?

Internally at Uber we found the MPI model to be much more straightforward and require far less code changes than the
Distributed TensorFlow with parameter servers. See the [Usage](#usage) section for more details.

In addition to being easy to use, Horovod is fast. Below is a chart representing the benchmark that was done on 32
servers with 4 Pascal GPUs each connected by RoCE-capable 25 Gbit/s network:
  
![128-GPU Benchmark](https://user-images.githubusercontent.com/16640218/31681220-7453e760-b32b-11e7-9ba3-6d01f83b7748.png)

Horovod achieves 90% scaling efficiency for both Inception V3 and ResNet-101, and 79% scaling efficiency for VGG-16.
See the [Benchmarks](docs/benchmarks.md) page to find out how to reproduce these numbers.

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

4. Add `hvd.BroadcastGlobalVariablesHook(0)` to broadcast initial variable states from rank 0 to all other processes.
    This is necessary to ensure consistent initialization of all workers when training is started with random weights or
    restored from a checkpoint. Alternatively, if you're not using `MonitoredTrainingSession`, you can simply execute
    the `hvd.broadcast_global_variables` op after global variables have been initialized.

5. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.
    This can be accomplished by passing `checkpoint_dir=None` to `tf.train.MonitoredTrainingSession` if
    `hvd.rank() != 0`.

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

# Save checkpoints only on worker 0 to prevent other workers from corrupting them.
checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

# The MonitoredTrainingSession takes care of session initialization,
# restoring from a checkpoint, saving to a checkpoint, and closing when done
# or an error occurs.
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       config=config,
                                       hooks=hooks) as mon_sess:
  while not mon_sess.should_stop():
    # Perform synchronous training.
    mon_sess.run(train_op)
```

## Running Horovod

The example commands below show how to run distributed training. See the [Running Horovod](docs/running.md)
page for more instructions, including RoCE/InfiniBand tweaks and tips for dealing with hangs.

1. To run on a machine with 4 GPUs:

```bash
$ mpirun -np 4 \
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    python train.py
```

2. To run on 4 machines with 4 GPUs each:

```bash
$ mpirun -np 16 \
    -bind-to none -oversubscribe \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
    -H server1:4,server2:4,server3:4,server4:4 \
    python train.py
```

## Keras

Horovod supports Keras and regular TensorFlow in similar ways.

See full training [simple](examples/keras_mnist.py) and [advanced](examples/keras_mnist_advanced.py) examples.

**Note**: Keras 2.0.9 has a [known issue](https://github.com/fchollet/keras/issues/8353) that makes each worker allocate
all GPUs on the server, instead of the GPU assigned by the *local rank*. If you have multiple GPUs per server, upgrade
to Keras 2.1.0 (when it's available), or downgrade to Keras 2.0.8.

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

## Guides

1. Run distributed training in Microsoft Azure using [Batch AI and Horovod](https://github.com/Azure/BatchAI/tree/master/recipes/Horovod).

## Troubleshooting

See the [Troubleshooting](docs/troubleshooting.md) page and please submit the [ticket](https://github.com/uber/horovod/issues/new)
if you can't find an answer.

### Publications

1. Sergeev, A., Del Balso, M. (2017) *Meet Horovod: Uber’s Open Source Distributed Deep Learning Framework for TensorFlow*.
Retrieved from [https://eng.uber.com/horovod/](https://eng.uber.com/horovod/)
2. Sergeev, A. (2017) *Horovod - Distributed TensorFlow Made Easy*. Retrieved from
[https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy](https://www.slideshare.net/AlexanderSergeev4/horovod-distributed-tensorflow-made-easy)

### References

1. Gibiansky, A. (2017). *Bringing HPC Techniques to Deep Learning*. Retrieved from
[http://research.baidu.com/bringing-hpc-techniques-deep-learning/](http://research.baidu.com/bringing-hpc-techniques-deep-learning/)
