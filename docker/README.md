# Horovod Docker Images

Often installing Horovod on bare metal can be difficult if your environment is not setup
correctly with CUDA, MPI, G++, CMake, etc. These Docker images are provided to simplify
the onboarding process for new users, and can serve as a starting point for building your
own runtime environment.

## Repositories

Separate images are provided for different Horovod configurations, and are published
to separate repos in DockerHub.

* `horovod/horovod` Horovod built with CUDA support and packaged with the latest stable TensorFlow, PyTorch, MXNet, 
  and Spark releases
* `horovod/horovod-cpu` Horovod built for CPU training and packaged with the latest stable TensorFlow, PyTorch, MXNet, 
  and Spark releases
* `horovod/horovod-ray` Horoovd built with CUDA support from the latest 
  [ray-project/ray:nightly-gpu](https://github.com/ray-project/ray) and packaged with the latest stable 
  TensorFlow and PyTorch releases

## Image Tags

* `master` - built from Horovod's `master` branch
* `nightly` - nightly build of Horovod
* `sha-<commit point>` - version of Horovod at designated git sha1 7-character commit point

## Building Custom Images

Build arguments are provided to allow the user to build Horovod against custom versions of various frameworks,
including:

* `TENSORFLOW_VERSION` - version of `tensorflow` pip package to install
* `PYTORCH_VERSION` - version of `torch` pip package to install
* `PYTORCH_LIGHTNING_VERSION` - version of `pytorch_lightning` pip package to install
* `TORCHVISION_VERSION` - version of `torchvision` pip package to install
* `MXNET_VERSION` - version of `mxnet` pip package to install
* `CUDNN_VERSION` - version of `libcudnn` apt package to install (only for `horovod` image)
* `NCCL_VERSION` - version of `libnccl` apt package to install (only for `horovod` image)
* `CUDA_DOCKER_VERSION` - tag of the `nvidia/cuda` image to build from (only for `horovod` image)
* `RAY_DOCKER_VERSION` - tag of the `rayproject/ray` GPU image to build from (only for `horovod-ray` image)

Building the Docker images should be run from the root Horovod directory. For example:

```
docker build \
    --build-arg TENSORFLOW_VERSION=2.3.1 \
    --build-arg PYTORCH_VERSION=1.7.0+cu110 \
    -f docker/horovod/Dockerfile .
```

## Running Containers

See the [Horovod in Docker](../docs/docker.rst) documentation for guidance on running these Docker images, and
[Horovod on Ray](../docs/ray.rst) for usage with Ray.
