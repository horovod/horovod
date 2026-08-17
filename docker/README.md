# Horovod Docker Images (launcher-only)

These images package the **launcher-only** build of Horovod: the `horovodrun` /
`horovod.runner` process launcher, together with Open MPI and passwordless SSH so it
can launch single-node and multi-node jobs. The TensorFlow / PyTorch / MXNet
integrations and the C++ allreduce core have been removed from Horovod, so there is
nothing to compile and the images contain no deep-learning framework. Distributed
communication in the launched processes is handled by your own framework code (for
example PyTorch `torch.distributed`) — install your framework packages on top of
these images as needed.

## Images

* `docker/horovod` — launcher on an NVIDIA CUDA base (for GPU jobs; bring your own
  PyTorch/NCCL).
* `docker/horovod-cpu` — launcher on a plain Ubuntu base (for CPU jobs).

Both images install Open MPI and configure passwordless SSH, which `horovodrun`
needs for multi-node launches.

## Building

Build from the repository root:

```
docker build -f docker/horovod-cpu/Dockerfile -t horovod-launcher-cpu .
docker build -f docker/horovod/Dockerfile -t horovod-launcher .
```

You can build against a specific CUDA base with the `CUDA_DOCKER_VERSION` build
argument (only for the `horovod` image):

```
docker build --build-arg CUDA_DOCKER_VERSION=11.3.1-devel-ubuntu20.04 \
    -f docker/horovod/Dockerfile -t horovod-launcher .
```

## Running

Launch a training script on the local container (Gloo launcher, no MPI required):

```
docker run --rm horovod-launcher-cpu horovodrun -np 2 --gloo python train.py
```

See `examples/` (and `examples/README.md`) for a complete PyTorch
`torch.distributed` example, including the mapping from the `HOROVOD_*` environment
variables set by the launcher to the variables `torch.distributed` expects.

For multi-node runs, `horovodrun` must be able to SSH to every host without a
password; the images already disable strict host-key checking, and you need to
provide a shared SSH key across the containers. See `docs/running.rst`.
