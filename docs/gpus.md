## Horovod on GPU

To use Horovod on GPU, read the options below and see which one applies to you best.

### Have GPUs?

In most situations, using NCCL 2 will significantly improve performance over the CPU version.  NCCL 2 provides the *allreduce*
operation optimized for NVIDIA GPUs and a variety of networking devices, such as RoCE or InfiniBand.

1. Install [NCCL 2](https://developer.nvidia.com/nccl).

Steps to install NCCL 2 are listed [here](http://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html).

If you have installed NCCL 2 using the `nccl-<version>.txz` package, you should add the library path to `LD_LIBRARY_PATH`
environment variable or register it in `/etc/ld.so.conf`.

```bash
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nccl-<version>/lib
```

2. (Optional) If you're using an NVIDIA Tesla GPU and NIC with GPUDirect RDMA support, you can further speed up NCCL 2
by installing an [nv_peer_memory](http://www.mellanox.com/page/products_dyn?product_family=116) driver.

[GPUDirect](https://developer.nvidia.com/gpudirect) allows GPUs to transfer memory among each other without CPU
involvement, which significantly reduces latency and load on CPU.  NCCL 2 is able to use GPUDirect automatically for
*allreduce* operation if it detects it.

3. Install [Open MPI](https://www.open-mpi.org/) or another MPI implementation.

Steps to install Open MPI are listed [here](https://www.open-mpi.org/faq/?category=building#easy-build).

**Note**: Open MPI 3.1.3 has an issue that may cause hangs.  It is recommended
to downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

4. Install the `horovod` pip package.

If you have installed NCCL 2 using the `nccl-<version>.txz` package, you should specify the path to NCCL 2 using the `HOROVOD_NCCL_HOME`
environment variable.

```bash
$ HOROVOD_NCCL_HOME=/usr/local/nccl-<version> HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
```

If you have installed NCCL 2 using the Ubuntu package, you can simply run:

```bash
$ HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
```

**Note**: Some models with a high computation to communication ratio benefit from doing allreduce on CPU, even if a
GPU version is available. To force allreduce to happen on CPU, pass `device_dense='/cpu:0'` to `hvd.DistributedOptimizer`:

```python
opt = hvd.DistributedOptimizer(opt, device_dense='/cpu:0')
```

### Advanced: Have a proprietary MPI implementation with GPU support optimized for your network?

This section is only relevant if you have a proprietary MPI implementation with GPU support, i.e. not Open MPI or MPICH.
Most users should follow one of the sections above.

If your MPI vendor's implementation of *allreduce* operation on GPU is faster than NCCL 2, you can configure Horovod to
use it instead:

```bash
$ HOROVOD_GPU_ALLREDUCE=MPI pip install --no-cache-dir horovod
```

Additionally, if your MPI vendor's implementation supports *allgather* and *broadcast* operations on GPU, you can
configure Horovod to use them as well:

```bash
$ HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_ALLGATHER=MPI HOROVOD_GPU_BROADCAST=MPI pip install --no-cache-dir horovod
```

**Note**: Allgather allocates an output tensor which is proportionate to the number of processes participating in the
training.  If you find yourself running out of GPU memory, you can force allgather to happen on CPU by passing
`device_sparse='/cpu:0'` to `hvd.DistributedOptimizer`:

```python
opt = hvd.DistributedOptimizer(opt, device_sparse='/cpu:0')
```
