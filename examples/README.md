# Horovod launcher examples

This is a **launcher-only** build of Horovod. `horovodrun` (or the Python API
`horovod.runner.run`) spawns one process per slot for single-node or multi-node
jobs; distributed communication inside the launched processes is handled by your
own framework code (e.g. PyTorch `torch.distributed`). The TensorFlow / PyTorch /
MXNet Horovod integrations and the C++ allreduce core have been removed, so there
is nothing to compile and `pip install` is a pure-Python install.

## PyTorch `torch.distributed`

See [`pytorch_torch_distributed.py`](pytorch_torch_distributed.py) for a complete,
runnable DDP example (synthetic data, no dataset download).

Single node, one process per GPU, using the pure-Python Gloo launcher:

    horovodrun -np 4 --gloo python examples/pytorch_torch_distributed.py

Single node, launched through an external `mpirun`:

    horovodrun -np 4 --mpi python examples/pytorch_torch_distributed.py

Multi-node with the Gloo launcher (requires passwordless SSH between hosts):

    horovodrun -np 8 -H host1:4,host2:4 --gloo python examples/pytorch_torch_distributed.py

### Environment-variable mapping

The example maps the launcher-provided variables to the `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT` variables expected by
`torch.distributed.init_process_group(init_method="env://")`. Which source variables
are present depends on the launcher:

| torch.distributed | `--gloo` launcher            | `--mpi` (mpirun)             |
|-------------------|------------------------------|------------------------------|
| `RANK`            | `HOROVOD_RANK`               | `OMPI_COMM_WORLD_RANK`       |
| `WORLD_SIZE`      | `HOROVOD_SIZE`               | `OMPI_COMM_WORLD_SIZE`       |
| `LOCAL_RANK`      | `HOROVOD_LOCAL_RANK`         | `OMPI_COMM_WORLD_LOCAL_RANK` |
| `MASTER_ADDR`     | `HOROVOD_GLOO_RENDEZVOUS_ADDR` | defaults to `127.0.0.1`    |
| `MASTER_PORT`     | any free port (default `29500`) | any free port (default `29500`) |

Notes:

- `MASTER_ADDR` can reuse `HOROVOD_GLOO_RENDEZVOUS_ADDR` because the driver IP is
  reachable by every rank. For multi-node MPI there is no rendezvous variable, so
  set `MASTER_ADDR` (and `MASTER_PORT`) yourself.
- `MASTER_PORT` must be **different** from `HOROVOD_GLOO_RENDEZVOUS_PORT` — the
  launcher's rendezvous server already occupies that port. Override `MASTER_PORT`
  in the environment if the default (`29500`) is not free.
